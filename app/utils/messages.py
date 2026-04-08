"""Message normalization and pairing utilities.

Ensures messages sent to the API are valid:
- tool_use blocks always have matching tool_result
- No orphaned tool_results without preceding tool_use
- Consecutive user messages are merged
- System/progress messages are stripped
"""

from __future__ import annotations

from app.models import Message, MessageRole, ToolUseBlock, create_tool_result_message


def normalize_messages_for_api(messages: list[Message]) -> list[Message]:
    """Normalize internal messages for API submission.

    Handles:
    1. Strip system messages (they go in system prompt)
    2. Ensure tool_use/tool_result pairing
    3. Merge consecutive user messages
    4. Remove empty messages
    """
    result: list[Message] = []

    for msg in messages:
        # Strip system messages (handled separately)
        if msg.role == MessageRole.SYSTEM:
            continue

        # Skip empty messages
        if not msg.get_text() and not msg.tool_calls and msg.role != MessageRole.TOOL:
            continue

        result.append(msg)

    # Ensure tool_use/tool_result pairing
    result = _ensure_tool_pairing(result)

    # Merge consecutive user messages
    result = _merge_consecutive_users(result)

    return result


def _ensure_tool_pairing(messages: list[Message]) -> list[Message]:
    """Ensure every tool_use has a matching tool_result and vice versa.

    - Orphaned tool_results (no matching tool_use) are removed
    - Unmatched tool_uses get synthetic error results injected
    """
    # Collect all tool_use IDs from assistant messages
    tool_use_ids: set[str] = set()
    for msg in messages:
        if msg.role == MessageRole.ASSISTANT:
            for tc in msg.tool_calls:
                tool_use_ids.add(tc.id)

    # Collect all tool_result IDs
    tool_result_ids: set[str] = set()
    for msg in messages:
        if msg.role == MessageRole.TOOL and msg.tool_call_id:
            tool_result_ids.add(msg.tool_call_id)

    result = list(messages)

    # Remove orphaned tool_results (no matching tool_use)
    result = [
        msg for msg in result
        if not (msg.role == MessageRole.TOOL and msg.tool_call_id and msg.tool_call_id not in tool_use_ids)
    ]

    # Add synthetic error results for unmatched tool_uses
    unmatched = tool_use_ids - tool_result_ids
    if unmatched:
        # Find the last assistant message with unmatched tool_uses and inject results after it
        for i in range(len(result) - 1, -1, -1):
            if result[i].role == MessageRole.ASSISTANT:
                unmatched_in_msg = [tc for tc in result[i].tool_calls if tc.id in unmatched]
                if unmatched_in_msg:
                    insert_pos = i + 1
                    for tc in unmatched_in_msg:
                        synthetic = create_tool_result_message(
                            tool_call_id=tc.id,
                            content="Tool execution was interrupted.",
                            is_error=True,
                        )
                        result.insert(insert_pos, synthetic)
                        insert_pos += 1
                    break

    return result


def _merge_consecutive_users(messages: list[Message]) -> list[Message]:
    """Merge consecutive user messages into a single message.

    OpenAI requires alternating user/assistant roles (except tool messages).
    Consecutive user messages get their text concatenated.
    """
    if not messages:
        return messages

    result: list[Message] = [messages[0]]

    for msg in messages[1:]:
        prev = result[-1]

        # Merge consecutive user text messages (not tool results)
        if (
            msg.role == MessageRole.USER
            and prev.role == MessageRole.USER
            and msg.tool_call_id is None
            and prev.tool_call_id is None
        ):
            merged_text = prev.get_text()
            new_text = msg.get_text()
            if new_text:
                prev.text = f"{merged_text}\n\n{new_text}" if merged_text else new_text
        else:
            result.append(msg)

    return result


def detect_interrupted_conversation(messages: list[Message]) -> tuple[list[Message], bool]:
    """Detect if a conversation was interrupted mid-response.

    Returns (cleaned_messages, was_interrupted).
    If interrupted, appends a synthetic continuation prompt.
    """
    if not messages:
        return messages, False

    # Find last non-system message
    last_relevant = None
    for msg in reversed(messages):
        if msg.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL):
            last_relevant = msg
            break

    if last_relevant is None:
        return messages, False

    # If last message is user (not tool_result) -> interrupted mid-response
    if last_relevant.role == MessageRole.USER and last_relevant.tool_call_id is None:
        return messages, True

    # If last message is tool_result without a follow-up assistant -> interrupted
    if last_relevant.role == MessageRole.TOOL:
        return messages, True

    return messages, False
