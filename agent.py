"""
Agentic Gemma 4 E4B - Sistema multi-agente com reasoning
Main Agent orquestra dois SubAgents via Ollama local.
"""

import json
import urllib.request

MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Faz inferência no Gemma 4 via Ollama API."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


# ---------------------------------------------------------------------------
# SubAgents
# ---------------------------------------------------------------------------

def subagent_analyst(task: str) -> str:
    """SubAgent 1 - Analista: decompõe o problema e raciocina passo a passo."""
    system = (
        "Você é um analista especializado em raciocínio lógico. "
        "Ao receber um problema, decomponha-o em partes menores, "
        "raciocine passo a passo (chain-of-thought) e apresente "
        "sua análise de forma estruturada. Responda em português."
    )
    print("  [Analista] Raciocinando...")
    return ollama_generate(task, system=system, temperature=0.3)


def subagent_critic(analysis: str, original_task: str) -> str:
    """SubAgent 2 - Crítico: revisa a análise e aponta falhas ou melhorias."""
    system = (
        "Você é um crítico rigoroso. Receba uma análise feita por outro agente "
        "e verifique se o raciocínio está correto, se há falhas lógicas, "
        "premissas erradas ou pontos que foram ignorados. "
        "Dê uma nota de 1 a 10 para a qualidade do raciocínio. "
        "Responda em português."
    )
    prompt = (
        f"## Problema original\n{original_task}\n\n"
        f"## Análise do Analista\n{analysis}\n\n"
        "Revise essa análise criticamente."
    )
    print("  [Crítico] Revisando...")
    return ollama_generate(prompt, system=system, temperature=0.4)


# ---------------------------------------------------------------------------
# Main Agent (Orquestrador)
# ---------------------------------------------------------------------------

def main_agent(task: str) -> None:
    """Orquestra os subagents e sintetiza a resposta final."""
    print(f"\n{'='*60}")
    print(f"  MAIN AGENT - Orquestrador")
    print(f"{'='*60}")
    print(f"  Tarefa: {task}\n")

    # Etapa 1: SubAgent Analista
    print("─" * 40)
    analysis = subagent_analyst(task)
    print(f"\n  [Analista] Resultado:\n{analysis}\n")

    # Etapa 2: SubAgent Crítico
    print("─" * 40)
    critique = subagent_critic(analysis, task)
    print(f"\n  [Crítico] Resultado:\n{critique}\n")

    # Etapa 3: Síntese final pelo Main Agent
    print("─" * 40)
    print("  [Orquestrador] Sintetizando resposta final...")
    system = (
        "Você é o agente principal. Receba a análise de um analista e a "
        "revisão de um crítico, e produza uma resposta final concisa e "
        "precisa que incorpore os melhores pontos de ambos. "
        "Responda em português."
    )
    synthesis_prompt = (
        f"## Problema\n{task}\n\n"
        f"## Análise do Analista\n{analysis}\n\n"
        f"## Revisão do Crítico\n{critique}\n\n"
        "Com base em ambos, dê a resposta final."
    )
    final = ollama_generate(synthesis_prompt, system=system, temperature=0.5)

    print(f"\n{'='*60}")
    print("  RESPOSTA FINAL")
    print(f"{'='*60}")
    print(final)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Tarefas de teste para reasoning
# ---------------------------------------------------------------------------

TASKS = [
    # Lógica
    "Se todos os gatos são mortais e Sócrates é um gato, Sócrates é mortal? "
    "E se nem todos os gatos forem mortais?",

    # Matemática / raciocínio
    "Um fazendeiro tem 17 ovelhas. Todas menos 9 morrem. Quantas restam?",

    # Raciocínio causal
    "Uma janela foi encontrada quebrada pela manhã. Ao lado havia uma bola de baseball "
    "e pegadas de criança. No entanto, houve uma tempestade forte durante a noite. "
    "O que provavelmente quebrou a janela? Justifique seu raciocínio.",
]


if __name__ == "__main__":
    print("\n🔬 Teste de Reasoning - Gemma 4 E4B (via Ollama)")
    print(f"   Modelo: {MODEL}\n")

    for i, task in enumerate(TASKS, 1):
        print(f"\n{'#'*60}")
        print(f"  TESTE {i}/{len(TASKS)}")
        print(f"{'#'*60}")
        main_agent(task)
