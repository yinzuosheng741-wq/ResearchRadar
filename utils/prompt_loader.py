from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE_DIR / "prompts"


@lru_cache
def _load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_paper_profile_prompt() -> str:
    return _load_prompt("paper_profile.txt")


def load_cited_qa_prompt() -> str:
    return _load_prompt("cited_qa.txt")


def load_research_agent_prompt() -> str:
    return _load_prompt("research_agent.txt")


def load_research_plan_prompt() -> str:
    return _load_prompt("research_plan.txt")
