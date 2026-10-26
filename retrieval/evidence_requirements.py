"""Conservative boundaries for questions that require body/table evidence."""

from __future__ import annotations

import re

from domain.models import EvidenceChunk

# A method name or high-level conclusion can legitimately appear in an
# abstract. Only explicit body locations and detailed reproducibility requests
# are gated here; this finite grammar is not semantic answer verification.
FULLTEXT_REQUEST = re.compile(
    r"超参数|学习率|训练.{0,8}(?:测试|验证).{0,6}(?:比例|划分)|"
    r"(?:表|公式|附录)\s*[A-ZＡ-Ｚ一二三四五六七八九十\d]|"
    r"补充材料|逐(?:行|列)|"
    r"\b(?:hyperparameters?|learning rate|table\s+\d+|equation\s+\d+|appendix\s+[a-z\d]|supplementary material)\b",
    re.I,
)


def evidence_requirement_gap(question: str, chunks: list[EvidenceChunk]) -> str | None:
    """Explain an abstract-only gap without claiming that full text is sufficient."""
    if chunks and all(chunk.page_number == 0 for chunk in chunks) and FULLTEXT_REQUEST.search(question):
        return "abstract_only_for_fulltext_detail"
    return None
