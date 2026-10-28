"""Small bilingual lexical facets; expansions never assert research facts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from retrieval.query_expansion import TERM_GROUPS

FACETS = (
    (("光谱",), ("spectral", "spectra", "spectrum")),
    (("波长",), ("wavelength", "wavelengths")),
    (("间隔", "采样间距"), ("interval", "intervals", "spacing")),
    (("不确定性",), ("uncertainty", "uncertainties")),
    (("方法细节",), ("methodological detail", "method details")),
    (("大气校正",), ("atmospheric correction", "atmospherically correct")),
    (("变量", "属性"), ("variable", "variables", "attribute", "attributes", "parameter", "parameters")),
    (("面积",), ("area", "surface")),
    (("混合像元",), ("mixed pixels", "mixed water", "land components")),
    (("云",), ("cloud", "clouds")),
    (("阴影",), ("shadow", "shadows")),
    (("训练",), ("train", "training")),
    (("划分",), ("split", "partition")),
    (("验证",), ("validation", "validated", "evaluation")),
    (("波段",), ("band", "bands")),
    (("质量控制",), ("quality control", "QC")),
)
_GENERIC = frozenset({"methods", "method", "paper", "study", "evidence", "dataset", "data", "the", "and", "or",
                      "what", "which", "how", "does", "is", "are", "for", "in", "of", "to", "a", "an"})


@dataclass(frozen=True)
class LexicalFocus:
    entities: tuple[str, ...] = ()
    concepts: tuple[tuple[str, ...], ...] = ()


def lexical_focus(query: str) -> LexicalFocus:
    """Only augment CJK queries; keep named ASCII terms, never invent sensors."""
    text = unicodedata.normalize("NFKC", query)[:4000]
    if not re.search(r"[\u3400-\u9fff]", text):
        return LexicalFocus()
    concepts = []
    for aliases, alternatives in FACETS:
        if any(alias in text for alias in aliases):
            concepts.append(alternatives)
    for aliases, english in TERM_GROUPS:
        if any(re.search(r"[\u3400-\u9fff]", alias) and alias in text for alias in aliases):
            # In particular, 传感器 -> sensor, not an invented list of platforms.
            alternatives = (english[0], "WQ") if english[0] == "water quality" else (english[0],)
            concepts.append(alternatives)
    concepts = list(dict.fromkeys(concepts))[:6]
    entities = tuple(dict.fromkeys(token for token in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*|\d+", text)
                                   if token.casefold() not in _GENERIC))[:8]
    return LexicalFocus(entities=entities, concepts=tuple(concepts))
