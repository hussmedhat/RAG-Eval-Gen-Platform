"""
Answer Tool: Citation Formatter

Takes the raw answer text produced by the Answer Agent's LLM — which
cites evidence chunks using inline tags like [E3] referring to the
Nth chunk in the evidence bundle passed to the prompt — and converts
those tags into clean, sequential citation markers ([1], [2], ...)
in order of first appearance. Also returns the ordered list of
Documents actually cited, so the Source Formatter can build a
matching reference list.

Working with our own evidence-index tags (rather than trusting the
LLM to invent citation numbers itself) keeps every citation
traceable back to a real, specific chunk — a wrong tag just fails to
match anything, rather than silently pointing at the wrong source.
"""
import re

from app.ingestion.document import Document

_CITATION_TAG = re.compile(r"\[E(\d+)\]")


def format_citations(raw_answer: str, evidence: list[Document]) -> tuple[str, list[Document]]:
    """
    Renumbers [E<i>] tags (1-indexed positions into `evidence`) into
    sequential [n] markers in order of first appearance, collapsing
    repeats of the same source to the same number.

    Returns (formatted_answer, cited_evidence) where cited_evidence[i]
    is the Document that citation marker [i+1] refers to.
    """
    seen_order: dict[int, int] = {}  # evidence_index -> citation_number
    cited_evidence: list[Document] = []

    def _replace(match: re.Match) -> str:
        evidence_idx = int(match.group(1)) - 1  # tags are 1-indexed (E1 == evidence[0])
        if evidence_idx < 0 or evidence_idx >= len(evidence):
            return ""  # dangling/hallucinated tag — drop silently rather than mis-cite

        if evidence_idx not in seen_order:
            seen_order[evidence_idx] = len(seen_order) + 1
            cited_evidence.append(evidence[evidence_idx])

        return f"[{seen_order[evidence_idx]}]"

    formatted = _CITATION_TAG.sub(_replace, raw_answer)
    formatted = re.sub(r"[ \t]{2,}", " ", formatted)  # tidy gaps left by dropped tags
    return formatted, cited_evidence