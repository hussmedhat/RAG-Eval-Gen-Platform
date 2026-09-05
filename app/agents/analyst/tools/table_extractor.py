"""
Analyst Tool 2: Table Extractor

Pulls structured tables (metrics, results, comparisons) out of the
plain-text chunk content the Retriever hands over. Tables in source
documents often survive chunking as either:
  1. Markdown-style rows using "|" separators, or
  2. Loosely whitespace-aligned columns (common when a PDF table gets
     flattened to plain text during extraction).
"""

import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.config import get_settings


class ExtractedTable(BaseModel):
    headers: list[str] = Field(description="Column headers, in order")
    rows: list[list[str]] = Field(description="Each row's cell values, in the same column order as headers")

_parser = PydanticOutputParser(pydantic_object=ExtractedTable)

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract any tabular data from the passage into a structured table. "
     "If there is no real table in the passage, return empty headers and rows."),
    ("human", "Passage:\n{passage}\n\n{format_instructions}"),
]).partial(format_instructions=_parser.get_format_instructions())


def _try_regex_extract(text: str) -> ExtractedTable | None:
    """Fast path: looks for markdown-style "| a | b | c |" rows."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]

    if len(table_lines) < 2:  # need at least a header row + one data row
        return None

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    headers = split_row(table_lines[0])
    data_rows = []
    for line in table_lines[1:]:
        # skip markdown separator rows like "|---|---|---|"
        if re.fullmatch(r"[\s:|-]+", line):
            continue
        data_rows.append(split_row(line))

    if not data_rows:
        return None

    return ExtractedTable(headers=headers, rows=data_rows)

def _build_llm_chain():
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.evaluator_model,
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        temperature=0,
    )
    return _EXTRACT_PROMPT | llm | _parser


def extract_table(text: str) -> ExtractedTable | None:

    """Returns an ExtractedTable if the passage contains tabular data,
    or None if it doesn't."""

    regex_result = _try_regex_extract(text)
    if regex_result is not None:
        return regex_result

    try:
        chain = _build_llm_chain()
        result = chain.invoke({"passage": text[:1500]})
        if not result.headers or not result.rows:
            return None
        return result
    except Exception:
        return None  # extraction failure shouldn't crash the Analyst — just means no table found

