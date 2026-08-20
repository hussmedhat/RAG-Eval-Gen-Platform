# app/ingestion/document.py

from dataclasses import dataclass,field

@dataclass
class Document:
    content: str
    metadata:dict= field(default_factory=dict)
