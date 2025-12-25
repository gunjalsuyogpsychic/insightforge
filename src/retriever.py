import json
from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

@dataclass
class KBItem:
    id: str
    title: str
    text: str
    metadata: dict

def tables_to_kbitems(summary: dict) -> List[KBItem]:
    items: List[KBItem] = []

    meta = summary.get("meta", {})
    items.append(KBItem(
        id="meta",
        title="Dataset metadata",
        text=json.dumps(meta, indent=2),
        metadata={"type": "meta"}
    ))

    tables = summary.get("tables", {})
    for key, value in tables.items():
        if value is None:
            continue
        if isinstance(value, dict):
            for sk, sv in value.items():
                text = sv.to_string(index=False) if hasattr(sv, "to_string") else str(sv)
                items.append(KBItem(
                    id=f"{key}:{sk}",
                    title=f"{key} - {sk}",
                    text=text,
                    metadata={"type": key, "subtype": sk}
                ))
        else:
            text = value.to_string(index=False) if hasattr(value, "to_string") else str(value)
            items.append(KBItem(
                id=key,
                title=key,
                text=text,
                metadata={"type": key}
            ))

    return items

def build_or_load_faiss(kbitems: List[KBItem], faiss_dir: str, embedding_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    docs = [
        Document(
            page_content=f"{it.title}\n\n{it.text}",
            metadata={"id": it.id, **it.metadata}
        )
        for it in kbitems
    ]
    db = FAISS.from_documents(docs, embeddings)
    PathLike = __import__("pathlib").Path
    PathLike(faiss_dir).mkdir(parents=True, exist_ok=True)
    db.save_local(faiss_dir)
    return db

def load_faiss(faiss_dir: str, embedding_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

def retrieve(db: FAISS, query: str, k: int = 4) -> List[Document]:
    return db.similarity_search(query, k=k)
