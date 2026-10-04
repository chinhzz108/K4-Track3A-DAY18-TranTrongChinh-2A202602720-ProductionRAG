from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_SEMANTIC_MODEL = None


def _get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            class SimpleEmbeddingModel:
                def encode(self, sentences):
                    import numpy as np
                    ngrams = set()
                    for s in sentences:
                        s_clean = s.lower()
                        for i in range(len(s_clean) - 2):
                            ngrams.add(s_clean[i:i+3])
                    ng_list = sorted(list(ngrams))
                    if not ng_list:
                        return np.zeros((len(sentences), 1))
                    ng2idx = {ng: i for i, ng in enumerate(ng_list)}
                    vecs = []
                    for s in sentences:
                        v = np.zeros(len(ng_list))
                        s_clean = s.lower()
                        for i in range(len(s_clean) - 2):
                            v[ng2idx[s_clean[i:i+3]]] += 1
                        norm = np.linalg.norm(v)
                        if norm > 0:
                            v = v / norm
                        vecs.append(v)
                    return np.array(vecs)
            _SEMANTIC_MODEL = SimpleEmbeddingModel()
    return _SEMANTIC_MODEL


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    # Split text into sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    import numpy as np

    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        vec_prev = embeddings[i - 1]
        vec_curr = embeddings[i]
        denom = (np.linalg.norm(vec_prev) * np.linalg.norm(vec_curr)) + 1e-9
        sim = float(np.dot(vec_prev, vec_curr) / denom)

        if sim < threshold:
            chunk_text = " ".join(current_group).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
                ))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])

    if current_group:
        chunk_text = " ".join(current_group).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # Group paragraphs into parent chunks
    parents: list[Chunk] = []
    parent_texts: list[str] = []
    curr_parent = ""

    for para in paragraphs:
        if len(curr_parent) + len(para) > parent_size and curr_parent:
            parent_texts.append(curr_parent.strip())
            curr_parent = ""
        curr_parent += para + "\n\n"
    if curr_parent.strip():
        parent_texts.append(curr_parent.strip())

    children: list[Chunk] = []
    for p_idx, p_text in enumerate(parent_texts):
        pid = f"parent_{p_idx}"
        p_chunk = Chunk(
            text=p_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_idx}
        )
        parents.append(p_chunk)

        # Split parent into children
        p_sentences = re.split(r'(?<=[.!?])\s+|\n+', p_text)
        p_sentences = [s.strip() for s in p_sentences if s.strip()]
        
        curr_child = ""
        for s in p_sentences:
            if len(curr_child) + len(s) > child_size and curr_child:
                children.append(Chunk(
                    text=curr_child.strip(),
                    metadata={**metadata, "chunk_type": "child", "chunk_index": len(children)},
                    parent_id=pid
                ))
                curr_child = ""
            curr_child += (s + " ")
        if curr_child.strip():
            children.append(Chunk(
                text=curr_child.strip(),
                metadata={**metadata, "chunk_type": "child", "chunk_index": len(children)},
                parent_id=pid
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    lines = text.split("\n")
    chunks: list[Chunk] = []

    current_header = ""
    current_content: list[str] = []

    header_regex = re.compile(r'^(#{1,6}\s+.+)$')

    def _flush():
        nonlocal current_header, current_content
        text_body = "\n".join(current_content).strip()
        if text_body or current_header:
            full_text = f"{current_header}\n\n{text_body}".strip() if current_header and text_body and not text_body.startswith(current_header) else (text_body or current_header)
            meta = {
                **metadata,
                "chunk_index": len(chunks),
                "strategy": "structure"
            }
            if current_header:
                meta["section"] = current_header.lstrip("#").strip()
            chunks.append(Chunk(text=full_text, metadata=meta))
        current_content = []

    for line in lines:
        if header_regex.match(line):
            if current_content or current_header:
                _flush()
            current_header = line.strip()
            current_content = [line.strip()]
        else:
            current_content.append(line)

    _flush()

    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "chunk_index": 0, "strategy": "structure"}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
