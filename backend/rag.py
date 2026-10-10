"""Book-limited PDF → chunks → BGE embeddings → Chroma retrieval."""
from __future__ import annotations

import re
from typing import Any
from pathlib import Path
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PATH as DB_DIR, PDF_PATH as PDF_DIR
ALLOWED = (
    "Inspired Talks",
    "Reminiscences of Swami Vivekananda",
    "Vivekananda: A Biography",
)
_model: SentenceTransformer | None = None
_collection: Any = None
_keyword_cache: dict[str, Any] | None = None

PROCESSED_DIR = PDF_DIR.parent / "processed"
PREPARED_CHUNKS = PROCESSED_DIR / "chunks.jsonl"
PREPARED_MANIFEST = PROCESSED_DIR / "source_manifest.json"

_STOPWORDS = set("a an and are as at be been being but by can could did do does for from had has have he her hers him his how i if in into is it its me my of on or our she so than that the their them then there these they this those through to too up us was we were what when where which who why will with would you your swami vivekananda say says said speak speaks teaches teaching teachings teach according book books".split())
_TOPIC_EXPANSIONS = {
    "fear": {"fearless", "fearlessness", "afraid", "courage", "bravery"},
    "afraid": {"fear", "fearless", "fearlessness", "courage"},
    "failure": {"fail", "failed", "failing", "success", "effort"},
    "failed": {"failure", "fail", "failing", "effort"},
    "strength": {"strong", "power", "powerful"},
    "strong": {"strength", "power", "powerful"},
    "focus": {"focused", "concentration", "concentrate", "attention"},
    "focused": {"focus", "concentration", "concentrate", "attention"},
    "distraction": {"distracted", "focus", "concentration", "concentrate", "attention"},
    "distracted": {"distraction", "focus", "concentration", "concentrate", "attention"},
    "phone": {"mobile", "smartphone", "distraction", "focus", "concentration", "attention"},
    "mobile": {"phone", "smartphone", "distraction", "focus", "concentration", "attention"},
    "smartphone": {"phone", "mobile", "distraction", "focus", "concentration", "attention"},
    "procrastination": {"procrastinate", "discipline", "duty", "work", "effort", "action"},
    "procrastinate": {"procrastination", "discipline", "duty", "work", "effort", "action"},
    "study": {"studying", "student", "students", "education", "learning"},
    "studying": {"study", "student", "students", "education", "learning"},
    "confidence": {"self", "faith", "strength", "capable"},
    "doubt": {"self-doubt", "confidence", "faith", "uncertain"},
    "career": {"work", "purpose", "profession", "vocation"},
    "purpose": {"work", "career", "duty", "service"},
    "exam": {"examination", "study", "concentration", "effort", "learning"},
    "examination": {"exam", "study", "concentration", "effort", "learning"},
    "anxious": {"anxiety", "fear", "courage", "strength"},
    "anxiety": {"anxious", "fear", "courage", "strength"},
    "pressure": {"expectations", "anxiety", "fear", "strength", "duty"},
    "motivation": {"inspiration", "purpose", "work", "effort", "energy"},
    "lonely": {"loneliness", "love", "compassion", "friendship"},
    "loneliness": {"lonely", "love", "compassion", "friendship"},
    "parent": {"parents", "family", "expectations", "education", "children", "respect"},
    "parents": {"parent", "family", "expectations", "education", "children", "respect"},
    "family": {"parents", "parent", "expectations", "relationships", "respect"},
    "worried": {"worry", "anxiety", "anxious", "fear", "strength"},
    "worry": {"worried", "anxiety", "anxious", "fear", "strength"},
    "expectation": {"expectations", "pressure", "anxiety", "family", "parents"},
    "expectations": {"expectation", "pressure", "anxiety", "family", "parents"},
    "confident": {"confidence", "faith", "strength", "self-belief"},
    "courage": {"fearless", "fearlessness", "fear", "strength", "confidence"},
}
_BROKEN_FONT_MARKS = re.compile(r"[−¢¡¤£¦§]")


def _prepared_chunks_for(pdf: Path) -> list[dict[str, Any]] | None:
    """Read the archive's prepared chunks only when bound to this exact PDF."""
    try:
        manifest = json.loads(PREPARED_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("source_pdf") != pdf.name:
            return None
        digest = hashlib.sha256()
        with pdf.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != manifest.get("sha256"):
            return None
        records = [json.loads(line) for line in PREPARED_CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != manifest.get("prepared_chunks") or not records:
            return None
        return records
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def model() -> SentenceTransformer:
    global _model
    if _model is None:
        # CPU avoids MPS allocation spikes when indexing large combined volumes.
        model_id = "BAAI/bge-small-en-v1.5"
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cached = Path(HF_HUB_CACHE) / "models--BAAI--bge-small-en-v1.5"
            revision = (cached / "refs" / "main").read_text(encoding="utf-8").strip()
            snapshot = cached / "snapshots" / revision
            if (snapshot / "modules.json").is_file():
                model_id = str(snapshot)
        except (OSError, ImportError):
            pass
        _model = SentenceTransformer(model_id, device="cpu")
    return _model


def collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_or_create_collection(
            "viveka_books", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return [w for w in re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", normalized) if len(w) > 1 and w not in _STOPWORDS]


def _usable_source_text(text: str) -> bool:
    """Drop unusable short fragments and pages with visibly broken PDF fonts."""
    if len(_tokens(text)) < 25:
        return False
    marks = len(_BROKEN_FONT_MARKS.findall(text))
    return marks < 8 or marks / max(len(text), 1) <= 0.025


def _keyword_index() -> dict[str, Any]:
    """Build a small in-process BM25 postings index from Chroma's current corpus."""
    global _keyword_cache
    if _keyword_cache is not None:
        return _keyword_cache
    data = collection().get(include=["documents", "metadatas"])
    records = {}
    postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    lengths = {}
    for chunk_id, text, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
        terms = _tokens(text or "")
        counts = Counter(terms)
        records[chunk_id] = {"text": text or "", "metadata": metadata or {}}
        lengths[chunk_id] = len(terms)
        for term, frequency in counts.items():
            postings[term].append((chunk_id, frequency))
            document_frequency[term] += 1
    _keyword_cache = {
        "records": records,
        "postings": postings,
        "df": document_frequency,
        "lengths": lengths,
        "avgdl": (sum(lengths.values()) / max(len(lengths), 1)),
        "total": len(records),
    }
    return _keyword_cache


def _keyword_ranking(query: str, top_n: int) -> list[str]:
    index = _keyword_index()
    terms = set(_tokens(query))
    for word in tuple(terms):
        terms.update(_TOPIC_EXPANSIONS.get(word, ()))
    for word in tuple(terms):
        terms.update(_TOPIC_EXPANSIONS.get(word, ()))
    if not terms:
        return []
    scores: Counter[str] = Counter()
    n_docs = max(index["total"], 1)
    avgdl = max(index["avgdl"], 1.0)
    k1, b = 1.5, 0.75
    for term in terms:
        matches = index["postings"].get(term, ())
        df = index["df"].get(term, 0)
        if not df:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for chunk_id, frequency in matches:
            length = index["lengths"].get(chunk_id, avgdl)
            denominator = frequency + k1 * (1.0 - b + b * length / avgdl)
            scores[chunk_id] += idf * frequency * (k1 + 1.0) / denominator
    return [chunk_id for chunk_id, _ in scores.most_common(top_n)]


def chunk_text(text: str, size: int = 1000, overlap: int = 160) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks, step = [], max(1, size // 5 - overlap // 5)
    width = max(1, size // 5)
    for i in range(0, len(words), step):
        part = " ".join(words[i : i + width]).strip()
        if part:
            chunks.append(part)
    return chunks


def ingest_pdfs() -> dict[str, Any]:
    """Index PDFs placed in data/pdfs. Filename should identify its allowed work."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PDF_DIR.glob("*.pdf"))
    c = collection()
    indexed = 0
    accepted = []
    for pdf in files:
        normalized_stem = pdf.stem.casefold().replace("–", "-").replace("—", "-")
        complete_set_match = "complete works" in normalized_stem and re.search(r"\b1\s*[-–]\s*9\b", normalized_stem)
        volume_match = re.search(r"complete\s+works.*?\b(?:vol(?:ume)?\.?\s*)([1-9])\b", normalized_stem)
        merged_complete_works = "complete works" in normalized_stem and not volume_match
        if complete_set_match or volume_match or merged_complete_works:
            title = "Complete Works of Swami Vivekananda"
            volume = volume_match.group(1) if volume_match and not complete_set_match else None
        else:
            title = next((name for name in ALLOWED if name.casefold() in normalized_stem), None)
            volume = None
        if title is None:
            continue
        ids, texts, metas = [], [], []
        prepared = _prepared_chunks_for(pdf) if title == "Complete Works of Swami Vivekananda" else None
        if prepared:
            for record in prepared:
                source_meta = record.get("metadata") or {}
                text = str(record.get("text") or "").strip()
                key = str(record.get("chunk_id") or "")
                page_number = source_meta.get("pdf_page")
                if not text or not _usable_source_text(text) or not key or not isinstance(page_number, int):
                    continue
                heading = str(source_meta.get("title") or "").strip()
                generic_heading = heading.casefold() in {"complete works", "complete works of swami vivekananda"}
                if generic_heading and not source_meta.get("section"):
                    # These pages are contents lists, not passages of teaching.
                    continue
                if generic_heading:
                    heading = str(source_meta.get("section") or "").strip()
                elif not ((any(char.isalpha() for char in heading) and heading.upper() == heading) or re.match(r"(?i)^chapter\s+", heading)):
                    # The archive's title helper falls back to the first body line;
                    # only keep explicit heading-like metadata as a chapter label.
                    heading = ""
                ids.append(key)
                texts.append(text)
                metas.append({"book": title, "file": pdf.name, "page": page_number, "volume": str(source_meta["volume"]) if source_meta.get("volume") else None, "chapter": heading or None, "section": str(source_meta.get("section") or "") or None, "chunk_id": key})
            print(f"Using {len(texts)} prepared page-linked chunks for {pdf.name}.", flush=True)
        else:
            try:
                doc = fitz.open(pdf)
            except Exception as exc:
                print(f"Skipping unreadable PDF {pdf.name}: {exc}")
                continue
            toc = doc.get_toc()
            for page_index, page in enumerate(doc):
                page_text = page.get_text("text")
                page_volume = volume
                if title == "Complete Works of Swami Vivekananda" and volume is None:
                    markers = set(re.findall(r"(?i)(?:complete\s+works(?:\s+of\s+swami\s+vivekananda)?\s+)?volume\s*[-–:]?\s*([1-9])\b", page_text[:1600]))
                    page_volume = next(iter(markers)) if len(markers) == 1 else None
                for part_index, chunk in enumerate(chunk_text(page_text)):
                    if not _usable_source_text(chunk):
                        continue
                    key = f"{pdf.name}:{page_index + 1}:{part_index}"
                    ids.append(key)
                    texts.append(chunk)
                    chapter = None
                    for level, toc_title, toc_page in toc:
                        if toc_page == page_index + 1 and not re.fullmatch(r"(?i)\s*(?:complete works\s*)?vol(?:ume)?\.?\s*[1-9]\s*", toc_title):
                            chapter = toc_title
                    metas.append({"book": title, "file": pdf.name, "page": page_index + 1, "volume": page_volume, "chapter": chapter, "chunk_id": key})
            doc.close()
        if texts:
            # Replace this file's existing rows, so re-ingestion updates rather than duplicates it.
            old = c.get(where={"file": pdf.name}, include=[])
            if old["ids"]:
                c.delete(ids=old["ids"])
            # Keep peak memory bounded for large combined volumes.
            embedder = model()
            batch_size = 16
            for start in range(0, len(texts), batch_size):
                stop = start + batch_size
                vectors = embedder.encode(["Represent this passage for retrieval: " + t for t in texts[start:stop]], normalize_embeddings=True, batch_size=batch_size).tolist()
                c.upsert(ids=ids[start:stop], documents=texts[start:stop], metadatas=metas[start:stop], embeddings=vectors)
                if stop == batch_size or stop >= len(texts) or stop % 1024 == 0:
                    print(f"Indexed {min(stop, len(texts))}/{len(texts)} chunks from {pdf.name}...", flush=True)
            indexed += len(texts)
            accepted.append(pdf.name)
            global _keyword_cache
            _keyword_cache = None
    return {"indexed_chunks": indexed, "files": accepted, "available": c.count() > 0}


def retrieve(query: str, limit: int = 4) -> list[dict[str, Any]]:
    c = collection()
    if c.count() == 0 or not query.strip():
        return []
    vector = model().encode(["Represent this passage for retrieval: " + query], normalize_embeddings=True).tolist()
    # Combine meaning-based BGE search with BM25 term matching. Reciprocal-rank
    # fusion avoids mixing incompatible raw cosine and lexical score scales.
    candidate_count = min(max(limit * 16, 40), c.count())
    result = c.query(query_embeddings=vector, n_results=candidate_count, include=["documents", "metadatas", "distances"])
    semantic_ids = [chunk_id for chunk_id, text in zip(result["ids"][0], result["documents"][0]) if _usable_source_text(text or "")]
    lexical_ids = _keyword_ranking(query, candidate_count)
    ranks: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(semantic_ids, start=1):
        ranks[chunk_id] += 0.70 / (60 + rank)
    for rank, chunk_id in enumerate(lexical_ids, start=1):
        ranks[chunk_id] += 0.30 / (60 + rank)
    index = _keyword_index()
    hits = []
    for chunk_id, score in sorted(ranks.items(), key=lambda item: item[1], reverse=True):
        record = index["records"].get(chunk_id)
        if not record:
            continue
        doc, meta = record["text"], record["metadata"]
        if not _usable_source_text(doc):
            continue
        hits.append({"text": doc, "book": meta["book"], "file": meta["file"], "page": meta.get("page"), "volume": meta.get("volume"), "chapter": meta.get("chapter"), "section": meta.get("section"), "chunk_id": chunk_id, "hybrid_score": round(score, 6)})
        if len(hits) >= limit:
            break
    return hits


def status() -> dict[str, Any]:
    c = collection()
    return {"indexed_chunks": c.count(), "supported_works": ["Complete Works of Swami Vivekananda Volumes 1–9", *ALLOWED]}
