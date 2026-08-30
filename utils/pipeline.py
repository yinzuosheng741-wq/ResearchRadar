import hashlib
import json
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from domain.models import EvidenceChunk, PaperCandidate
from domain.statuses import FAILED, INDEXED
from ingestion.pipeline import ResearchIngestor, VectorStoreIndex
from providers.registry import ProviderRegistry
from storage.database import ResearchDatabase, normalize_doi, normalize_title
from storage.paths import data_dir, default_database_path
from utils.config import load_agent_config, load_rag_config
from utils.content_loader import load_pdf_text, load_url_text
from utils.logger import logger
from utils.search import search_papers
from utils.text_splitter import split_text

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = DATA_DIR / "reports"
SOURCES_PATH = DATA_DIR / "sources.json"
PAPER_SOURCES_PATH = DATA_DIR / "paper_sources.json"
PAPER_CACHE_DIR = DATA_DIR / "paper_cache"


def _normalize_paper_queries(queries: list[str] | None) -> list[str]:
    if queries:
        return queries
    cfg = load_agent_config()
    return cfg.get("paper_queries", [])


def _normalize_paper_providers(provider: str | None) -> list[str]:
    if provider:
        return [provider]
    cfg = load_agent_config()
    return cfg.get("papers", {}).get("providers", ["openalex"])


def save_sources(sources: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_PATH, "w", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=True, indent=2)


def save_paper_sources(sources: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPER_SOURCES_PATH, "w", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=True, indent=2)


def ingest_sources(sources: list[dict]) -> dict:
    from rag.vector_store import VectorStoreService

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = VectorStoreService()
    database = ResearchDatabase(default_database_path())
    ingested = 0
    for src in sources:
        url = src.get("url", "")
        if not url:
            continue
        cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.txt"

        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            try:
                text = load_url_text(url)
            except Exception as exc:
                logger.warning("failed to load %s: %s", url, exc)
                continue
            cache_path.write_text(text, encoding="utf-8")

        if not text:
            continue

        chunks = split_text(text)
        record = database.upsert_candidate(
            PaperCandidate(
                source=src.get("source") or "manual",
                source_id=src.get("source_id") or url,
                title=src.get("title") or url,
                landing_url=url,
            )
        )
        paper_id = record.paper_id
        metadata = {
            "url": url,
            "paper_id": paper_id,
            "title": src.get("title") or url,
            "page_number": 0,
            "section": "legacy_url",
            "source": src.get("source"),
            "topic": src.get("topic"),
        }
        evidence_chunks = [
            EvidenceChunk(
                chunk_id=f"{paper_id}:legacy:c{index}",
                paper_id=paper_id,
                title=metadata["title"],
                page_number=0,
                section="legacy_url",
                text=chunk,
            )
            for index, chunk in enumerate(chunks)
        ]
        database.replace_chunks(paper_id, evidence_chunks)
        docs = [
            Document(
                page_content=f"标题：{chunk.title}\n位置：{chunk.section or '摘要'}\n正文：{chunk.text}",
                metadata={**metadata, "chunk_id": chunk.chunk_id, "canonical_text": chunk.text},
            )
            for chunk in evidence_chunks
        ]
        store.add_documents(docs)
        database.update_status(paper_id, INDEXED)
        ingested += 1
        logger.info("ingested %s chunks for %s", len(chunks), url)

    return {"sources": len(sources), "ingested": ingested}


def collect_paper_sources(
    queries: list[str],
    providers: list[str],
    max_results: int,
    min_year: int | None,
) -> list[dict]:
    sources = []
    seen = set()
    for provider in providers:
        for query in queries:
            results = search_papers(query, provider, max_results)
            for item in results:
                year = item.get("year")
                if min_year and year and int(year) < min_year:
                    continue
                url = item.get("url") or item.get("pdf_url") or ""
                key = f"{item.get('title')}|{year}|{url}"
                if key in seen:
                    continue
                seen.add(key)
                item["query"] = query
                sources.append(item)
    return sources


def _format_paper_text(paper: dict) -> str:
    parts = [
        f"Title: {paper.get('title', '')}",
        f"Authors: {', '.join(paper.get('authors') or [])}",
        f"Year: {paper.get('year', '')}",
        f"Venue: {paper.get('venue', '')}",
        f"URL: {paper.get('url', '')}",
    ]
    abstract = paper.get("abstract") or ""
    if abstract:
        parts.append("Abstract: " + abstract)
    return "\n".join(parts).strip()


def _truncate_text(text: str) -> str:
    cfg = load_rag_config()
    max_chars = int(cfg.get("max_text_chars", 20000))
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def ingest_papers(papers: list[dict], include_pdf: bool) -> dict:
    from rag.vector_store import VectorStoreService

    PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = VectorStoreService()
    database = ResearchDatabase(default_database_path())
    ingested = 0

    for paper in papers:
        url = paper.get("url") or ""
        cache_key = hashlib.md5((url or paper.get("title", "")).encode("utf-8")).hexdigest()
        cache_path = PAPER_CACHE_DIR / f"{cache_key}.txt"

        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            text = _format_paper_text(paper)
            pdf_url = paper.get("pdf_url")
            if include_pdf and pdf_url:
                try:
                    pdf_text = load_pdf_text(pdf_url)
                    if pdf_text:
                        text = text + "\n\nFull Text:\n" + pdf_text
                except Exception as exc:
                    logger.warning("failed to load pdf %s: %s", pdf_url, exc)
            cache_path.write_text(text, encoding="utf-8")

        if not text:
            continue

        trimmed = _truncate_text(text)
        chunks = split_text(trimmed)
        record = database.upsert_candidate(
            PaperCandidate(
                source=paper.get("source") or "legacy",
                source_id=paper.get("source_id") or url or cache_key,
                title=paper.get("title") or url or "Untitled paper",
                doi=paper.get("doi"),
                authors=paper.get("authors") or [],
                year=paper.get("year"),
                venue=paper.get("venue"),
                abstract=paper.get("abstract"),
                landing_url=url or None,
                pdf_url=paper.get("pdf_url"),
                license=paper.get("license"),
                cited_by_count=int(paper.get("cited_by_count") or 0),
            )
        )
        paper_id = record.paper_id
        metadata = {
            "url": paper.get("url"),
            "paper_id": paper_id,
            "title": paper.get("title") or paper.get("url") or "Untitled paper",
            "page_number": 0,
            "section": "legacy_paper",
            "source": paper.get("source"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
        }
        evidence_chunks = [
            EvidenceChunk(
                chunk_id=f"{paper_id}:legacy:c{index}",
                paper_id=paper_id,
                title=metadata["title"],
                page_number=0,
                section="legacy_paper",
                text=chunk,
            )
            for index, chunk in enumerate(chunks)
        ]
        database.replace_chunks(paper_id, evidence_chunks)
        docs = [
            Document(
                page_content=f"标题：{chunk.title}\n位置：{chunk.section or '摘要'}\n正文：{chunk.text}",
                metadata={**metadata, "chunk_id": chunk.chunk_id, "canonical_text": chunk.text},
            )
            for chunk in evidence_chunks
        ]
        store.add_documents(docs)
        database.update_status(paper_id, INDEXED)
        ingested += 1
        logger.info("ingested %s chunks for %s", len(chunks), paper.get("title"))

    return {"sources": len(papers), "ingested": ingested}


def ingest_url_list(urls: list[str]) -> str:
    sources = [{"url": url, "source": "manual"} for url in urls]
    stats = ingest_sources(sources)
    return f"ingested_sources={stats['ingested']}"


def collect_papers_and_ingest(
    queries: list[str] | None = None,
    provider: str | None = None,
    max_results: int | None = None,
    include_pdf: bool | None = None,
    *,
    registry: ProviderRegistry | None = None,
    ingestor: ResearchIngestor | None = None,
) -> str:
    cfg = load_agent_config()
    queries = _normalize_paper_queries(queries)
    if not queries:
        return "no paper queries configured"

    providers = _normalize_paper_providers(provider)
    if providers != ["openalex"]:
        raise ValueError(f"unsupported paper provider: {providers[0]}")
    paper_cfg = cfg.get("papers", {})
    max_results = max_results or int(paper_cfg.get("max_results", 10))
    include_pdf = include_pdf if include_pdf is not None else bool(paper_cfg.get("include_pdf", False))
    min_year = paper_cfg.get("min_year")
    if min_year is not None:
        min_year = int(min_year)

    registry = registry or ProviderRegistry()
    if ingestor is None:
        from rag.vector_store import VectorStoreService

        database = ResearchDatabase(default_database_path())
        ingestor = ResearchIngestor(
            registry=registry,
            database=database,
            download_dir=data_dir() / "papers",
            vector_index=VectorStoreIndex(VectorStoreService()),
            allow_full_text=include_pdf,
        )

    papers = []
    seen = set()
    for query in queries:
        for paper in registry.discover(
            query,
            from_year=min_year,
            max_results=max_results,
        ):
            identity = (
                ("doi", normalize_doi(paper.doi))
                if normalize_doi(paper.doi)
                else ("title", normalize_title(paper.title), paper.year)
            )
            if identity in seen:
                continue
            seen.add(identity)
            papers.append(paper)

    save_paper_sources([paper.model_dump(mode="json") for paper in papers])
    results = [ingestor.ingest(paper) for paper in papers]
    ingested_count = sum(result.status != FAILED for result in results)
    return f"collected={len(papers)} ingested={ingested_count}"


def sync_papers() -> dict[str, int | str]:
    """Expose legacy discovery with a stable, count-only CLI response."""
    try:
        summary = collect_papers_and_ingest()
    except Exception:
        return {"status": "sync_failed", "collected": 0, "ingested": 0}
    counts: dict[str, int] = {}
    for item in summary.split():
        name, separator, raw = item.partition("=")
        if separator and name in {"collected", "ingested"}:
            try:
                counts[name] = int(raw)
            except ValueError:
                return {"status": "sync_failed", "collected": 0, "ingested": 0}
    if set(counts) != {"collected", "ingested"}:
        return {"status": "sync_failed", "collected": 0, "ingested": 0}
    return {"status": "ok", **counts}
