"""Credential-lazy command line entry point for the research assistant."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Water-color research assistant")
    subparsers = parser.add_subparsers(dest="command")

    seed = subparsers.add_parser("seed", help="Collect a curated seed corpus")
    seed.add_argument("--config", required=True)
    subparsers.add_parser("sync", help="Discover and ingest configured papers")
    subparsers.add_parser("stats", help="Show count-only catalog statistics")
    audit = subparsers.add_parser("knowledge-audit", help="Audit catalog, evidence, and vector coverage")
    audit.add_argument("--json", action="store_true", help="Emit the machine-readable JSON report")
    retry = subparsers.add_parser("retry-fulltext", help="Retry lawful full-text ingestion for DOI-backed abstracts")
    retry.add_argument("--limit", type=int, required=True)
    retry.add_argument(
        "--provider",
        choices=["semantic_scholar"],
        default=None,
        help="Restrict DOI full-text resolution to one provider",
    )
    retry.add_argument("--request-rate", type=float, default=None)
    retry.add_argument("--download-timeout", type=int, default=None)
    retry.add_argument(
        "--discovered-only",
        action="store_true",
        help="Process only discovered DOI candidates and skip previously failed abstract records",
    )
    abstract_profile = subparsers.add_parser("profile-abstracts", help="Build citation-linked profiles from stored abstracts")
    abstract_profile.add_argument("--limit", type=int, required=True)
    corpus = subparsers.add_parser("corpus-curate", help="Report the active evidence-backed RAG corpus")
    corpus.add_argument("--limit", type=int, required=True)

    ask = subparsers.add_parser("ask", help="Ask an evidence-grounded question")
    ask.add_argument("question")
    subparsers.add_parser("rebuild-index", help="Guardedly rebuild the vector index")
    profile = subparsers.add_parser("profile", help="Extract profiles from local parsed PDFs")
    profile.add_argument("--retry-failed", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="Run the Task 10 evaluation")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM calls and measure retrieval metrics only",
    )
    answer_evaluate = subparsers.add_parser(
        "evaluate-answers", help="Evaluate sanitized answer/citation samples"
    )
    answer_evaluate.add_argument("--dataset", required=True)
    subparsers.add_parser("provider-health", help="Run sanitized bounded provider checks")

    legacy = subparsers.add_parser("collect-papers", help="Legacy discovery command")
    legacy.add_argument("--queries")
    legacy.add_argument("--provider", choices=["openalex"])
    legacy.add_argument("--max-results", type=int)
    legacy.add_argument("--include-pdf", action="store_true", default=None)
    return parser


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(value: Any, stdout: TextIO) -> None:
    if isinstance(value, str):
        stdout.write(value + "\n")
    else:
        stdout.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")


class DefaultServices:
    """Construct external/model dependencies only inside the selected operation."""

    @staticmethod
    def _database():
        from storage.database import ResearchDatabase
        from storage.paths import default_database_path

        return ResearchDatabase(default_database_path())

    @staticmethod
    def _model():
        from dotenv import load_dotenv
        from model.factory import build_chat_model

        load_dotenv()
        return build_chat_model()

    def _vector_store(self, database=None, *, load_existing: bool = True):
        from dotenv import load_dotenv
        from rag.vector_store import VectorStoreService

        load_dotenv()
        return VectorStoreService(
            database=database or self._database(),
            load_existing=load_existing,
        )

    @staticmethod
    def _allowed_paper_ids(database):
        if not hasattr(database, "list_papers") or not hasattr(database, "list_chunks"):
            return None
        from workflows.corpus_curator import CorpusCurator

        return CorpusCurator(database).select_ids(limit=600)

    def _ingestion_parts(
        self, *, allow_full_text=True, download_timeout=None, extract_profiles=True
    ):
        from ingestion.pipeline import (
            DeferredProfileExtractor,
            NullVectorIndex,
            ResearchIngestor,
            VectorStoreIndex,
        )
        from ingestion.profile_extractor import PaperProfileExtractor
        from ingestion.downloader import PdfDownloader
        from dotenv import load_dotenv
        from providers.registry import ProviderRegistry
        from storage.paths import data_dir

        load_dotenv()
        database = self._database()
        registry = ProviderRegistry()
        profile_extractor = DeferredProfileExtractor()
        vector_index = NullVectorIndex()
        if os.environ.get("OPENAI_API_KEY", "").strip() and extract_profiles:
            model = self._model()
            profile_extractor = PaperProfileExtractor(model)
        if os.environ.get("OPENAI_API_KEY", "").strip():
            vector_index = VectorStoreIndex(self._vector_store(database))
        ingestor = ResearchIngestor(
            registry=registry,
            database=database,
            download_dir=data_dir() / "papers",
            profile_extractor=profile_extractor,
            vector_index=vector_index,
            allow_full_text=allow_full_text,
            downloader=(PdfDownloader(registry, timeout=download_timeout) if download_timeout else None),
        )
        return registry, ingestor, database

    def seed(self, config):
        from workflows.seed import SeedService

        registry, ingestor, database = self._ingestion_parts(
            allow_full_text=True, download_timeout=10, extract_profiles=False
        )
        return SeedService(registry, ingestor, database).collect(config)

    def sync(self):
        from utils.pipeline import sync_papers

        return sync_papers()

    def stats(self):
        from domain.statuses import ABSTRACT_ONLY, FAILED, INDEXED, PARSED, PDF_READY, PROFILED

        return self._database().catalog_statistics(
            status_names=(PDF_READY, PARSED, PROFILED, INDEXED, ABSTRACT_ONLY, FAILED)
        )

    def knowledge_audit(self):
        from workflows.knowledge_audit import KnowledgeAuditService

        database = self._database()
        return KnowledgeAuditService(database, vector_store=self._vector_store(database)).run()

    def retry_fulltext(
        self,
        limit,
        *,
        discovered_only=False,
        provider=None,
        request_rate=None,
        download_timeout=None,
    ):
        from workflows.knowledge_audit import KnowledgeAuditService
        from workflows.retry_fulltext import RetryFullTextService

        registry, ingestor, database = self._ingestion_parts(
            allow_full_text=True,
            download_timeout=download_timeout or 10,
            extract_profiles=False,
        )
        if request_rate is not None:
            if request_rate <= 0 or request_rate >= 1:
                raise ValueError("retry_fulltext_invalid_request_rate")
            if provider == "semantic_scholar":
                registry.semantic_scholar.requests_per_second = request_rate
        report = RetryFullTextService(database, ingestor).run(
            limit=limit, discovered_only=discovered_only, provider=provider
        )
        audit = KnowledgeAuditService(
            database, vector_store=self._vector_store(database)
        ).run()
        return {"retry": report.model_dump(), "audit": audit}

    def profile_abstracts(self, limit):
        from workflows.metadata_profiles import MetadataProfileService

        return MetadataProfileService(self._database()).run(limit=limit)

    def cited_qa(self, question):
        from retrieval.hybrid import HybridRetriever
        from retrieval.keyword_index import KeywordIndex
        from utils.config import load_rag_config
        from workflows.qa import CitedQaService

        database = self._database()
        vector = self._vector_store(database)
        config = load_rag_config()
        retriever = HybridRetriever(
            KeywordIndex(database), vector,
            keyword_weight=float(config.get("keyword_weight", 2.0)),
            vector_weight=float(config.get("vector_weight", 0.5)),
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_k=int(config.get("candidate_k", 20)),
            max_chunks_per_paper=int(config.get("max_chunks_per_paper", 4)),
            allowed_paper_ids=self._allowed_paper_ids(database),
        )
        return CitedQaService(
            retriever, self._model(), chunk_store=database
        ).answer(question)

    def research_chat(self, message, conversation=None):
        from agent.research_agent import ResearchAgentService
        from retrieval.hybrid import HybridRetriever
        from retrieval.keyword_index import KeywordIndex
        from workflows.qa import CitedQaService
        from workflows.research_plan import ResearchPlanService
        from utils.config import load_rag_config

        database = self._database()
        vector = self._vector_store(database)
        config = load_rag_config()
        retriever = HybridRetriever(
            KeywordIndex(database), vector,
            keyword_weight=float(config.get("keyword_weight", 2.0)),
            vector_weight=float(config.get("vector_weight", 0.5)),
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_k=int(config.get("candidate_k", 20)),
            max_chunks_per_paper=int(config.get("max_chunks_per_paper", 4)),
            allowed_paper_ids=self._allowed_paper_ids(database),
        )
        model = self._model()
        qa_service = CitedQaService(retriever, model, chunk_store=database)
        plan_service = ResearchPlanService(
            retriever, model, chunk_store=database
        )
        return ResearchAgentService(
            model=model,
            qa_service=qa_service,
            plan_service=plan_service,
            memory_store=database,
        ).chat(message, conversation)

    def rebuild_index(self):
        database = self._database()
        return self._vector_store(
            database, load_existing=False
        ).rebuild_from_database()

    def profile(self, *, retry_failed=False):
        from ingestion.pdf_parser import PdfParser
        from ingestion.profile_extractor import PaperProfileExtractor
        from storage.paths import data_dir
        from workflows.profile import ProfileService

        return ProfileService(
            database=self._database(),
            pdf_dir=data_dir() / "papers",
            parser=PdfParser(),
            extractor=PaperProfileExtractor(self._model()),
        ).run(retry_failed=retry_failed)

    def evaluate(self, dataset, *, retrieval_only: bool = False):
        from evaluation.dataset import load_questions
        from evaluation.run import run_evaluation
        from retrieval.hybrid import HybridRetriever
        from retrieval.keyword_index import KeywordIndex
        from utils.config import load_rag_config
        from workflows.qa import CitedQaService

        # Reject malformed/template input before constructing embeddings or models.
        load_questions(Path(dataset))
        database = self._database()
        keyword = KeywordIndex(database)
        vector = self._vector_store(database)
        model = self._model()
        rag_config = load_rag_config()
        evaluation_config = {
            "retrieval_k": 5,
            "hybrid": {
                "keyword_weight": float(rag_config.get("keyword_weight", 2.0)),
                "vector_weight": float(rag_config.get("vector_weight", 0.5)),
                "rrf_k": int(rag_config.get("rrf_k", 60)),
            },
            "two_stage": {
                "paper_candidate_k": int(
                    rag_config.get("two_stage", {}).get("paper_candidate_k", 50)
                ),
                "paper_k": int(rag_config.get("two_stage", {}).get("paper_k", 12)),
                "chunk_candidate_k": int(
                    rag_config.get("two_stage", {}).get("chunk_candidate_k", 40)
                ),
                "max_chunks_per_paper": int(
                    rag_config.get("two_stage", {}).get("max_chunks_per_paper", 2)
                ),
            },
        }
        return run_evaluation(
            Path(dataset),
            keyword_retriever=keyword,
            vector_retriever=vector,
            hybrid_factory=self._evaluation_hybrid,
            two_stage_factory=self._evaluation_two_stage,
            qa_factory=lambda retriever: CitedQaService(
                retriever, model, chunk_store=database
            ),
            chunk_store=database,
            config=evaluation_config,
            include_answer_metrics=not retrieval_only,
        )

    def evaluate_answers(self, dataset):
        from evaluation.run import run_answer_evaluation

        return run_answer_evaluation(Path(dataset))

    @staticmethod
    def _evaluation_hybrid(keyword, vector, **settings):
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever(keyword, vector, **settings)

    @staticmethod
    def _evaluation_two_stage(keyword, vector, _hybrid=None, **settings):
        from retrieval.two_stage import TwoStageRetriever

        return TwoStageRetriever(keyword, vector, **settings)

    @staticmethod
    def provider_health():
        from providers.health import default_provider_health

        return default_provider_health()

    def corpus_curate(self, limit):
        from workflows.corpus_curator import CorpusCurator

        return CorpusCurator(self._database()).report(limit=limit)

    @staticmethod
    def collect_papers(args):
        from utils.pipeline import collect_papers_and_ingest

        queries = [line.strip() for line in (args.queries or "").splitlines() if line.strip()] or None
        return collect_papers_and_ingest(
            queries, args.provider, args.max_results, args.include_pdf
        )

def run(argv=None, *, services=None, stdout: TextIO | None = None) -> int:
    stdout = stdout or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(file=stdout)
        return 0
    services = services or DefaultServices()

    if args.command == "seed":
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        result = services.seed(config)
    elif args.command == "retry-fulltext":
        if args.limit <= 0:
            _emit({"error": "retry_fulltext_invalid_limit", "status": "error"}, stdout)
            return 2
        try:
            if not args.discovered_only and args.provider is None and args.request_rate is None and args.download_timeout is None:
                result = services.retry_fulltext(args.limit)
            else:
                result = services.retry_fulltext(
                    args.limit,
                    discovered_only=args.discovered_only,
                    provider=args.provider,
                    request_rate=args.request_rate,
                    download_timeout=args.download_timeout,
                )
        except ValueError as exc:
            _emit({"error": str(exc), "status": "error"}, stdout)
            return 2
    elif args.command == "profile-abstracts":
        if args.limit <= 0:
            _emit({"error": "metadata_profile_invalid_limit", "status": "error"}, stdout)
            return 2
        try:
            result = services.profile_abstracts(args.limit)
        except ValueError as exc:
            _emit({"error": str(exc), "status": "error"}, stdout)
            return 2
    elif args.command == "corpus-curate":
        if args.limit <= 0:
            _emit({"error": "corpus_invalid_limit", "status": "error"}, stdout)
            return 2
        result = services.corpus_curate(args.limit)
    elif args.command == "sync":
        result = services.sync()
    elif args.command == "stats":
        result = services.stats()
    elif args.command == "knowledge-audit":
        result = services.knowledge_audit()
    elif args.command == "ask":
        result = services.cited_qa(args.question)
    elif args.command == "rebuild-index":
        result = {"indexed": services.rebuild_index(), "status": "ok"}
    elif args.command == "profile":
        result = services.profile(retry_failed=args.retry_failed)
    elif args.command == "evaluate":
        try:
            if args.retrieval_only:
                result = services.evaluate(args.dataset, retrieval_only=True)
            else:
                result = services.evaluate(args.dataset)
        except Exception as exc:
            from evaluation.dataset import DatasetError
            from evaluation.run import EvaluationError

            if isinstance(exc, (DatasetError, EvaluationError)):
                _emit({"error": str(exc), "status": "error"}, stdout)
                return 2
            raise
    elif args.command == "evaluate-answers":
        try:
            result = services.evaluate_answers(args.dataset)
        except Exception as exc:
            from evaluation.dataset import DatasetError

            if isinstance(exc, DatasetError):
                _emit({"error": str(exc), "status": "error"}, stdout)
                return 2
            raise
    elif args.command == "provider-health":
        result = services.provider_health()
    elif args.command == "collect-papers":
        result = services.collect_papers(args)
    else:
        raise ValueError("unknown_command")
    _emit(result, stdout)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
