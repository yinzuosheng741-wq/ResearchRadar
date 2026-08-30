"""Stable data contracts for the research catalog."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class PaperCandidate(BaseModel):
    source: str
    source_id: str
    title: str
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    landing_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    cited_by_count: int = 0
    source_updated_at: str | None = None


class PaperRecord(PaperCandidate):
    paper_id: str
    normalized_doi: str | None = None
    normalized_title: str
    status: str
    source_fingerprint: str
    first_seen_at: datetime
    updated_at: datetime
    last_error: str | None = None


class PageText(BaseModel):
    page_number: int
    text: str


class EvidenceChunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    page_number: int
    section: str | None = None
    text: str
    score: float = 0.0


class EvidenceRef(BaseModel):
    page_number: int
    quote: str


class ExtractedField(BaseModel):
    value: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class PaperProfile(BaseModel):
    prediction_target: ExtractedField
    sensors: list[ExtractedField] = Field(default_factory=list)
    study_area: ExtractedField
    time_span: ExtractedField
    sample_size: ExtractedField
    preprocessing: list[ExtractedField] = Field(default_factory=list)
    models: list[ExtractedField] = Field(default_factory=list)
    baselines: list[ExtractedField] = Field(default_factory=list)
    datasets: list[ExtractedField] = Field(default_factory=list)
    metrics: list[ExtractedField] = Field(default_factory=list)
    conclusions: list[ExtractedField] = Field(default_factory=list)
    limitations: list[ExtractedField] = Field(default_factory=list)
    future_work: list[ExtractedField] = Field(default_factory=list)


class IngestionResult(BaseModel):
    paper_id: str
    status: str
    skipped: bool = False
    chunks_indexed: int = 0


class AnswerCitation(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    page_number: int
    quote: str


class AnswerClaim(BaseModel):
    text: str
    kind: Literal["direct", "synthesis"]
    citations: list[AnswerCitation] = Field(default_factory=list)


class CitedAnswer(BaseModel):
    answer_markdown: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_sufficient: bool
    suggested_search_query: str | None = None
    evidence_level: Literal["direct", "related", "weak", "none"] = "none"
    evidence_reason: str | None = None


class SeedReport(BaseModel):
    metadata_count: int
    fulltext_count: int
    indexed_count: int
    failures: dict[str, int] = Field(default_factory=dict)


class RetryFullTextReport(BaseModel):
    attempted: int
    indexed: int
    abstract_only: int
    failed: int
    failures: dict[str, int] = Field(default_factory=dict)


class ResearchScope(BaseModel):
    topic: str = ""
    prediction_target: str = ""
    sensor: str = ""
    study_area: str = ""
    year_range: str = ""
    method_constraints: list[str] = Field(default_factory=list)


class MemoryFact(BaseModel):
    field: str
    value: str
    source: Literal["user_confirmed", "extracted", "inferred"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchProjectMemory(BaseModel):
    project_id: str = "water-color-prediction"
    topic: str = ""
    prediction_target: str = ""
    sensors: list[str] = Field(default_factory=list)
    study_area: str = ""
    year_range: str = ""
    method_constraints: list[str] = Field(default_factory=list)
    confirmed_paper_ids: list[str] = Field(default_factory=list)
    facts: list[MemoryFact] = Field(default_factory=list)
    last_active_skill: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchPlanFinding(BaseModel):
    text: str
    citations: list[AnswerCitation] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    topic: str
    findings: list[ResearchPlanFinding] = Field(default_factory=list)
    suggested_steps: list[str] = Field(default_factory=list)
    evidence_sufficient: bool
    suggested_search_query: str | None = None


class AgentDiagnostics(BaseModel):
    route_mode: Literal["model", "fallback"]
    route_reason: Literal[
        "model_structured_route",
        "fallback_rule_research_plan",
        "fallback_rule_domain_question",
        "fallback_rule_general_chat",
    ]
    skill_id: Literal["evidence_qa", "research_plan", "general_chat"]
    skill_version: str = "not_applicable"
    evidence_sufficient: bool = False
    retrieval_candidates: int = Field(default=0, ge=0)
    evidence_chunks: int = Field(default=0, ge=0)
    citation_count: int = Field(default=0, ge=0)
    fallback: bool = False
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    model_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)


class ResearchConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    tool_name: str | None = None
    citations: list[AnswerCitation] = Field(default_factory=list)
    diagnostics: AgentDiagnostics | None = None


class ResearchConversationState(BaseModel):
    messages: list[ResearchConversationMessage] = Field(default_factory=list)
    summary: str = ""
    scope: ResearchScope = Field(default_factory=ResearchScope)


class ResearchAgentReply(BaseModel):
    content: str
    tool_name: Literal["evidence_qa", "research_plan", "general_chat"]
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence_sufficient: bool
    suggested_search_query: str | None = None
    diagnostics: AgentDiagnostics | None = None


class ResearchAgentTurn(BaseModel):
    reply: ResearchAgentReply
    state: ResearchConversationState
