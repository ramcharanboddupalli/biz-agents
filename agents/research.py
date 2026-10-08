"""Deterministic, source-aware business research orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from rag import BusinessProfile


class ResearchCategory(str, Enum):
    BUSINESS = "business"
    MARKET = "market"
    COMPETITORS = "competitors"
    LEADS = "leads"


class ResearchStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ResearchQuery:
    category: ResearchCategory
    query: str


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """One result supplied by an injected research provider."""

    title: str
    url: str
    snippet: str
    provider: str
    source_type: str = "web"
    metadata: Mapping[str, str] = field(default_factory=dict)


class ResearchProvider(Protocol):
    """Provider boundary for deterministic tests and future search adapters."""

    def search(self, query: str) -> Sequence[ProviderResult]:
        """Return source-backed results for a single query."""
        ...


@dataclass(frozen=True, slots=True)
class ResearchSource:
    source_id: str
    title: str
    url: str
    snippet: str
    provider: str
    source_type: str
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class BusinessFinding:
    topic: str
    summary: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MarketResearch:
    topic: str
    summary: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompetitorResearch:
    name: str
    description: str
    relevance: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeadResearch:
    name: str
    description: str
    relevance: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResearchIssue:
    category: ResearchCategory
    query: str
    message: str


@dataclass(frozen=True, slots=True)
class ResearchResult:
    business_profile_id: str
    status: ResearchStatus
    business_findings: tuple[BusinessFinding, ...]
    market_findings: tuple[MarketResearch, ...]
    competitors: tuple[CompetitorResearch, ...]
    leads: tuple[LeadResearch, ...]
    sources: tuple[ResearchSource, ...]
    issues: tuple[ResearchIssue, ...]


class ResearchAgent:
    """Plan and collect profile-driven research through an injected provider."""

    def __init__(self, provider: ResearchProvider) -> None:
        if not callable(getattr(provider, "search", None)):
            raise TypeError("provider must implement search(query).")
        self._provider = provider

    def build_plan(self, profile: BusinessProfile) -> tuple[ResearchQuery, ...]:
        self._validate_profile(profile)
        offering = profile.offering
        target_customers = profile.target_customers
        location = profile.location
        goals = profile.goals

        return (
            ResearchQuery(
                ResearchCategory.BUSINESS,
                f"{offering} business and product context",
            ),
            ResearchQuery(
                ResearchCategory.BUSINESS,
                f"{offering} business priorities for {goals}",
            ),
            ResearchQuery(
                ResearchCategory.MARKET,
                f"{offering} market trends in {location}",
            ),
            ResearchQuery(
                ResearchCategory.MARKET,
                f"{target_customers} market needs and opportunities in {location} "
                f"to support {goals}",
            ),
            ResearchQuery(
                ResearchCategory.COMPETITORS,
                f"businesses offering {offering} in {location}",
            ),
            ResearchQuery(
                ResearchCategory.LEADS,
                f"organizations serving {target_customers} in {location} "
                f"to support {goals}",
            ),
        )

    def research(self, profile: BusinessProfile) -> ResearchResult:
        plan = self.build_plan(profile)
        sources_by_url: dict[str, ResearchSource] = {}
        business_findings: list[BusinessFinding] = []
        market_findings: list[MarketResearch] = []
        competitors: list[CompetitorResearch] = []
        leads: list[LeadResearch] = []
        issues: list[ResearchIssue] = []
        successful_queries = 0

        for planned_query in plan:
            try:
                provider_results = self._provider.search(planned_query.query)
            except Exception as exc:
                issues.append(
                    ResearchIssue(
                        category=planned_query.category,
                        query=planned_query.query,
                        message=f"Provider search failed: {exc}",
                    )
                )
                continue

            if not isinstance(provider_results, Sequence) or isinstance(
                provider_results, (str, bytes)
            ):
                issues.append(
                    ResearchIssue(
                        category=planned_query.category,
                        query=planned_query.query,
                        message="Provider returned malformed results; expected a sequence.",
                    )
                )
                continue

            successful_queries += 1
            for provider_result in provider_results:
                source, issue_message = self._normalize_source(provider_result)
                if issue_message is not None:
                    issues.append(
                        ResearchIssue(
                            category=planned_query.category,
                            query=planned_query.query,
                            message=issue_message,
                        )
                    )
                    continue
                if source is None:
                    continue

                canonical_url = source.url
                sources_by_url.setdefault(canonical_url, source)
                stored_source = sources_by_url[canonical_url]
                source_ids = (stored_source.source_id,)
                self._append_finding(
                    planned_query,
                    stored_source,
                    source_ids,
                    business_findings,
                    market_findings,
                    competitors,
                    leads,
                )

        if successful_queries == 0:
            status = ResearchStatus.FAILED
        elif issues:
            status = ResearchStatus.PARTIAL
        else:
            status = ResearchStatus.SUCCESS

        return ResearchResult(
            business_profile_id=profile.business_profile_id,
            status=status,
            business_findings=tuple(business_findings),
            market_findings=tuple(market_findings),
            competitors=tuple(competitors),
            leads=tuple(leads),
            sources=tuple(sources_by_url.values()),
            issues=tuple(issues),
        )

    @staticmethod
    def _validate_profile(profile: BusinessProfile) -> None:
        if not isinstance(profile, BusinessProfile):
            raise TypeError("profile must be a BusinessProfile.")
        for field_name in (
            "offering",
            "target_customers",
            "location",
            "budget",
            "goals",
            "business_profile_id",
        ):
            value = getattr(profile, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"profile.{field_name} must be a non-empty string.")

    @staticmethod
    def _normalize_source(
        value: object,
    ) -> tuple[ResearchSource | None, str | None]:
        if not isinstance(value, ProviderResult):
            return None, "Provider returned a malformed result; expected ProviderResult."

        for field_name in ("title", "url", "snippet", "provider", "source_type"):
            field_value = getattr(value, field_name)
            if not isinstance(field_value, str):
                return None, f"Provider result has invalid {field_name}; expected text."

        title = value.title.strip()
        url = value.url.strip()
        snippet = value.snippet.strip()
        provider = value.provider.strip()
        source_type = value.source_type.strip()
        if not title or not snippet or not provider or not source_type:
            return None, "Provider result is missing a title, snippet, provider, or source type."

        canonical_url = ResearchAgent._canonicalize_url(url)
        if canonical_url is None:
            return None, "Provider result has no usable HTTP or HTTPS source URL."

        metadata: dict[str, str] = {}
        if not isinstance(value.metadata, Mapping):
            return None, "Provider result metadata must be a mapping of text values."
        try:
            metadata_items = value.metadata.items()
            for key, item in metadata_items:
                if not isinstance(key, str) or not isinstance(item, str):
                    return None, "Provider result metadata keys and values must be text."
                metadata[key] = item
        except Exception:
            return None, "Provider result metadata could not be read."

        source_id = "source-" + sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        return (
            ResearchSource(
                source_id=source_id,
                title=title,
                url=canonical_url,
                snippet=snippet,
                provider=provider,
                source_type=source_type,
                metadata=metadata,
            ),
            None,
        )

    @staticmethod
    def _canonicalize_url(url: str) -> str | None:
        if not url:
            return None
        try:
            parts = urlsplit(url)
            if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
                return None
            if parts.username is not None or parts.password is not None:
                return None
            parts.port
            netloc = parts.netloc.casefold()
            path = parts.path.rstrip("/")
            return urlunsplit(
                (parts.scheme.casefold(), netloc, path, parts.query, "")
            )
        except ValueError:
            return None

    @staticmethod
    def _append_finding(
        query: ResearchQuery,
        source: ResearchSource,
        source_ids: tuple[str, ...],
        business_findings: list[BusinessFinding],
        market_findings: list[MarketResearch],
        competitors: list[CompetitorResearch],
        leads: list[LeadResearch],
    ) -> None:
        if query.category is ResearchCategory.BUSINESS:
            business_findings.append(
                BusinessFinding(
                    topic=query.query,
                    summary=source.snippet,
                    source_ids=source_ids,
                )
            )
        elif query.category is ResearchCategory.MARKET:
            market_findings.append(
                MarketResearch(
                    topic=query.query,
                    summary=source.snippet,
                    source_ids=source_ids,
                )
            )
        elif query.category is ResearchCategory.COMPETITORS:
            competitors.append(
                CompetitorResearch(
                    name=source.title,
                    description=source.snippet,
                    relevance=f"Matched research query: {query.query}",
                    source_ids=source_ids,
                )
            )
        else:
            leads.append(
                LeadResearch(
                    name=source.title,
                    description=source.snippet,
                    relevance=f"Matched research query: {query.query}",
                    source_ids=source_ids,
                )
            )
