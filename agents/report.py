"""Deterministic business reports built from research and analysis results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Callable, TypeVar
from urllib.parse import urlsplit

from rag import BusinessProfile

from .analysis import (
    AnalysisResult,
    AnalysisStatus,
    BusinessOpportunity,
    LeadScore,
    MarketEvidenceStatus,
    MarketGap,
    MarketSignal,
    Recommendation,
)
from .research import (
    BusinessFinding,
    CompetitorResearch,
    LeadResearch,
    MarketResearch,
    ResearchIssue,
    ResearchResult,
    ResearchSource,
    ResearchStatus,
)


@dataclass(frozen=True, slots=True)
class ReportSourceReference:
    """A supplied source with a usable ID and HTTP(S) URL."""

    source_id: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class BusinessReport:
    """Structured report; its finding types are reused from Research and Analysis."""

    business_profile: BusinessProfile | None
    business_profile_id: str | None
    research_status: ResearchStatus | None
    analysis_status: AnalysisStatus | None
    market_evidence_status: MarketEvidenceStatus | None
    executive_summary: str
    business_findings: tuple[BusinessFinding, ...] = ()
    market_findings: tuple[MarketResearch, ...] = ()
    competitors: tuple[CompetitorResearch, ...] = ()
    leads: tuple[LeadResearch, ...] = ()
    lead_scores: tuple[LeadScore, ...] = ()
    market_signals: tuple[MarketSignal, ...] = ()
    market_gaps: tuple[MarketGap, ...] = ()
    opportunities: tuple[BusinessOpportunity, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    source_references: tuple[ReportSourceReference, ...] = ()
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    missing_data: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        """Render this report deterministically as Markdown."""
        return render_report(self)


T = TypeVar("T")


class ReportAgent:
    """Combine supplied research and analysis without generating new claims."""

    def generate(
        self,
        research_result: ResearchResult | None = None,
        analysis_result: AnalysisResult | None = None,
        *,
        profile: BusinessProfile | None = None,
    ) -> BusinessReport:
        if research_result is not None and not isinstance(research_result, ResearchResult):
            raise TypeError("research_result must be a ResearchResult or None.")
        if analysis_result is not None and not isinstance(analysis_result, AnalysisResult):
            raise TypeError("analysis_result must be an AnalysisResult or None.")
        if profile is not None and not isinstance(profile, BusinessProfile):
            raise TypeError("profile must be a BusinessProfile or None.")

        warnings: list[str] = []
        missing_data: list[str] = []
        research_status = (
            research_result.status
            if research_result is not None
            and isinstance(research_result.status, ResearchStatus)
            else None
        )
        analysis_status = (
            analysis_result.status
            if analysis_result is not None
            and isinstance(analysis_result.status, AnalysisStatus)
            else None
        )
        market_evidence_status = (
            analysis_result.market_status
            if analysis_result is not None
            and isinstance(analysis_result.market_status, MarketEvidenceStatus)
            else None
        )
        if research_result is not None and research_status is None:
            warnings.append("Research result has an invalid status; its status is not reported.")
        if analysis_result is not None and analysis_status is None:
            warnings.append("Analysis result has an invalid status; its status is not reported.")
        if analysis_result is not None and market_evidence_status is None:
            warnings.append("Analysis result has an invalid market evidence status.")
        if analysis_result is not None:
            self._append_analysis_warnings(analysis_result.warnings, warnings)

        research_id = (
            research_result.business_profile_id
            if research_result is not None
            and isinstance(research_result.business_profile_id, str)
            and research_result.business_profile_id.strip()
            else None
        )
        if research_result is not None and research_id is None:
            warnings.append("Research result has no valid business profile ID.")
        if profile is not None and research_id and profile.business_profile_id != research_id:
            warnings.append(
                "The supplied business profile ID does not match the research result; "
                "both are retained without assuming they describe the same profile."
            )

        sources = self._validated_sources(
            research_result.sources if research_result is not None else (),
            warnings,
        )
        if research_result is None:
            missing_data.append("No research result was supplied.")
        elif not sources:
            missing_data.append("No usable research source references are available.")

        business_findings = self._sanitize_items(
            research_result.business_findings if research_result else (),
            BusinessFinding,
            "business finding",
            sources,
            warnings,
        )
        market_findings = self._sanitize_items(
            research_result.market_findings if research_result else (),
            MarketResearch,
            "market finding",
            sources,
            warnings,
        )
        competitors = self._sanitize_items(
            research_result.competitors if research_result else (),
            CompetitorResearch,
            "competitor finding",
            sources,
            warnings,
        )
        leads = self._sanitize_items(
            research_result.leads if research_result else (),
            LeadResearch,
            "lead finding",
            sources,
            warnings,
        )
        if research_result is not None:
            self._append_research_issues(research_result.issues, warnings)

        lead_scores = self._sanitize_items(
            analysis_result.lead_scores if analysis_result else (),
            LeadScore,
            "lead score",
            sources,
            warnings,
        )
        market_signals = self._sanitize_items(
            analysis_result.market_signals if analysis_result else (),
            MarketSignal,
            "market signal",
            sources,
            warnings,
        )
        market_gaps = self._sanitize_items(
            analysis_result.market_gaps if analysis_result else (),
            MarketGap,
            "market gap",
            sources,
            warnings,
        )
        opportunities = self._sanitize_items(
            analysis_result.opportunities if analysis_result else (),
            BusinessOpportunity,
            "business opportunity",
            sources,
            warnings,
        )
        recommendations = self._sanitize_items(
            analysis_result.recommendations if analysis_result else (),
            Recommendation,
            "recommendation",
            sources,
            warnings,
        )

        if research_result is not None and isinstance(research_result.issues, Sequence):
            if research_result.issues:
                warnings.append(
                    f"Research reported {len(research_result.issues)} issue(s); findings may be incomplete."
                )
                if research_status is ResearchStatus.SUCCESS:
                    warnings.append(
                        "Research status is success despite reported issues; treat findings as incomplete."
                    )
        if research_status is ResearchStatus.PARTIAL:
            warnings.append("Research was partial; findings may be incomplete.")
        elif research_status is ResearchStatus.FAILED:
            warnings.append("Research failed; any supplied findings may be unavailable or incomplete.")
        if research_result is not None and not any(
            (business_findings, market_findings, competitors, leads)
        ):
            missing_data.append("Research supplied no usable findings.")
        if analysis_result is None:
            missing_data.append("No analysis result was supplied.")
            warnings.append("Analysis was not supplied; no analysis conclusions are included.")
        elif analysis_status is AnalysisStatus.FAILED:
            warnings.append("Analysis failed; no analysis conclusion should be relied upon.")
        elif analysis_status in (
            AnalysisStatus.PARTIAL,
            AnalysisStatus.INSUFFICIENT_EVIDENCE,
        ):
            warnings.append(f"Analysis status is {analysis_status.value}; conclusions are limited.")
        if analysis_result is not None and analysis_status is None:
            missing_data.append("Analysis status is unavailable.")
        if analysis_result is not None and not any(
            (
                lead_scores,
                market_signals,
                market_gaps,
                opportunities,
                recommendations,
            )
        ):
            missing_data.append("Analysis supplied no usable scored leads or conclusions.")

        source_urls = {source.url for source in sources.values()}
        if analysis_result is not None:
            self._check_analysis_urls(analysis_result.source_references, source_urls, warnings)

        limitations = [
            "Research snippets are source-reported and have not been independently verified.",
            "Lead scores are profile-match heuristics, not measured conversion probabilities.",
            "Market signals and potential gaps do not establish market demand or commercial viability.",
        ]
        if recommendations:
            limitations.append(
                "Recommendations are advisory only and do not guarantee outcomes; no action is performed and budget fit is not assumed."
            )
        next_steps: list[str] = []
        for recommendation in recommendations:
            step = recommendation.next_step.strip()
            if step:
                next_steps.append(step)
        if research_status in (ResearchStatus.PARTIAL, ResearchStatus.FAILED):
            next_steps.append(
                "Review the reported research issues and obtain additional source-backed information before relying on the report."
            )
        if not sources:
            next_steps.append("Collect source-backed research before drawing business conclusions.")

        unique_warnings = tuple(dict.fromkeys(warnings))
        unique_missing = tuple(dict.fromkeys(missing_data))
        unique_next_steps = tuple(dict.fromkeys(next_steps))
        summary = self._executive_summary(
            research_status,
            analysis_status,
            len(business_findings),
            len(market_findings),
            len(competitors),
            len(leads),
            len(lead_scores),
            len(market_signals),
            len(market_gaps),
            unique_missing,
        )

        return BusinessReport(
            business_profile=profile,
            business_profile_id=research_id,
            research_status=research_status,
            analysis_status=analysis_status,
            market_evidence_status=market_evidence_status,
            executive_summary=summary,
            business_findings=business_findings,
            market_findings=market_findings,
            competitors=competitors,
            leads=leads,
            lead_scores=lead_scores,
            market_signals=market_signals,
            market_gaps=market_gaps,
            opportunities=opportunities,
            recommendations=recommendations,
            source_references=tuple(
                ReportSourceReference(source.source_id, source.title, source.url)
                for source in sorted(sources.values(), key=lambda item: item.source_id)
            ),
            warnings=unique_warnings,
            limitations=tuple(limitations),
            missing_data=unique_missing,
            next_steps=unique_next_steps,
        )

    @staticmethod
    def render(report: BusinessReport) -> str:
        """Render a report as deterministic Markdown."""
        return render_report(report)

    @classmethod
    def _validated_sources(
        cls, values: object, warnings: list[str]
    ) -> dict[str, ResearchSource]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            if values:
                warnings.append("Research sources are malformed; source references were omitted.")
            return {}
        grouped: dict[str, list[ResearchSource]] = {}
        for index, value in enumerate(values):
            if not isinstance(value, ResearchSource):
                warnings.append(f"Research source entry {index} is malformed and was omitted.")
                continue
            source_id = value.source_id
            if (
                not isinstance(source_id, str)
                or not source_id.strip()
                or source_id != source_id.strip()
                or any(char.isspace() for char in source_id)
            ):
                warnings.append(f"Research source entry {index} has an invalid ID and was omitted.")
                continue
            grouped.setdefault(source_id, []).append(value)

        valid: dict[str, ResearchSource] = {}
        for source_id, entries in sorted(grouped.items()):
            source = entries[0]
            if any(item != source for item in entries[1:]):
                warnings.append(
                    f"Source ID {source_id!r} has conflicting records; all records for that ID were omitted."
                )
                continue
            if not all(
                isinstance(value, str) and value.strip()
                for value in (source.title, source.snippet, source.provider, source.source_type)
            ):
                warnings.append(f"Source {source_id!r} has malformed fields and was omitted.")
                continue
            if not cls._valid_http_url(source.url):
                warnings.append(f"Source {source_id!r} has an invalid URL and was omitted.")
                continue
            try:
                metadata_valid = isinstance(source.metadata, Mapping) and all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in source.metadata.items()
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                metadata_valid = False
            if not metadata_valid:
                warnings.append(f"Source {source_id!r} has malformed metadata and was omitted.")
                continue
            valid[source_id] = source
        return valid

    @staticmethod
    def _valid_http_url(url: object) -> bool:
        if not isinstance(url, str) or not url or any(char.isspace() for char in url):
            return False
        try:
            parsed = urlsplit(url)
            parsed.port
            return (
                parsed.scheme.casefold() in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            return False

    @staticmethod
    def _source_ids(value: object, label: str, sources: Mapping[str, ResearchSource],
                    warnings: list[str]) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            warnings.append(f"{label} has malformed source IDs; none were retained.")
            return ()
        ids: set[str] = set()
        invalid = False
        for source_id in value:
            if (
                not isinstance(source_id, str)
                or not source_id.strip()
                or source_id != source_id.strip()
                or any(char.isspace() for char in source_id)
                or source_id not in sources
            ):
                invalid = True
                continue
            ids.add(source_id)
        if invalid:
            warnings.append(
                f"{label} contains missing or invalid source IDs; those IDs were excluded."
            )
        elif not ids:
            warnings.append(f"{label} has no source IDs and cannot be traced to supplied research.")
        return tuple(sorted(ids))

    @classmethod
    def _sanitize_items(
        cls,
        values: object,
        item_type: type[T],
        label: str,
        sources: Mapping[str, ResearchSource],
        warnings: list[str],
    ) -> tuple[T, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            if values:
                warnings.append(f"{label.title()} collection is malformed and was omitted.")
            return ()
        cleaned: list[T] = []
        for index, item in enumerate(values):
            if not isinstance(item, item_type):
                warnings.append(f"{label.title()} {index} is malformed and was omitted.")
                continue
            if not cls._record_is_well_formed(item):
                warnings.append(f"{label.title()} {index} has malformed fields and was omitted.")
                continue
            ids = cls._source_ids(
                getattr(item, "source_ids", None),
                f"{label.title()} {index}",
                sources,
                warnings,
            )
            cleaned.append(replace(item, source_ids=ids))
        return tuple(cleaned)

    @staticmethod
    def _record_is_well_formed(item: object) -> bool:
        text_fields: dict[type[object], tuple[str, ...]] = {
            BusinessFinding: ("topic", "summary"),
            MarketResearch: ("topic", "summary"),
            CompetitorResearch: ("name", "description", "relevance"),
            LeadResearch: ("name", "description", "relevance"),
            LeadScore: (
                "lead_name",
                "priority",
                "rationale",
                "uncertainty",
            ),
            MarketSignal: ("description", "limitations"),
            MarketGap: (
                "description",
                "why_it_matters",
                "confidence",
                "limitations",
            ),
            BusinessOpportunity: (
                "description",
                "relevance",
                "potential",
                "risks",
            ),
            Recommendation: ("action", "why", "priority", "next_step", "risks"),
        }
        expected_type = type(item)
        fields = text_fields.get(expected_type, ())
        if not fields or any(not isinstance(getattr(item, name, None), str) for name in fields):
            return False
        if isinstance(item, LeadScore):
            if (
                not isinstance(item.score, int)
                or isinstance(item.score, bool)
                or not 0 <= item.score <= 100
                or not isinstance(item.rank, int)
                or isinstance(item.rank, bool)
                or item.rank < 1
            ):
                return False
        for field_name in ("supporting_evidence", "evidence", "missing_evidence", "matched_dimensions"):
            value = getattr(item, field_name, ())
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return False
            if any(not isinstance(entry, str) for entry in value):
                return False
        return True

    @staticmethod
    def _append_research_issues(values: object, warnings: list[str]) -> None:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            if values:
                warnings.append("Research issues are malformed and could not be listed.")
            return
        for index, issue in enumerate(values):
            if not isinstance(issue, ResearchIssue):
                warnings.append(f"Research issue {index} is malformed.")
                continue
            message = issue.message.strip() if isinstance(issue.message, str) else ""
            query = issue.query.strip() if isinstance(issue.query, str) else ""
            category = getattr(issue.category, "value", "unknown")
            if message:
                context = f" [{category}; query: {query}]" if query else f" [{category}]"
                warnings.append(f"Research issue{context}: {message}")
            else:
                warnings.append(f"Research issue {index} has no usable message.")

    @staticmethod
    def _append_analysis_warnings(values: object, warnings: list[str]) -> None:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            if values:
                warnings.append("Analysis warnings are malformed and could not be listed.")
            return
        for index, warning in enumerate(values):
            if isinstance(warning, str) and warning.strip():
                warnings.append(f"Analysis warning: {warning.strip()}")
            else:
                warnings.append(f"Analysis warning {index} is malformed.")

    @staticmethod
    def _check_analysis_urls(values: object, valid_urls: set[str], warnings: list[str]) -> None:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            if values:
                warnings.append("Analysis source references are malformed and were not retained.")
            return
        for url in values:
            if not isinstance(url, str) or url not in valid_urls:
                warnings.append(
                    "Analysis cited a source URL not present among valid research sources; "
                    "that citation is excluded from the report."
                )

    @staticmethod
    def _executive_summary(
        research_status: ResearchStatus | None,
        analysis_status: AnalysisStatus | None,
        business_count: int,
        market_count: int,
        competitor_count: int,
        lead_count: int,
        score_count: int,
        signal_count: int,
        gap_count: int,
        missing_data: Sequence[str],
    ) -> str:
        research_label = research_status.value if research_status else "not supplied or invalid"
        analysis_label = analysis_status.value if analysis_status else "not supplied or invalid"
        summary = (
            f"Research status: **{research_label}**. Analysis status: **{analysis_label}**. "
            f"Supplied findings: {business_count} business, {market_count} market, "
            f"{competitor_count} competitor, and {lead_count} lead; analysis supplied "
            f"{_count_label(score_count, 'lead score')}, "
            f"{_count_label(signal_count, 'market signal')}, and "
            f"{_count_label(gap_count, 'potential gap')}."
        )
        if missing_data:
            summary += " Missing or unavailable data: " + " ".join(missing_data)
        return summary


def _citations(source_ids: Sequence[str], sources: Mapping[str, ReportSourceReference]) -> str:
    citations = [
        f"[{source_id}]({sources[source_id].url})"
        for source_id in source_ids
        if source_id in sources
    ]
    return " Sources: " + ", ".join(citations) + "." if citations else ""


def _count_label(count: int, label: str) -> str:
    return f"{count} {label}{'' if count == 1 else 's'}"


def render_report(report: BusinessReport) -> str:
    """Render all supplied report sections without adding findings."""
    if not isinstance(report, BusinessReport):
        raise TypeError("report must be a BusinessReport.")
    source_map = {item.source_id: item for item in report.source_references}
    lines = [
        "# Business Report",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Business Context",
        "",
    ]
    if report.business_profile is None:
        lines.append("No business profile was supplied.")
    else:
        profile = report.business_profile
        lines.extend(
            (
                f"- **Offering:** {profile.offering}",
                f"- **Target customers:** {profile.target_customers}",
                f"- **Location:** {profile.location}",
                f"- **Budget:** {profile.budget} (not independently interpreted)",
                f"- **Goals:** {profile.goals}",
            )
        )
    if report.business_profile_id:
        lines.append(f"- **Research profile ID:** {report.business_profile_id}")

    _render_findings(
        lines,
        "Business Findings",
        report.business_findings,
        source_map,
        lambda item: f"**{item.topic}:** {item.summary}",
    )
    _render_findings(
        lines,
        "Market Findings",
        report.market_findings,
        source_map,
        lambda item: f"**{item.topic}:** {item.summary}",
    )
    _render_findings(
        lines,
        "Competitors",
        report.competitors,
        source_map,
        lambda item: f"**{item.name}:** {item.description} Relevance: {item.relevance}",
    )
    lines.extend(("", "## Leads and Scores", ""))
    if not report.leads and not report.lead_scores:
        lines.append("No lead findings or scores were supplied.")
    for lead in report.leads:
        lines.append(
            f"- **{lead.name}:** {lead.description} Relevance: {lead.relevance}"
            + _citations(lead.source_ids, source_map)
        )
    for score in report.lead_scores:
        evidence = f" Evidence: {'; '.join(score.evidence)}" if score.evidence else ""
        lines.append(
            f"- **{score.lead_name} - {score.score}/100 (rank {score.rank}, {score.priority}):** "
            f"{score.rationale} Uncertainty: {score.uncertainty}{evidence}"
            + _citations(score.source_ids, source_map)
        )

    lines.extend(("", "## Analysis", ""))
    lines.append(f"- **Analysis status:** {report.analysis_status.value if report.analysis_status else 'not supplied'}")
    lines.append(
        f"- **Market evidence status:** "
        f"{report.market_evidence_status.value if report.market_evidence_status else 'not supplied'}"
    )
    _render_findings(
        lines,
        "Market Signals",
        report.market_signals,
        source_map,
        lambda item: (
            f"{item.description} Evidence: {'; '.join(item.supporting_evidence)} "
            f"Limitation: {item.limitations}"
        ),
    )
    _render_findings(
        lines,
        "Potential Market Gaps",
        report.market_gaps,
        source_map,
        lambda item: (
            f"{item.description} Why it matters: {item.why_it_matters} "
            f"Evidence: {'; '.join(item.supporting_evidence)} "
            f"Evidence status: {item.confidence}. Limitations: {item.limitations}"
        ),
    )
    lines.extend(("", "### Business Opportunities", ""))
    if not report.opportunities:
        lines.append("No business opportunities were supplied.")
    for item in report.opportunities:
        lines.append(
            f"- **Potential only:** {item.description} Relevance: {item.relevance} "
            f"Potential: {item.potential}. Evidence: {'; '.join(item.supporting_evidence)} "
            f"Risks: {item.risks}"
            + _citations(item.source_ids, source_map)
        )

    lines.extend(("", "### Advisory Recommendations", ""))
    if not report.recommendations:
        lines.append("No recommendations were supplied.")
    for item in report.recommendations:
        lines.append(
            f"- **{item.priority}: {item.action}.** Why: {item.why} "
            f"Next step: {item.next_step} Risks: {item.risks}"
            + _citations(item.source_ids, source_map)
        )

    _render_findings(
        lines,
        "Source References",
        report.source_references,
        source_map,
        lambda item: f"[{item.source_id}]({item.url}) - {item.title}",
    )
    _render_text_items(lines, "Warnings", report.warnings)
    _render_text_items(lines, "Limitations", report.limitations)
    _render_text_items(lines, "Missing Data", report.missing_data)
    _render_text_items(lines, "Next Steps", report.next_steps)
    return "\n".join(lines).rstrip() + "\n"


def _render_findings(
    lines: list[str],
    heading: str,
    values: Sequence[T],
    sources: Mapping[str, ReportSourceReference],
    describe: Callable[[T], str],
) -> None:
    lines.extend(("", f"## {heading}", ""))
    if not values:
        lines.append(f"No {heading.casefold()} were supplied.")
        return
    for item in values:
        text = describe(item)
        source_ids = getattr(item, "source_ids", ())
        lines.append(f"- {text}" + _citations(source_ids, sources))


def _render_text_items(lines: list[str], heading: str, values: Sequence[str]) -> None:
    lines.extend(("", f"## {heading}", ""))
    if not values:
        lines.append(f"No {heading.casefold()} reported.")
        return
    lines.extend(f"- {item}" for item in values)
