"""Deterministic, source-traceable analysis of structured business research."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from rag import BusinessProfile

from .research import (
    LeadResearch,
    MarketResearch,
    ResearchResult,
    ResearchSource,
    ResearchStatus,
)


class AnalysisStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class MarketEvidenceStatus(str, Enum):
    UNKNOWN = "unknown"
    RESEARCH_SIGNAL = "research_signal"
    POTENTIAL_GAP = "potential_gap"


@dataclass(frozen=True, slots=True)
class LeadScore:
    """A 0..100 heuristic lead score; it is not a conversion probability."""

    lead_name: str
    score: int
    rank: int
    priority: str
    rationale: str
    source_ids: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    uncertainty: str = ""
    missing_evidence: tuple[str, ...] = ()
    matched_dimensions: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.lead_name


@dataclass(frozen=True, slots=True)
class MarketSignal:
    """A market statement present in a supplied source, not an independently verified fact."""

    description: str
    supporting_evidence: tuple[str, ...]
    source_ids: tuple[str, ...]
    limitations: str


@dataclass(frozen=True, slots=True)
class MarketGap:
    """An unvalidated potential gap inferred from a source-backed market signal."""

    description: str
    why_it_matters: str
    supporting_evidence: tuple[str, ...]
    source_ids: tuple[str, ...]
    confidence: str
    limitations: str = ""


@dataclass(frozen=True, slots=True)
class BusinessOpportunity:
    """A business opportunity to investigate, grounded in supplied research."""

    description: str
    relevance: str
    potential: str
    supporting_evidence: tuple[str, ...]
    source_ids: tuple[str, ...]
    risks: str = ""


@dataclass(frozen=True, slots=True)
class Recommendation:
    """An advisory action based on available research; it is not performed."""

    action: str
    why: str
    priority: str
    source_ids: tuple[str, ...]
    next_step: str
    risks: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Structured output from research synthesis."""

    status: AnalysisStatus
    lead_scores: tuple[LeadScore, ...] = ()
    market_signals: tuple[MarketSignal, ...] = ()
    market_gaps: tuple[MarketGap, ...] = ()
    market_status: MarketEvidenceStatus = MarketEvidenceStatus.UNKNOWN
    opportunities: tuple[BusinessOpportunity, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    source_references: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


_MARKET_SIGNAL_PHRASES = (
    "unmet need",
    "unmet demand",
    "underserved",
    "under-served",
    "limited access",
    "not currently served",
    "not served",
    "demand exceeds supply",
)
_NEGATION_WORDS = frozenset(
    {"no", "not", "never", "without", "isnt", "doesnt", "cannot", "lack"}
)
_PROFILE_SCORE_POINTS = (
    ("offering", 30, "business offering"),
    ("target_customers", 30, "target customers"),
    ("location", 20, "business location"),
    ("goals", 20, "business goal"),
)
_CORE_LEAD_DIMENSIONS = frozenset({"offering", "target_customers"})
_BUDGET_CAVEAT = (
    "No budget fit is assumed. Verify the cost and available budget before "
    "committing resources."
)
_NOT_PROBABILITY = (
    "This heuristic score is not a measured probability of conversion."
)


class AnalysisAgent:
    """Analyze a ResearchResult without an LLM, randomness, or external services.

    The lead score is the sum of exact profile-context matches:
    offering +30, target customers +30, location +20, and goals +20.
    A contribution is awarded only when the normalized profile phrase occurs
    in both the lead's description/relevance and a valid linked source snippet.
    Scores are clamped to 0..100. Scores rank evidence alignment only; they do
    not estimate likelihood of conversion. Ties sort by case-folded lead name,
    then original name, then valid source IDs.
    """

    SCORE_MIN = 0
    SCORE_MAX = 100

    def __init__(self, profile: BusinessProfile | None = None) -> None:
        if profile is not None and not isinstance(profile, BusinessProfile):
            raise TypeError("profile must be a BusinessProfile or None.")
        self._profile = profile

    @property
    def profile(self) -> BusinessProfile | None:
        return self._profile

    def analyze(
        self,
        research_result: ResearchResult,
        *,
        profile: BusinessProfile | None = None,
    ) -> AnalysisResult:
        if not isinstance(research_result, ResearchResult):
            raise TypeError("research_result must be a ResearchResult.")
        selected_profile = profile if profile is not None else self._profile
        profile_context = self._normalize_profile(selected_profile)
        self._validate_result_shape(research_result)

        warnings: list[str] = []
        if not isinstance(research_result.status, ResearchStatus):
            raise ValueError("research_result.status must be a ResearchStatus.")
        if research_result.status is ResearchStatus.FAILED:
            warnings.append(
                "Research execution failed; any available analysis is limited to partial results."
            )
        if research_result.issues:
            warnings.append(
                f"Research result contains {len(research_result.issues)} issue(s); treat findings as incomplete."
            )
            if any(not hasattr(issue, "message") for issue in research_result.issues):
                warnings.append("One or more research issues are malformed.")

        source_map, source_warnings = self._validated_sources(research_result.sources)
        warnings.extend(source_warnings)
        if not source_map:
            warnings.append("No usable source references were supplied.")

        valid_leads = self._validated_leads(research_result.leads, source_map, warnings)
        valid_market_findings = self._validated_market_findings(
            research_result.market_findings, source_map, warnings
        )
        self._warn_unreferenced_findings(
            research_result, source_map, warnings
        )

        lead_scores = self._score_leads(valid_leads, profile_context, source_map)
        if lead_scores and all(score.score == 0 for score in lead_scores):
            warnings.append(
                "All leads have zero evidence-alignment scores; their supporting material "
                "did not corroborate any available business-profile dimension."
            )
        elif lead_scores and not any(
            _CORE_LEAD_DIMENSIONS.issubset(score.matched_dimensions)
            for score in lead_scores
        ):
            warnings.append(
                "No leads have corroborated both offering and target-customer dimensions; "
                "available lead evidence is weak and does not establish business fit."
            )
        if not lead_scores:
            warnings.append("No source-supported leads were available to score.")

        market_signals, market_gaps = self._analyze_market(
            valid_market_findings, profile_context, source_map
        )
        if not market_signals:
            warnings.append(
                "Market need is unknown: no supplied, valid market source corroborates an explicit unmet-need signal."
            )

        opportunities = self._build_opportunities(
            lead_scores, market_gaps, profile_context, source_map
        )
        recommendations = self._build_recommendations(
            lead_scores, market_gaps, profile_context
        )
        source_references = self._collect_source_references(source_map)
        market_status = self._market_status(market_signals, market_gaps)
        status = self._determine_status(
            research_result.status,
            lead_scores,
            market_signals,
            market_gaps,
            source_map,
            source_warnings,
            warnings,
        )

        return AnalysisResult(
            status=status,
            lead_scores=lead_scores,
            market_signals=market_signals,
            market_gaps=market_gaps,
            market_status=market_status,
            opportunities=opportunities,
            recommendations=recommendations,
            source_references=source_references,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def analysis(
        self,
        research_result: ResearchResult,
        *,
        profile: BusinessProfile | None = None,
    ) -> AnalysisResult:
        """Compatibility alias for :meth:`analyze`."""
        return self.analyze(research_result, profile=profile)

    @staticmethod
    def _normalize_profile(profile: BusinessProfile | None) -> dict[str, str]:
        if profile is None:
            return {}
        if not isinstance(profile, BusinessProfile):
            raise TypeError("profile must be a BusinessProfile or None.")
        values = {
            "offering": profile.offering,
            "target_customers": profile.target_customers,
            "location": profile.location,
            "budget": profile.budget,
            "goals": profile.goals,
            "business_profile_id": profile.business_profile_id,
        }
        invalid = [
            name
            for name, value in values.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid:
            raise ValueError(f"Invalid profile values for: {', '.join(invalid)}")
        return {name: value.strip() for name, value in values.items()}

    @staticmethod
    def _validate_result_shape(result: ResearchResult) -> None:
        if not isinstance(result.business_profile_id, str) or not result.business_profile_id.strip():
            raise ValueError("research_result.business_profile_id must be non-empty text.")
        for field_name in (
            "business_findings",
            "market_findings",
            "competitors",
            "leads",
            "sources",
            "issues",
        ):
            values = getattr(result, field_name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"research_result.{field_name} must be a sequence.")

    @staticmethod
    def _valid_url(value: object) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(ord(char) < 32 or char.isspace() for char in value):
            return False
        try:
            parsed = urlsplit(value.strip())
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
                return False
            if parsed.username is not None or parsed.password is not None:
                return False
            parsed.port
        except ValueError:
            return False
        return True

    @classmethod
    def _validated_sources(
        cls, sources: Sequence[object]
    ) -> tuple[dict[str, ResearchSource], tuple[str, ...]]:
        warnings: list[str] = []
        grouped: dict[str, list[ResearchSource]] = defaultdict(list)
        for index, value in enumerate(sources):
            if not isinstance(value, ResearchSource):
                warnings.append(f"Source entry {index} is malformed and was ignored.")
                continue
            source_id = value.source_id
            if (
                not isinstance(source_id, str)
                or not source_id.strip()
                or any(char.isspace() for char in source_id)
            ):
                warnings.append(f"Source entry {index} has an invalid source ID and was ignored.")
                continue
            grouped[source_id].append(value)

        valid: dict[str, ResearchSource] = {}
        for source_id in sorted(grouped):
            entries = grouped[source_id]
            first = entries[0]
            if any(entry != first for entry in entries[1:]):
                warnings.append(
                    f"Source ID {source_id!r} is duplicated with conflicting records and was excluded."
                )
                continue
            required_text = (first.title, first.snippet, first.provider, first.source_type)
            if any(not isinstance(item, str) or not item.strip() for item in required_text):
                warnings.append(f"Source {source_id!r} has malformed required text and was ignored.")
                continue
            if not cls._valid_url(first.url):
                warnings.append(f"Source {source_id!r} has an invalid URL and was ignored.")
                continue
            try:
                metadata_valid = isinstance(first.metadata, Mapping) and all(
                    isinstance(key, str) and isinstance(item, str)
                    for key, item in first.metadata.items()
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                metadata_valid = False
            if not metadata_valid:
                warnings.append(f"Source {source_id!r} has malformed metadata and was ignored.")
                continue
            valid[source_id] = first
        return valid, tuple(warnings)

    @staticmethod
    def _validated_leads(
        leads: Sequence[object],
        source_map: Mapping[str, ResearchSource],
        warnings: list[str],
    ) -> tuple[tuple[LeadResearch, tuple[str, ...]], ...]:
        validated: list[tuple[LeadResearch, tuple[str, ...]]] = []
        for index, lead in enumerate(leads):
            if not isinstance(lead, LeadResearch):
                warnings.append(f"Lead entry {index} is malformed and was ignored.")
                continue
            if not isinstance(lead.name, str) or not lead.name.strip() or any(
                not isinstance(item, str)
                for item in (lead.description, lead.relevance)
            ):
                warnings.append(f"Lead entry {index} has malformed fields and was ignored.")
                continue
            source_ids = AnalysisAgent._valid_finding_source_ids(
                lead.source_ids, source_map, f"lead {lead.name!r}", warnings
            )
            validated.append((lead, source_ids))
        return tuple(
            sorted(
                validated,
                key=lambda item: (
                    item[0].name.casefold(),
                    item[0].name,
                    item[0].description.casefold(),
                    item[0].description,
                    item[0].relevance.casefold(),
                    item[0].relevance,
                    item[1],
                ),
            )
        )

    @staticmethod
    def _validated_market_findings(
        findings: Sequence[object],
        source_map: Mapping[str, ResearchSource],
        warnings: list[str],
    ) -> tuple[tuple[MarketResearch, tuple[str, ...]], ...]:
        validated: list[tuple[MarketResearch, tuple[str, ...]]] = []
        for index, finding in enumerate(findings):
            if not isinstance(finding, MarketResearch):
                warnings.append(f"Market finding {index} is malformed and was ignored.")
                continue
            if not isinstance(finding.topic, str) or not isinstance(finding.summary, str):
                warnings.append(f"Market finding {index} has malformed fields and was ignored.")
                continue
            source_ids = AnalysisAgent._valid_finding_source_ids(
                finding.source_ids,
                source_map,
                f"market finding {finding.topic!r}",
                warnings,
            )
            if source_ids:
                validated.append((finding, source_ids))
        return tuple(validated)

    @staticmethod
    def _valid_finding_source_ids(
        source_ids: object,
        source_map: Mapping[str, ResearchSource],
        finding_label: str,
        warnings: list[str],
    ) -> tuple[str, ...]:
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
            warnings.append(f"{finding_label} has malformed source references.")
            return ()
        valid_ids: set[str] = set()
        invalid_found = False
        for source_id in source_ids:
            if (
                not isinstance(source_id, str)
                or not source_id.strip()
                or source_id != source_id.strip()
                or any(char.isspace() for char in source_id)
            ):
                invalid_found = True
                continue
            if source_id not in source_map:
                invalid_found = True
                continue
            valid_ids.add(source_id)
        if invalid_found:
            warnings.append(
                f"{finding_label} contains missing or invalid source references; those references were excluded."
            )
        if not valid_ids:
            warnings.append(f"{finding_label} has no valid supporting source.")
            return ()
        return tuple(sorted(valid_ids))

    @staticmethod
    def _warn_unreferenced_findings(
        result: ResearchResult,
        source_map: Mapping[str, ResearchSource],
        warnings: list[str],
    ) -> None:
        for field_name, findings in (
            ("business", result.business_findings),
            ("competitor", result.competitors),
        ):
            for index, finding in enumerate(findings):
                if not hasattr(finding, "source_ids"):
                    warnings.append(f"{field_name.title()} finding {index} is malformed.")
                    continue
                AnalysisAgent._valid_finding_source_ids(
                    finding.source_ids,
                    source_map,
                    f"{field_name} finding {index}",
                    warnings,
                )

    @staticmethod
    def _normalize_words(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+", text.casefold()))

    @classmethod
    def _contains_phrase(cls, text: str, phrase: str) -> bool:
        words = cls._normalize_words(text)
        phrase_words = cls._normalize_words(phrase)
        if not words or not phrase_words or len(phrase_words) > len(words):
            return False
        for start in range(len(words) - len(phrase_words) + 1):
            if words[start : start + len(phrase_words)] != phrase_words:
                continue
            preceding = words[max(0, start - 4) : start]
            if any(word in _NEGATION_WORDS for word in preceding):
                continue
            return True
        return False

    @classmethod
    def _score_leads(
        cls,
        leads: Sequence[tuple[LeadResearch, tuple[str, ...]]],
        profile_context: Mapping[str, str],
        source_map: Mapping[str, ResearchSource],
    ) -> tuple[LeadScore, ...]:
        scored: list[LeadScore] = []
        for lead, source_ids in leads:
            lead_text = f"{lead.description} {lead.relevance}"
            source_snippets = tuple(source_map[source_id].snippet for source_id in source_ids)
            awarded: list[tuple[str, int, str]] = []
            missing: list[str] = []
            for field_name, points, label in _PROFILE_SCORE_POINTS:
                phrase = profile_context.get(field_name)
                if not phrase:
                    missing.append(f"No business profile {label} was supplied.")
                    continue
                lead_matches = cls._contains_phrase(lead_text, phrase)
                source_matches = any(
                    cls._contains_phrase(snippet, phrase) for snippet in source_snippets
                )
                if lead_matches and source_matches:
                    awarded.append((field_name, points, label))
                else:
                    missing.append(
                        f"No corroborated source-backed match for {label} ({phrase})."
                    )
            if not source_ids:
                missing.append("No valid research source references were supplied.")

            score = min(cls.SCORE_MAX, max(cls.SCORE_MIN, sum(points for _, points, _ in awarded)))
            contribution_text = (
                ", ".join(f"{label} +{points}" for _, points, label in awarded)
                if awarded
                else "no corroborated profile matches (+0)"
            )
            rationale = f"Score {score}/100 = {contribution_text}."
            if missing:
                rationale += " Missing/uncorroborated evidence: " + " ".join(missing)
            rationale += f" {_NOT_PROBABILITY}"
            evidence = tuple(
                source_map[source_id].snippet for source_id in source_ids
            )
            if _CORE_LEAD_DIMENSIONS.issubset(
                field_name for field_name, _, _ in awarded
            ):
                uncertainty = _NOT_PROBABILITY
            else:
                uncertainty = (
                    "Weak evidence: the business-offering and target-customer dimensions "
                    "were not both corroborated, so profile-term overlap does not establish lead fit. "
                    f"{_NOT_PROBABILITY}"
                )
            scored.append(
                LeadScore(
                    lead_name=lead.name.strip(),
                    score=score,
                    rank=0,
                    priority=cls._priority_for_score(score),
                    rationale=rationale,
                    source_ids=source_ids,
                    evidence=evidence,
                    uncertainty=uncertainty,
                    missing_evidence=tuple(missing),
                    matched_dimensions=tuple(field_name for field_name, _, _ in awarded),
                )
            )

        ordered = sorted(
            scored,
            key=lambda item: (
                -item.score,
                item.lead_name.casefold(),
                item.lead_name,
                item.source_ids,
            ),
        )
        return tuple(
            LeadScore(
                lead_name=item.lead_name,
                score=item.score,
                rank=rank,
                priority=item.priority,
                rationale=item.rationale,
                source_ids=item.source_ids,
                evidence=item.evidence,
                uncertainty=item.uncertainty,
                missing_evidence=item.missing_evidence,
                matched_dimensions=item.matched_dimensions,
            )
            for rank, item in enumerate(ordered, start=1)
        )

    @staticmethod
    def _priority_for_score(score: int) -> str:
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    @classmethod
    def _analyze_market(
        cls,
        findings: Sequence[tuple[MarketResearch, tuple[str, ...]]],
        profile_context: Mapping[str, str],
        source_map: Mapping[str, ResearchSource],
    ) -> tuple[tuple[MarketSignal, ...], tuple[MarketGap, ...]]:
        signals: list[MarketSignal] = []
        gaps: list[MarketGap] = []
        for finding, source_ids in findings:
            for source_id in source_ids:
                source = source_map[source_id]
                explicit_phrases = tuple(
                    phrase
                    for phrase in _MARKET_SIGNAL_PHRASES
                    if cls._contains_phrase(finding.summary, phrase)
                    and cls._contains_phrase(source.snippet, phrase)
                )
                if not explicit_phrases:
                    continue
                evidence = source.snippet.strip()
                signal = MarketSignal(
                    description=f"Source-reported market signal: {evidence}",
                    supporting_evidence=(evidence,),
                    source_ids=(source_id,),
                    limitations=(
                        "This is a provider-supplied snippet, not independently verified market data "
                        "or proof of unmet demand."
                    ),
                )
                if signal not in signals:
                    signals.append(signal)
                relevant_profile_fields = (
                    profile_context.get("offering", ""),
                    profile_context.get("target_customers", ""),
                )
                if not any(
                    phrase
                    and cls._contains_phrase(evidence, phrase)
                    for phrase in relevant_profile_fields
                ):
                    continue
                gap = MarketGap(
                    description=f"Unvalidated potential gap to investigate: {evidence}",
                    why_it_matters=(
                        "The source-reported signal overlaps the supplied business offering or target customers; "
                        "whether it represents unmet demand remains unknown."
                    ),
                    supporting_evidence=(evidence,),
                    source_ids=(source_id,),
                    confidence="unvalidated-potential-gap",
                    limitations=(
                        "A source snippet and profile overlap do not establish market size, demand, "
                        "competitor weakness, or commercial viability. Validate independently before acting."
                    ),
                )
                if gap not in gaps:
                    gaps.append(gap)
        signals.sort(key=lambda item: (item.source_ids, item.description.casefold()))
        gaps.sort(key=lambda item: (item.source_ids, item.description.casefold()))
        return tuple(signals), tuple(gaps)

    @staticmethod
    def _build_opportunities(
        lead_scores: Sequence[LeadScore],
        market_gaps: Sequence[MarketGap],
        profile_context: Mapping[str, str],
        source_map: Mapping[str, ResearchSource],
    ) -> tuple[BusinessOpportunity, ...]:
        opportunities: list[BusinessOpportunity] = []
        for lead in lead_scores:
            if (
                lead.score <= 0
                or not lead.source_ids
                or not _CORE_LEAD_DIMENSIONS.issubset(lead.matched_dimensions)
            ):
                continue
            goals = profile_context.get("goals")
            target = profile_context.get("target_customers")
            relevance = "Investigate the lead's fit using the linked research."
            if goals and target:
                relevance = (
                    f"Investigate whether the lead could help serve {target} in support of {goals}; "
                    "the current evidence does not establish a business relationship."
                )
            opportunities.append(
                BusinessOpportunity(
                    description=f"Validate potential fit with {lead.lead_name}.",
                    relevance=relevance,
                    potential="qualitative; requires validation",
                    supporting_evidence=tuple(
                        source_map[source_id].snippet for source_id in lead.source_ids
                    ),
                    source_ids=lead.source_ids,
                    risks=(
                        "Research alignment does not establish intent, purchasing authority, "
                        "conversion likelihood, or a commercial relationship."
                    ),
                )
            )
        for gap in market_gaps:
            opportunities.append(
                BusinessOpportunity(
                    description="Validate whether the reported market signal supports a viable opportunity.",
                    relevance=gap.why_it_matters,
                    potential="unvalidated; no numeric estimate is supported",
                    supporting_evidence=gap.supporting_evidence,
                    source_ids=gap.source_ids,
                    risks=gap.limitations,
                )
            )
        return tuple(opportunities)

    @staticmethod
    def _build_recommendations(
        lead_scores: Sequence[LeadScore],
        market_gaps: Sequence[MarketGap],
        profile_context: Mapping[str, str],
    ) -> tuple[Recommendation, ...]:
        recommendations: list[Recommendation] = []
        for lead in lead_scores:
            if (
                lead.score <= 0
                or not lead.source_ids
                or not _CORE_LEAD_DIMENSIONS.issubset(lead.matched_dimensions)
            ):
                continue
            offering = profile_context.get("offering", "the business offering")
            target = profile_context.get("target_customers", "the supplied target customers")
            goals = profile_context.get("goals", "the supplied business goals")
            recommendations.append(
                Recommendation(
                    action=f"Review the source-backed fit of {lead.lead_name}",
                    why=(
                        f"Its score is {lead.score}/100 under the documented profile-match heuristic, "
                        f"relevant to {offering}, {target}, and {goals}."
                    ),
                    priority=AnalysisAgent._priority_for_score(lead.score),
                    source_ids=lead.source_ids,
                    next_step=(
                        "Read the linked source material and confirm the lead's identity, relevance, and "
                        "any next action before proceeding."
                    ),
                    risks=f"{_NOT_PROBABILITY} {_BUDGET_CAVEAT}",
                )
            )
        for gap in market_gaps:
            recommendations.append(
                Recommendation(
                    action="Validate the reported market signal before planning spend",
                    why=gap.why_it_matters,
                    priority="Medium",
                    source_ids=gap.source_ids,
                    next_step=(
                        "Seek independent evidence for the target need and compare any proposed test cost "
                        "with the available budget."
                    ),
                    risks=f"{gap.limitations} {_BUDGET_CAVEAT}",
                )
            )
        order = {"High": 0, "Medium": 1, "Low": 2}
        recommendations.sort(
            key=lambda item: (
                order.get(item.priority, 3),
                item.action.casefold(),
                item.action,
                item.source_ids,
            )
        )
        return tuple(recommendations)

    @staticmethod
    def _collect_source_references(
        source_map: Mapping[str, ResearchSource],
    ) -> tuple[str, ...]:
        urls = {
            source.url.strip()
            for source in source_map.values()
            if AnalysisAgent._valid_url(source.url)
        }
        return tuple(sorted(urls, key=lambda url: (url.casefold(), url)))

    @staticmethod
    def _market_status(
        signals: Sequence[MarketSignal], gaps: Sequence[MarketGap]
    ) -> MarketEvidenceStatus:
        if gaps:
            return MarketEvidenceStatus.POTENTIAL_GAP
        if signals:
            return MarketEvidenceStatus.RESEARCH_SIGNAL
        return MarketEvidenceStatus.UNKNOWN

    @staticmethod
    def _determine_status(
        research_status: ResearchStatus,
        lead_scores: Sequence[LeadScore],
        market_signals: Sequence[MarketSignal],
        market_gaps: Sequence[MarketGap],
        source_map: Mapping[str, ResearchSource],
        source_warnings: Sequence[str],
        warnings: Sequence[str],
    ) -> AnalysisStatus:
        substantive_evidence = any(
            _CORE_LEAD_DIMENSIONS.issubset(score.matched_dimensions)
            for score in lead_scores
        ) or bool(market_signals or market_gaps)
        if research_status is ResearchStatus.FAILED:
            return AnalysisStatus.PARTIAL if substantive_evidence else AnalysisStatus.FAILED
        if research_status is ResearchStatus.PARTIAL or any(
            "research result contains" in warning.lower() for warning in warnings
        ):
            return AnalysisStatus.PARTIAL
        if not substantive_evidence or not source_map:
            return AnalysisStatus.INSUFFICIENT_EVIDENCE
        if source_warnings:
            return AnalysisStatus.PARTIAL
        if any(
            "invalid source" in warning.lower()
            or "malformed" in warning.lower()
            or "invalid url" in warning.lower()
            or "source references" in warning.lower()
            for warning in warnings
        ):
            return AnalysisStatus.PARTIAL
        return AnalysisStatus.SUCCESS


analysis_agent = AnalysisAgent
