"""Deterministic orchestration of the existing business agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from urllib.parse import urlsplit

from rag import BusinessProfile

from .analysis import (
    AnalysisAgent,
    AnalysisResult,
    AnalysisStatus,
    BusinessOpportunity,
    LeadScore,
    MarketGap,
    MarketSignal,
    Recommendation,
)
from .report import BusinessReport, ReportAgent, ReportSourceReference
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


class StageStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True, slots=True)
class WorkflowStage:
    name: str
    status: StageStatus
    attempts: int = 0
    message: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    status: WorkflowStatus
    stages: tuple[WorkflowStage, ...]
    research_result: ResearchResult | None = None
    analysis_result: AnalysisResult | None = None
    report: BusinessReport | None = None
    warnings: tuple[str, ...] = ()


class _ResearchAgent(Protocol):
    def research(self, profile: BusinessProfile) -> ResearchResult:
        ...


class _AnalysisAgent(Protocol):
    def analyze(
        self, research_result: ResearchResult, *, profile: BusinessProfile
    ) -> AnalysisResult:
        ...


class _ReportAgent(Protocol):
    def generate(
        self,
        research_result: ResearchResult | None = None,
        analysis_result: AnalysisResult | None = None,
        *,
        profile: BusinessProfile | None = None,
    ) -> BusinessReport:
        ...


_RESEARCH_STAGE = "research"
_ANALYSIS_STAGE = "analysis"
_REPORT_STAGE = "report"
_VALIDATION_STAGE = "validation"
_STAGE_ORDER = (
    _VALIDATION_STAGE,
    _RESEARCH_STAGE,
    _ANALYSIS_STAGE,
    _REPORT_STAGE,
)


class ManagerAgent:
    """Run Research, Analysis, and Report in dependency order.

    ``research_retry_limit`` counts retries after the first attempt. It is
    applied only to unexpected exceptions and explicit FAILED research results;
    validation errors and quality-check failures are not retried.
    """

    def __init__(
        self,
        research_agent: _ResearchAgent,
        analysis_agent: _AnalysisAgent | None = None,
        report_agent: _ReportAgent | None = None,
        *,
        research_retry_limit: int = 1,
    ) -> None:
        if not callable(getattr(research_agent, "research", None)):
            raise TypeError("research_agent must implement research(profile).")
        if analysis_agent is not None and not callable(
            getattr(analysis_agent, "analyze", None)
        ):
            raise TypeError("analysis_agent must implement analyze(research_result).")
        if report_agent is not None and not callable(
            getattr(report_agent, "generate", None)
        ):
            raise TypeError(
                "report_agent must implement generate(research_result, analysis_result)."
            )
        if (
            not isinstance(research_retry_limit, int)
            or isinstance(research_retry_limit, bool)
            or research_retry_limit < 0
        ):
            raise ValueError("research_retry_limit must be a non-negative integer.")

        self._research_agent = research_agent
        self._analysis_agent = analysis_agent or AnalysisAgent()
        self._report_agent = report_agent or ReportAgent()
        self._research_retry_limit = research_retry_limit

    def run(self, profile: BusinessProfile) -> WorkflowResult:
        """Run a workflow for one validated business profile."""
        invalid_profile = self._profile_validation_error(profile)
        if invalid_profile is not None:
            warning = f"Business profile validation failed: {invalid_profile}"
            return WorkflowResult(
                status=WorkflowStatus.INVALID_INPUT,
                stages=(
                    WorkflowStage(
                        _VALIDATION_STAGE,
                        StageStatus.FAILED,
                        message=warning,
                        warnings=(warning,),
                    ),
                    *(
                        WorkflowStage(name, StageStatus.SKIPPED, message=warning)
                        for name in _STAGE_ORDER[1:]
                    ),
                ),
                warnings=(warning,),
            )

        research_result, research_stage = self._run_research(profile)
        stages = [
            WorkflowStage(
                _VALIDATION_STAGE,
                StageStatus.COMPLETED,
                message="Business profile validated.",
            ),
            research_stage,
        ]
        warnings = list(research_stage.warnings)
        if research_result is None:
            stages.extend(
                WorkflowStage(
                    name,
                    StageStatus.SKIPPED,
                    message=f"Skipped because the {_RESEARCH_STAGE} stage produced no usable result.",
                )
                for name in (_ANALYSIS_STAGE, _REPORT_STAGE)
            )
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                stages=tuple(stages),
                warnings=tuple(warnings),
            )

        analysis_result, analysis_stage = self._run_analysis(profile, research_result)
        stages.append(analysis_stage)
        warnings.extend(analysis_stage.warnings)

        report, report_stage = self._run_report(
            profile, research_result, analysis_result
        )
        stages.append(report_stage)
        warnings.extend(report_stage.warnings)

        status = self._workflow_status(stages)
        return WorkflowResult(
            status=status,
            stages=tuple(stages),
            research_result=research_result,
            analysis_result=analysis_result,
            report=report,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _profile_validation_error(profile: object) -> str | None:
        if not isinstance(profile, BusinessProfile):
            return "profile must be a BusinessProfile."
        for field_name in (
            "offering",
            "target_customers",
            "location",
            "budget",
            "goals",
            "business_profile_id",
        ):
            value = getattr(profile, field_name, None)
            if not isinstance(value, str) or not value.strip():
                return f"profile.{field_name} must be non-empty text."
        return None

    def _run_research(
        self, profile: BusinessProfile
    ) -> tuple[ResearchResult | None, WorkflowStage]:
        attempts = 0
        last_error = ""
        while attempts <= self._research_retry_limit:
            attempts += 1
            try:
                result = self._research_agent.research(profile)
            except (TypeError, ValueError) as exc:
                message = f"Research input or validation error: {exc}"
                return None, WorkflowStage(
                    _RESEARCH_STAGE,
                    StageStatus.FAILED,
                    attempts,
                    message,
                    (message,),
                )
            except Exception as exc:
                last_error = f"Research attempt {attempts} failed: {type(exc).__name__}: {exc}"
                if attempts <= self._research_retry_limit:
                    continue
                return None, WorkflowStage(
                    _RESEARCH_STAGE,
                    StageStatus.FAILED,
                    attempts,
                    last_error,
                    (last_error,),
                )

            if not isinstance(result, ResearchResult):
                message = "Research agent returned an invalid result type."
                return None, WorkflowStage(
                    _RESEARCH_STAGE,
                    StageStatus.FAILED,
                    attempts,
                    message,
                    (message,),
                )
            if not isinstance(result.status, ResearchStatus):
                message = "Research result has an invalid status."
                return result, WorkflowStage(
                    _RESEARCH_STAGE,
                    StageStatus.FAILED,
                    attempts,
                    message,
                    (message,),
                )
            quality_warnings = self._research_quality_warnings(result, profile)
            if (
                result.status is ResearchStatus.FAILED
                and attempts <= self._research_retry_limit
                and self._research_failure_is_retryable(result, profile)
            ):
                last_error = (
                    "Research returned FAILED status; retrying within the configured limit."
                )
                continue
            if result.status is ResearchStatus.FAILED:
                quality_warnings.append(
                    "Research remained FAILED after the configured attempts."
                )
                return result, WorkflowStage(
                    _RESEARCH_STAGE,
                    StageStatus.FAILED,
                    attempts,
                    "Research returned FAILED status.",
                    tuple(dict.fromkeys(quality_warnings)),
                )
            if result.status is ResearchStatus.PARTIAL or quality_warnings:
                message = (
                    "Research completed with partial data or quality limitations."
                    if quality_warnings
                    else "Research returned PARTIAL status."
                )
                return result, WorkflowStage(
                    _RESEARCH_STAGE,
                    StageStatus.PARTIAL,
                    attempts,
                    message,
                    tuple(dict.fromkeys(quality_warnings)),
                )
            return result, WorkflowStage(
                _RESEARCH_STAGE,
                StageStatus.COMPLETED,
                attempts,
                "Research completed.",
            )

        message = last_error or "Research did not produce a result."
        return None, WorkflowStage(
            _RESEARCH_STAGE, StageStatus.FAILED, attempts, message, (message,)
        )

    @staticmethod
    def _research_quality_warnings(
        result: ResearchResult, profile: BusinessProfile
    ) -> list[str]:
        warnings: list[str] = []
        if result.business_profile_id != profile.business_profile_id:
            warnings.append(
                "Research business profile ID does not match the requested profile."
            )
        if not isinstance(result.status, ResearchStatus):
            warnings.append("Research result has an invalid status.")
            return warnings
        sources_value = result.sources
        if not isinstance(sources_value, Sequence) or isinstance(
            sources_value, (str, bytes)
        ):
            return ["Research source collection is malformed."]
        source_ids: set[str] = set()
        for source in sources_value:
            if (
                not isinstance(source, ResearchSource)
                or not isinstance(source.source_id, str)
                or not source.source_id.strip()
                or source.source_id != source.source_id.strip()
                or any(character.isspace() for character in source.source_id)
                or not all(
                    isinstance(value, str) and value.strip()
                    for value in (
                        source.title,
                        source.url,
                        source.snippet,
                        source.provider,
                        source.source_type,
                    )
                )
                or not ManagerAgent._valid_source_url(source.url)
            ):
                warnings.append("Research contains a malformed source record.")
                continue
            if source.source_id in source_ids:
                warnings.append("Research contains duplicate source IDs.")
            source_ids.add(source.source_id)
        finding_types = {
            "business_findings": BusinessFinding,
            "market_findings": MarketResearch,
            "competitors": CompetitorResearch,
            "leads": LeadResearch,
        }
        has_findings = False
        for field_name, finding_type in finding_types.items():
            findings = getattr(result, field_name)
            if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
                warnings.append(f"Research {field_name} collection is malformed.")
                continue
            has_findings = has_findings or bool(findings)
            for index, finding in enumerate(findings):
                if not isinstance(finding, finding_type):
                    warnings.append(f"Research {field_name} finding {index} is malformed.")
                    continue
                ids = getattr(finding, "source_ids", None)
                if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                    warnings.append(f"Research {field_name} finding {index} has malformed source IDs.")
                    continue
                if any(not isinstance(source_id, str) or source_id not in source_ids for source_id in ids):
                    warnings.append(
                        f"Research {field_name} finding {index} references a source absent from the result."
                    )
        if not has_findings:
            warnings.append("Research supplied no usable findings; evidence remains limited.")
        issues = result.issues
        if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
            warnings.append("Research issues collection is malformed.")
        elif issues:
            warnings.append("Research reported issues; findings may be incomplete.")
        if not source_ids:
            warnings.append("Research produced no source records; evidence remains limited.")
        return warnings

    @staticmethod
    def _research_failure_is_retryable(
        result: ResearchResult, profile: BusinessProfile
    ) -> bool:
        if result.business_profile_id != profile.business_profile_id:
            return False
        sources = result.sources
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            return False
        source_ids: set[str] = set()
        for source in sources:
            if (
                not isinstance(source, ResearchSource)
                or not isinstance(source.source_id, str)
                or not source.source_id.strip()
                or source.source_id != source.source_id.strip()
                or any(character.isspace() for character in source.source_id)
                or source.source_id in source_ids
                or not isinstance(source.url, str)
                or not ManagerAgent._valid_source_url(source.url)
                or not all(
                    isinstance(value, str) and value.strip()
                    for value in (
                        source.title,
                        source.snippet,
                        source.provider,
                        source.source_type,
                    )
                )
            ):
                return False
            source_ids.add(source.source_id)
        finding_types = {
            "business_findings": BusinessFinding,
            "market_findings": MarketResearch,
            "competitors": CompetitorResearch,
            "leads": LeadResearch,
        }
        for field_name, finding_type in finding_types.items():
            findings = getattr(result, field_name)
            if not isinstance(findings, Sequence) or isinstance(
                findings, (str, bytes)
            ):
                return False
            for finding in findings:
                if not isinstance(finding, finding_type):
                    return False
                ids = finding.source_ids
                if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                    return False
                if any(
                    not isinstance(source_id, str) or source_id not in source_ids
                    for source_id in ids
                ):
                    return False
        issues = result.issues
        return (
            isinstance(issues, Sequence)
            and not isinstance(issues, (str, bytes))
            and all(isinstance(issue, ResearchIssue) for issue in issues)
        )

    def _run_analysis(
        self, profile: BusinessProfile, research_result: ResearchResult
    ) -> tuple[AnalysisResult | None, WorkflowStage]:
        try:
            result = self._analysis_agent.analyze(research_result, profile=profile)
        except Exception as exc:
            message = f"Analysis failed: {type(exc).__name__}: {exc}"
            return None, WorkflowStage(
                _ANALYSIS_STAGE, StageStatus.FAILED, 1, message, (message,)
            )
        if not isinstance(result, AnalysisResult):
            message = "Analysis agent returned an invalid result type."
            return None, WorkflowStage(
                _ANALYSIS_STAGE, StageStatus.FAILED, 1, message, (message,)
            )
        quality_warnings = self._analysis_quality_warnings(result, research_result)
        if result.status is AnalysisStatus.FAILED:
            quality_warnings.append("Analysis returned FAILED status.")
            return result, WorkflowStage(
                _ANALYSIS_STAGE,
                StageStatus.FAILED,
                1,
                "Analysis returned FAILED status.",
                tuple(dict.fromkeys(quality_warnings)),
            )
        if not isinstance(result.status, AnalysisStatus):
            message = "Analysis result has an invalid status."
            return result, WorkflowStage(
                _ANALYSIS_STAGE, StageStatus.FAILED, 1, message, (message,)
            )
        if (
            result.status is AnalysisStatus.PARTIAL
            or result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
            or quality_warnings
        ):
            message = (
                "Analysis completed with partial evidence or quality limitations."
                if quality_warnings
                else f"Analysis status is {result.status.value}."
            )
            return result, WorkflowStage(
                _ANALYSIS_STAGE,
                StageStatus.PARTIAL,
                1,
                message,
                tuple(dict.fromkeys(quality_warnings)),
            )
        return result, WorkflowStage(
            _ANALYSIS_STAGE, StageStatus.COMPLETED, 1, "Analysis completed."
        )

    @staticmethod
    def _analysis_quality_warnings(
        result: AnalysisResult, research_result: ResearchResult
    ) -> list[str]:
        warnings: list[str] = []
        sources = research_result.sources
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            return ["Analysis cannot be validated against malformed research sources."]
        source_map: dict[str, ResearchSource] = {}
        for item in sources:
            if (
                isinstance(item, ResearchSource)
                and isinstance(item.source_id, str)
                and item.source_id.strip()
                and item.source_id == item.source_id.strip()
                and not any(character.isspace() for character in item.source_id)
                and isinstance(item.url, str)
                and ManagerAgent._valid_source_url(item.url)
            ):
                source_map[item.source_id] = item
        findings: list[object] = []
        finding_types = {
            "lead_scores": LeadScore,
            "market_signals": MarketSignal,
            "market_gaps": MarketGap,
            "opportunities": BusinessOpportunity,
            "recommendations": Recommendation,
        }
        for field_name, finding_type in finding_types.items():
            collection = getattr(result, field_name)
            if not isinstance(collection, Sequence) or isinstance(
                collection, (str, bytes)
            ):
                warnings.append(f"Analysis {field_name} collection is malformed.")
                continue
            for finding in collection:
                if not isinstance(finding, finding_type):
                    warnings.append(
                        f"Analysis {field_name} contains a malformed finding."
                    )
                    continue
                findings.append(finding)
        for finding in findings:
            source_ids = getattr(finding, "source_ids", None)
            if not isinstance(source_ids, Sequence) or isinstance(
                source_ids, (str, bytes)
            ):
                warnings.append("Analysis finding has malformed source IDs.")
                continue
            if any(
                not isinstance(source_id, str) or source_id not in source_map
                for source_id in source_ids
            ):
                warnings.append("Analysis finding references an absent research source.")
        references = result.source_references
        if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
            warnings.append("Analysis source references are malformed.")
        else:
            known_urls = {
                item.url for item in source_map.values() if isinstance(item.url, str)
            }
            if any(not isinstance(url, str) or url not in known_urls for url in references):
                warnings.append("Analysis cites a URL absent from the research sources.")
        result_warnings = result.warnings
        if not isinstance(result_warnings, Sequence) or isinstance(
            result_warnings, (str, bytes)
        ):
            warnings.append("Analysis warnings collection is malformed.")
        else:
            warnings.extend(
                f"Analysis warning: {warning}"
                for warning in result_warnings
                if isinstance(warning, str) and warning.strip()
            )
            if any(
                not isinstance(warning, str) or not warning.strip()
                for warning in result_warnings
            ):
                warnings.append("Analysis contains a malformed warning.")
        return warnings

    def _run_report(
        self,
        profile: BusinessProfile,
        research_result: ResearchResult,
        analysis_result: AnalysisResult | None,
    ) -> tuple[BusinessReport | None, WorkflowStage]:
        try:
            report = self._report_agent.generate(
                research_result,
                analysis_result,
                profile=profile,
            )
        except Exception as exc:
            message = f"Report generation failed: {type(exc).__name__}: {exc}"
            return None, WorkflowStage(
                _REPORT_STAGE, StageStatus.FAILED, 1, message, (message,)
            )
        if not isinstance(report, BusinessReport):
            message = "Report agent returned an invalid result type."
            return None, WorkflowStage(
                _REPORT_STAGE, StageStatus.FAILED, 1, message, (message,)
            )
        quality_warnings = self._report_quality_warnings(
            report, research_result, analysis_result, profile
        )
        if quality_warnings:
            return None, WorkflowStage(
                _REPORT_STAGE,
                StageStatus.FAILED,
                1,
                "Report failed source-integrity quality checks.",
                tuple(quality_warnings),
            )
        content_warnings = (
            *report.warnings,
            *report.missing_data,
        )
        all_warnings = list(dict.fromkeys(content_warnings))
        if all_warnings:
            return report, WorkflowStage(
                _REPORT_STAGE,
                StageStatus.PARTIAL,
                1,
                "Report completed with quality limitations.",
                tuple(all_warnings),
            )
        return report, WorkflowStage(
            _REPORT_STAGE, StageStatus.COMPLETED, 1, "Report completed."
        )

    @staticmethod
    def _report_quality_warnings(
        report: BusinessReport,
        research_result: ResearchResult,
        analysis_result: AnalysisResult | None,
        profile: BusinessProfile,
    ) -> list[str]:
        warnings: list[str] = []
        if not isinstance(report.executive_summary, str) or not report.executive_summary.strip():
            warnings.append("Report executive summary is missing or malformed.")
        if report.business_profile != profile:
            warnings.append("Report business context does not match the requested profile.")
        if report.business_profile_id != research_result.business_profile_id:
            warnings.append("Report business profile ID does not match research.")
        expected_analysis_status = (
            analysis_result.status if analysis_result is not None else None
        )
        if report.analysis_status is not expected_analysis_status:
            warnings.append("Report analysis status does not match the analysis result.")
        for field_name in ("warnings", "limitations", "missing_data", "next_steps"):
            values = getattr(report, field_name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                warnings.append(f"Report {field_name} collection is malformed.")
            elif any(not isinstance(value, str) or not value.strip() for value in values):
                warnings.append(f"Report {field_name} contains malformed text.")
        sources = research_result.sources
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            return ["Report sources cannot be validated against malformed research."]
        source_map = {
            item.source_id: item
            for item in sources
            if isinstance(item, ResearchSource)
            and isinstance(item.source_id, str)
            and item.source_id.strip()
            and item.source_id == item.source_id.strip()
            and not any(character.isspace() for character in item.source_id)
            and isinstance(item.url, str)
            and ManagerAgent._valid_source_url(item.url)
        }
        finding_types = {
            "business_findings": BusinessFinding,
            "market_findings": MarketResearch,
            "competitors": CompetitorResearch,
            "leads": LeadResearch,
            "lead_scores": LeadScore,
            "market_signals": MarketSignal,
            "market_gaps": MarketGap,
            "opportunities": BusinessOpportunity,
            "recommendations": Recommendation,
        }
        for field_name, finding_type in finding_types.items():
            findings = getattr(report, field_name)
            if not isinstance(findings, Sequence) or isinstance(
                findings, (str, bytes)
            ):
                warnings.append(f"Report {field_name} collection is malformed.")
                continue
            for finding in findings:
                if not isinstance(finding, finding_type):
                    warnings.append(f"Report {field_name} contains a malformed finding.")
                    continue
                source_ids = finding.source_ids
                if not isinstance(source_ids, Sequence) or isinstance(
                    source_ids, (str, bytes)
                ):
                    warnings.append(
                        f"Report {field_name} finding has malformed source IDs."
                    )
                    continue
                if any(
                    not isinstance(source_id, str) or source_id not in source_map
                    for source_id in source_ids
                ):
                    warnings.append(
                        f"Report {field_name} finding references a source absent from research."
                    )
        references = report.source_references
        if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
            return ["Report source references are malformed."]
        seen_reference_ids: set[str] = set()
        for reference in references:
            if not isinstance(reference, ReportSourceReference):
                warnings.append("Report contains a malformed source reference.")
                continue
            if not isinstance(reference.source_id, str) or reference.source_id not in source_map:
                warnings.append("Report contains a source ID absent from research.")
                continue
            if reference.source_id in seen_reference_ids:
                warnings.append("Report contains duplicate source references.")
            seen_reference_ids.add(reference.source_id)
            source = source_map[reference.source_id]
            if (
                not isinstance(reference.url, str)
                or not isinstance(reference.title, str)
                or reference.url != source.url
                or reference.title != source.title
            ):
                warnings.append("Report source details do not match the research source record.")
        if report.research_status is not research_result.status:
            warnings.append("Report research status does not match the research result.")
        return warnings

    @staticmethod
    def _valid_source_url(url: str) -> bool:
        if not isinstance(url, str) or not url.strip() or any(
            character.isspace() for character in url
        ):
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
    def _workflow_status(stages: Sequence[WorkflowStage]) -> WorkflowStatus:
        if all(stage.status is StageStatus.COMPLETED for stage in stages):
            return WorkflowStatus.SUCCESS
        if any(
            stage.status in (StageStatus.COMPLETED, StageStatus.PARTIAL)
            for stage in stages
        ):
            return WorkflowStatus.PARTIAL
        return WorkflowStatus.FAILED
