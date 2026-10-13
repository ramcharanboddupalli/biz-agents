"""Offline workflow evaluators and opt-in Phoenix span annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import re
from typing import Protocol

from agents.analysis import AnalysisResult, AnalysisStatus, Recommendation
from agents.manager import (
    StageStatus,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
)
from agents.report import BusinessReport, ReportAgent, ReportSourceReference
from agents.research import ResearchResult, ResearchSource, ResearchStatus


class EvaluationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """A stable, Phoenix-compatible evaluation outcome without business text."""

    evaluator_id: str
    status: EvaluationStatus
    score: float | None
    explanation: str

    @property
    def label(self) -> str:
        return self.status.value


EVALUATOR_IDS = (
    "bizagents.workflow.status_consistency",
    "bizagents.report.citation_integrity",
    "bizagents.report.claim_source_linkage",
    "bizagents.analysis.recommendation_source_linkage",
)

_STAGE_ORDER = ("validation", "research", "analysis", "report")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")
_CLAIM_COLLECTIONS = (
    "business_findings",
    "market_findings",
    "competitors",
    "leads",
    "lead_scores",
    "market_signals",
    "market_gaps",
    "opportunities",
    "recommendations",
)


class _PhoenixSpanResource(Protocol):
    def log_span_annotations(
        self, *, span_annotations: Sequence[Mapping[str, str | float]]
    ) -> object:
        ...


class _PhoenixClient(Protocol):
    spans: _PhoenixSpanResource


def evaluate_workflow(workflow: WorkflowResult) -> tuple[EvaluationResult, ...]:
    """Evaluate a Manager result without network calls or retaining report text."""
    if not isinstance(workflow, WorkflowResult):
        raise TypeError("workflow must be a WorkflowResult.")

    return (
        _evaluate_workflow_status(workflow),
        _evaluate_report_citations(workflow.report, workflow.research_result),
        _evaluate_claim_sources(workflow.report, workflow.research_result),
        _evaluate_recommendation_sources(
            workflow.analysis_result, workflow.research_result
        ),
    )


def export_evaluations_to_phoenix(
    client: _PhoenixClient,
    *,
    span_id: str,
    results: Sequence[EvaluationResult],
) -> None:
    """Explicitly export safe CODE annotations to an existing Phoenix span.

    The caller supplies an already-created Phoenix client and workflow span ID.
    Phoenix client failures are raised to the caller; this function is never
    invoked by ManagerAgent, so export cannot change its returned workflow.
    """
    if not isinstance(span_id, str) or not _SPAN_ID_PATTERN.fullmatch(span_id):
        raise ValueError("span_id must be a 16-character hexadecimal span ID.")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("results must be a sequence of EvaluationResult values.")
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")

    annotations: list[Mapping[str, str | float]] = []
    for result in results:
        if not isinstance(result, EvaluationResult):
            raise TypeError("results must contain only EvaluationResult values.")
        if result.evaluator_id not in EVALUATOR_IDS:
            raise ValueError("result has an unrecognized evaluator_id.")
        if not isinstance(result.status, EvaluationStatus):
            raise ValueError("result has an invalid evaluation status.")
        expected_score = {
            EvaluationStatus.PASS: 1.0,
            EvaluationStatus.FAIL: 0.0,
            EvaluationStatus.INSUFFICIENT_DATA: None,
        }[result.status]
        if (
            result.score != expected_score
            or isinstance(result.score, bool)
            or (
                result.score is not None
                and not isinstance(result.score, (int, float))
            )
        ):
            raise ValueError("result score does not match its evaluation status.")

        annotation: dict[str, str | float] = {
            "span_id": span_id,
            "name": result.evaluator_id,
            "annotator_kind": "CODE",
            "label": result.label,
        }
        if result.score is not None:
            annotation["score"] = result.score
        annotations.append(annotation)

    client.spans.log_span_annotations(span_annotations=annotations)


def _evaluate_workflow_status(workflow: WorkflowResult) -> EvaluationResult:
    stages = workflow.stages
    if (
        not isinstance(stages, Sequence)
        or isinstance(stages, (str, bytes))
        or len(stages) != len(_STAGE_ORDER)
        or any(not isinstance(stage, WorkflowStage) for stage in stages)
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow stages are missing or malformed.",
        )

    if tuple(stage.name for stage in stages) != _STAGE_ORDER:
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow stages do not follow the required order.",
        )
    if any(
        not isinstance(stage.status, StageStatus)
        or not isinstance(stage.attempts, int)
        or isinstance(stage.attempts, bool)
        or stage.attempts < 0
        for stage in stages
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow stage statuses or attempt counts are invalid.",
        )
    if not isinstance(workflow.status, WorkflowStatus):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow status is invalid.",
        )

    statuses = tuple(stage.status for stage in stages)
    research_stage = stages[1]
    analysis_stage = stages[2]
    if any(
        stage.status is StageStatus.SKIPPED and stage.attempts != 0
        for stage in stages
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Skipped workflow stages must have zero attempts.",
        )
    if any(
        stage.attempts < 1
        for stage in stages[1:]
        if stage.status is not StageStatus.SKIPPED
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Executed workflow stages must record at least one attempt.",
        )
    if (
        workflow.research_result is not None
        and not isinstance(workflow.research_result, ResearchResult)
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow research output is malformed.",
        )
    if (
        workflow.analysis_result is not None
        and not isinstance(workflow.analysis_result, AnalysisResult)
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow analysis output is malformed.",
        )
    if workflow.report is not None and not isinstance(workflow.report, BusinessReport):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow report output is malformed.",
        )
    if (
        workflow.research_result is not None
        and not isinstance(workflow.research_result.status, ResearchStatus)
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow research output has an invalid status.",
        )
    if (
        workflow.analysis_result is not None
        and not isinstance(workflow.analysis_result.status, AnalysisStatus)
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow analysis output has an invalid status.",
        )
    if statuses[1] is StageStatus.SKIPPED and workflow.research_result is not None:
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Research output is present although its stage was skipped.",
        )
    if statuses[2] is StageStatus.SKIPPED and workflow.analysis_result is not None:
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Analysis output is present although its stage was skipped.",
        )
    if statuses[3] is StageStatus.SKIPPED and workflow.report is not None:
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Report output is present although its stage was skipped.",
        )
    if (
        statuses[3] in (StageStatus.COMPLETED, StageStatus.PARTIAL)
        and workflow.report is None
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Report stage has no report output.",
        )
    if (
        workflow.research_result is not None
        and workflow.research_result.status is ResearchStatus.FAILED
        and research_stage.status is not StageStatus.FAILED
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Failed research output does not match its stage status.",
        )
    if (
        workflow.research_result is not None
        and workflow.research_result.status is ResearchStatus.PARTIAL
        and research_stage.status is not StageStatus.PARTIAL
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Partial research output does not match its stage status.",
        )
    if (
        workflow.analysis_result is not None
        and workflow.analysis_result.status is AnalysisStatus.FAILED
        and analysis_stage.status is not StageStatus.FAILED
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Failed analysis output does not match its stage status.",
        )
    if (
        workflow.analysis_result is not None
        and workflow.analysis_result.status
        in (AnalysisStatus.PARTIAL, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        and analysis_stage.status is not StageStatus.PARTIAL
    ):
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Partial analysis output does not match its stage status.",
        )

    consistent = False
    if workflow.status is WorkflowStatus.SUCCESS:
        consistent = (
            all(status is StageStatus.COMPLETED for status in statuses)
            and workflow.research_result is not None
            and workflow.analysis_result is not None
            and workflow.report is not None
        )
    elif workflow.status is WorkflowStatus.PARTIAL:
        consistent = (
            statuses[0] is StageStatus.COMPLETED
            and statuses[1] is not StageStatus.SKIPPED
            and statuses[2] is not StageStatus.SKIPPED
            and statuses[3] is not StageStatus.SKIPPED
            and workflow.research_result is not None
            and not all(
                status is StageStatus.COMPLETED for status in statuses
            )
        )
    elif workflow.status is WorkflowStatus.FAILED:
        consistent = (
            statuses
            == (
                StageStatus.COMPLETED,
                StageStatus.FAILED,
                StageStatus.SKIPPED,
                StageStatus.SKIPPED,
            )
            and workflow.research_result is None
            and workflow.analysis_result is None
            and workflow.report is None
        )
    elif workflow.status is WorkflowStatus.INVALID_INPUT:
        consistent = (
            statuses
            == (
                StageStatus.FAILED,
                StageStatus.SKIPPED,
                StageStatus.SKIPPED,
                StageStatus.SKIPPED,
            )
            and workflow.research_result is None
            and workflow.analysis_result is None
            and workflow.report is None
        )

    if not consistent:
        return _result(
            EVALUATOR_IDS[0],
            EvaluationStatus.FAIL,
            "Workflow status does not match its stage outcomes.",
        )
    return _result(
        EVALUATOR_IDS[0],
        EvaluationStatus.PASS,
        "Workflow status matches its stage outcomes.",
    )


def _evaluate_report_citations(
    report: BusinessReport | None,
    research: ResearchResult | None,
) -> EvaluationResult:
    evaluator_id = EVALUATOR_IDS[1]
    if report is None:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "A report is required.",
        )
    if not isinstance(report, BusinessReport):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Report input is malformed.",
        )

    claims = _report_claims(report)
    references = report.source_references
    if claims is None:
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "One or more report claim collections are malformed.",
        )
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Report source references are malformed.",
        )
    if any(not isinstance(reference, ReportSourceReference) for reference in references):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "One or more report source references are malformed.",
        )
    if any(
        not isinstance(reference.source_id, str)
        or not isinstance(reference.title, str)
        or not isinstance(reference.url, str)
        for reference in references
    ):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "One or more report source references are malformed.",
        )
    if not references:
        if claims:
            return _result(
                evaluator_id,
                EvaluationStatus.FAIL,
                "Report claims are missing source citations.",
            )
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "The report contains no claims or citations to validate.",
        )
    if research is None:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "Research sources are required to validate report citations.",
        )
    if not isinstance(research, ResearchResult):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Research input is malformed.",
        )
    source_map = _validated_source_map(research)
    if source_map is None or not source_map:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "No usable research sources are available to validate citations.",
        )

    cited_ids: set[str] = set()
    for reference in references:
        source = source_map.get(reference.source_id)
        if (
            source is None
            or reference.title != source.title
            or reference.url != source.url
            or reference.source_id in cited_ids
        ):
            return _result(
                evaluator_id,
                EvaluationStatus.FAIL,
                "One or more report citations do not match supplied research sources.",
            )
        cited_ids.add(reference.source_id)

    required_ids: set[str] = set()
    for claim in claims:
        source_ids = getattr(claim, "source_ids", None)
        if not _is_sequence(source_ids) or not source_ids:
            return _result(
                evaluator_id,
                EvaluationStatus.FAIL,
                "One or more report claims are missing source citations.",
            )
        if any(not isinstance(source_id, str) or source_id not in source_map for source_id in source_ids):
            return _result(
                evaluator_id,
                EvaluationStatus.FAIL,
                "One or more report claims cite unavailable research sources.",
            )
        required_ids.update(source_ids)
    if not required_ids.issubset(cited_ids):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "One or more report claims are missing source citations.",
        )
    return _result(
        evaluator_id,
        EvaluationStatus.PASS,
        f"All {len(cited_ids)} report citation(s) match supplied research sources.",
    )


def _evaluate_claim_sources(
    report: BusinessReport | None,
    research: ResearchResult | None,
) -> EvaluationResult:
    evaluator_id = EVALUATOR_IDS[2]
    if report is None:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "A report is required.",
        )
    if not isinstance(report, BusinessReport):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Report input is malformed.",
        )
    claims = _report_claims(report)
    if claims is None:
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "One or more report claim collections are malformed.",
        )
    if not claims:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "The report contains no structured claims to check.",
        )
    if research is None:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "Research sources are required to check report claim links.",
        )
    if not isinstance(research, ResearchResult):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Research input is malformed.",
        )
    source_map = _validated_source_map(research)
    if not source_map:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "No usable research sources are available to check claim links.",
        )

    unlinked = 0
    for claim in claims:
        source_ids = getattr(claim, "source_ids", None)
        if (
            not _is_sequence(source_ids)
            or not source_ids
            or any(
                not isinstance(source_id, str) or source_id not in source_map
                for source_id in source_ids
            )
        ):
            unlinked += 1
    if unlinked:
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            f"{unlinked} of {len(claims)} report claim(s) lack valid research-source links.",
        )
    return _result(
        evaluator_id,
        EvaluationStatus.PASS,
        f"All {len(claims)} report claim(s) link to supplied research sources.",
    )


def _evaluate_recommendation_sources(
    analysis: AnalysisResult | None,
    research: ResearchResult | None,
) -> EvaluationResult:
    evaluator_id = EVALUATOR_IDS[3]
    if analysis is None:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "Analysis is required.",
        )
    if not isinstance(analysis, AnalysisResult):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Analysis input is malformed.",
        )
    recommendations = analysis.recommendations
    if not _is_sequence(recommendations):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Analysis recommendations are malformed.",
        )
    if any(not isinstance(item, Recommendation) for item in recommendations):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "One or more analysis recommendations are malformed.",
        )
    if not recommendations:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "No analysis recommendations are available to check.",
        )
    if research is None:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "Research sources are required to check recommendations.",
        )
    if not isinstance(research, ResearchResult):
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            "Research input is malformed.",
        )
    source_map = _validated_source_map(research)
    if not source_map:
        return _result(
            evaluator_id,
            EvaluationStatus.INSUFFICIENT_DATA,
            "No usable research sources are available to check recommendations.",
        )

    unlinked = 0
    for recommendation in recommendations:
        source_ids = recommendation.source_ids
        if (
            not _is_sequence(source_ids)
            or not source_ids
            or any(
                not isinstance(source_id, str) or source_id not in source_map
                for source_id in source_ids
            )
        ):
            unlinked += 1
    if unlinked:
        return _result(
            evaluator_id,
            EvaluationStatus.FAIL,
            f"{unlinked} of {len(recommendations)} recommendation(s) lack valid research-source links.",
        )
    return _result(
        evaluator_id,
        EvaluationStatus.PASS,
        f"All {len(recommendations)} recommendation(s) link to supplied research sources.",
    )


def _report_claims(report: BusinessReport) -> list[object] | None:
    claims: list[object] = []
    for collection_name in _CLAIM_COLLECTIONS:
        values = getattr(report, collection_name, ())
        if not _is_sequence(values):
            return None
        claims.extend(values)
    return claims


def _validated_source_map(
    research: ResearchResult,
) -> dict[str, ResearchSource] | None:
    sources = getattr(research, "sources", None)
    if not _is_sequence(sources):
        return None

    return ReportAgent._validated_sources(sources, [])


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _result(
    evaluator_id: str,
    status: EvaluationStatus,
    explanation: str,
) -> EvaluationResult:
    score = {
        EvaluationStatus.PASS: 1.0,
        EvaluationStatus.FAIL: 0.0,
        EvaluationStatus.INSUFFICIENT_DATA: None,
    }[status]
    return EvaluationResult(evaluator_id, status, score, explanation)
