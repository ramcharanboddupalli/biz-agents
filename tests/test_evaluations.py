from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Sequence

from agents.analysis import AnalysisResult, AnalysisStatus, Recommendation
from agents.manager import (
    ManagerAgent,
    StageStatus,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
)
from agents.report import ReportAgent, ReportSourceReference
from agents.research import (
    BusinessFinding,
    MarketResearch,
    ResearchCategory,
    ResearchIssue,
    ResearchResult,
    ResearchSource,
    ResearchStatus,
)
from evals import (
    EVALUATOR_IDS,
    EvaluationResult,
    EvaluationStatus,
    evaluate_workflow,
    export_evaluations_to_phoenix,
)
from rag import BusinessProfile


class _ResearchAgent:
    def __init__(self, result: ResearchResult) -> None:
        self.result = result

    def research(self, profile: BusinessProfile) -> ResearchResult:
        _ = profile
        return self.result


class _AnalysisAgent:
    def __init__(self, result: AnalysisResult) -> None:
        self.result = result

    def analyze(
        self, research_result: ResearchResult, *, profile: BusinessProfile
    ) -> AnalysisResult:
        _ = (research_result, profile)
        return self.result


class _PhoenixSpanResource:
    def __init__(self, error: Exception | None = None) -> None:
        self.annotations: Sequence[object] | None = None
        self.error = error

    def log_span_annotations(
        self, *, span_annotations: Sequence[object]
    ) -> None:
        if self.error is not None:
            raise self.error
        self.annotations = span_annotations


class _PhoenixClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.spans = _PhoenixSpanResource(error)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = BusinessProfile(
            offering="Handmade soap",
            target_customers="natural skincare customers",
            location="Hyderabad",
            budget="confirm with finance",
            goals="grow local sales",
            business_profile_id="soap-shop",
        )
        self.source = ResearchSource(
            source_id="s1",
            title="Market source",
            url="https://example.test/market",
            snippet="A supplied source snippet.",
            provider="synthetic",
            source_type="web",
            metadata={},
        )
        self.research = ResearchResult(
            business_profile_id="soap-shop",
            status=ResearchStatus.SUCCESS,
            business_findings=(
                BusinessFinding("Business context", "Supplied context.", ("s1",)),
            ),
            market_findings=(
                MarketResearch("Market context", "Supplied market data.", ("s1",)),
            ),
            competitors=(),
            leads=(),
            sources=(self.source,),
            issues=(),
        )
        self.recommendation = Recommendation(
            action="Review the evidence.",
            why="The evidence may be relevant.",
            priority="Medium",
            source_ids=("s1",),
            next_step="Validate before acting.",
        )
        self.analysis = AnalysisResult(
            status=AnalysisStatus.SUCCESS,
            recommendations=(self.recommendation,),
            source_references=(self.source.url,),
        )
        self.workflow = ManagerAgent(
            _ResearchAgent(self.research),
            _AnalysisAgent(self.analysis),
            ReportAgent(),
        ).run(self.profile)

    def test_successful_workflow_has_stable_passing_evaluations(self) -> None:
        self.assertEqual(self.workflow.status, WorkflowStatus.SUCCESS)
        results = evaluate_workflow(self.workflow)

        self.assertEqual(tuple(result.evaluator_id for result in results), EVALUATOR_IDS)
        self.assertTrue(all(result.status is EvaluationStatus.PASS for result in results))
        self.assertTrue(all(result.score == 1.0 for result in results))
        self.assertTrue(all(result.label == "pass" for result in results))
        self.assertTrue(all(result.explanation for result in results))

    def test_partial_workflow_status_matches_partial_stage(self) -> None:
        partial_research = replace(
            self.research,
            status=ResearchStatus.PARTIAL,
            issues=(
                ResearchIssue(
                    ResearchCategory.MARKET,
                    "market research",
                    "One source query was unavailable.",
                ),
            ),
        )
        workflow = ManagerAgent(
            _ResearchAgent(partial_research),
            _AnalysisAgent(self.analysis),
            ReportAgent(),
        ).run(self.profile)

        self.assertEqual(workflow.status, WorkflowStatus.PARTIAL)
        self.assertEqual(
            evaluate_workflow(workflow)[0].status, EvaluationStatus.PASS
        )

    def test_inconsistent_workflow_status_fails(self) -> None:
        inconsistent = replace(self.workflow, status=WorkflowStatus.FAILED)

        result = evaluate_workflow(inconsistent)[0]

        self.assertEqual(result.status, EvaluationStatus.FAIL)
        self.assertEqual(result.score, 0.0)

    def test_missing_report_citations_fail_not_pass(self) -> None:
        report = replace(self.workflow.report, source_references=())
        workflow = replace(self.workflow, report=report)

        result = evaluate_workflow(workflow)[1]

        self.assertEqual(result.status, EvaluationStatus.FAIL)
        self.assertEqual(result.score, 0.0)

    def test_unlinked_report_claim_fails_citation_and_linkage_checks(self) -> None:
        report = replace(
            self.workflow.report,
            business_findings=(
                BusinessFinding("Unlinked", "No source.", ()),
            ),
        )
        workflow = replace(self.workflow, report=report)

        results = evaluate_workflow(workflow)

        self.assertEqual(results[1].status, EvaluationStatus.FAIL)
        self.assertEqual(results[2].status, EvaluationStatus.FAIL)

    def test_missing_sources_are_insufficient_not_a_pass(self) -> None:
        research = replace(self.research, sources=())
        workflow = replace(self.workflow, research_result=research)

        results = evaluate_workflow(workflow)

        self.assertEqual(results[1].status, EvaluationStatus.INSUFFICIENT_DATA)
        self.assertEqual(results[1].score, None)
        self.assertEqual(results[2].status, EvaluationStatus.INSUFFICIENT_DATA)
        self.assertEqual(results[3].status, EvaluationStatus.INSUFFICIENT_DATA)

    def test_unlinked_analysis_recommendation_fails(self) -> None:
        recommendation = replace(self.recommendation, source_ids=())
        analysis = replace(self.analysis, recommendations=(recommendation,))
        workflow = replace(self.workflow, analysis_result=analysis)

        result = evaluate_workflow(workflow)[3]

        self.assertEqual(result.status, EvaluationStatus.FAIL)
        self.assertEqual(result.score, 0.0)

    def test_invalid_input_stages_are_a_valid_failed_workflow(self) -> None:
        workflow = WorkflowResult(
            status=WorkflowStatus.INVALID_INPUT,
            stages=(
                WorkflowStage("validation", StageStatus.FAILED),
                WorkflowStage("research", StageStatus.SKIPPED),
                WorkflowStage("analysis", StageStatus.SKIPPED),
                WorkflowStage("report", StageStatus.SKIPPED),
            ),
        )

        self.assertEqual(
            evaluate_workflow(workflow)[0].status, EvaluationStatus.PASS
        )

    def test_failed_research_with_skipped_downstream_stages_is_consistent(self) -> None:
        workflow = WorkflowResult(
            status=WorkflowStatus.FAILED,
            stages=(
                WorkflowStage("validation", StageStatus.COMPLETED),
                WorkflowStage("research", StageStatus.FAILED, attempts=1),
                WorkflowStage("analysis", StageStatus.SKIPPED),
                WorkflowStage("report", StageStatus.SKIPPED),
            ),
        )

        self.assertEqual(
            evaluate_workflow(workflow)[0].status, EvaluationStatus.PASS
        )

    def test_invalid_report_source_reference_fails_citation_integrity(self) -> None:
        report = replace(
            self.workflow.report,
            source_references=(
                ReportSourceReference(
                    "unknown-source",
                    "Unverified source",
                    "https://example.test/unverified",
                ),
            ),
        )
        workflow = replace(self.workflow, report=report)

        self.assertEqual(
            evaluate_workflow(workflow)[1].status, EvaluationStatus.FAIL
        )

    def test_malformed_report_claim_collection_fails_evaluation(self) -> None:
        report = replace(self.workflow.report, recommendations="not-a-sequence")
        workflow = replace(self.workflow, report=report)

        self.assertEqual(
            evaluate_workflow(workflow)[2].status, EvaluationStatus.FAIL
        )

    def test_malformed_report_collection_fails_without_research(self) -> None:
        report = replace(self.workflow.report, source_references="not-a-sequence")
        workflow = replace(self.workflow, report=report, research_result=None)

        self.assertEqual(
            evaluate_workflow(workflow)[1].status, EvaluationStatus.FAIL
        )

    def test_malformed_recommendations_fail_without_research(self) -> None:
        analysis = replace(self.analysis, recommendations="not-a-sequence")
        workflow = replace(
            self.workflow,
            analysis_result=analysis,
            research_result=None,
        )

        self.assertEqual(
            evaluate_workflow(workflow)[3].status, EvaluationStatus.FAIL
        )

    def test_success_workflow_missing_outputs_fails_consistency(self) -> None:
        workflow = replace(self.workflow, report=None)

        self.assertEqual(
            evaluate_workflow(workflow)[0].status, EvaluationStatus.FAIL
        )

    def test_phoenix_export_sends_only_safe_annotation_fields(self) -> None:
        client = _PhoenixClient()
        results = evaluate_workflow(self.workflow)
        results = (
            replace(results[0], explanation="private business profile content"),
            *results[1:],
        )

        export_evaluations_to_phoenix(
            client,
            span_id="0123456789abcdef",
            results=results,
        )

        annotations = client.spans.annotations
        self.assertIsNotNone(annotations)
        self.assertEqual(len(annotations), len(EVALUATOR_IDS))
        for annotation, evaluator_id in zip(annotations, EVALUATOR_IDS):
            self.assertEqual(annotation["span_id"], "0123456789abcdef")
            self.assertEqual(annotation["name"], evaluator_id)
            self.assertEqual(annotation["annotator_kind"], "CODE")
            self.assertIn(annotation["label"], {"pass", "fail", "insufficient_data"})
            self.assertNotIn("explanation", annotation)
            if annotation["label"] == "insufficient_data":
                self.assertNotIn("score", annotation)
            else:
                self.assertIn(annotation["score"], {0.0, 1.0})

        self.assertNotIn(
            "private business profile content",
            repr(annotations),
        )

    def test_phoenix_export_failure_is_explicit_and_does_not_mutate_workflow(self) -> None:
        client = _PhoenixClient(RuntimeError("export unavailable"))
        original_workflow = self.workflow
        results = evaluate_workflow(original_workflow)

        with self.assertRaisesRegex(RuntimeError, "export unavailable"):
            export_evaluations_to_phoenix(
                client,
                span_id="0123456789abcdef",
                results=results,
            )

        self.assertIs(self.workflow, original_workflow)
        self.assertEqual(self.workflow.status, WorkflowStatus.SUCCESS)

    def test_export_rejects_unknown_evaluator_ids(self) -> None:
        client = _PhoenixClient()
        invalid = EvaluationResult(
            "untrusted.identifier",
            EvaluationStatus.PASS,
            1.0,
            "safe explanation",
        )

        with self.assertRaisesRegex(ValueError, "unrecognized evaluator_id"):
            export_evaluations_to_phoenix(
                client,
                span_id="0123456789abcdef",
                results=(invalid,),
            )
        self.assertIsNone(client.spans.annotations)

    def test_export_rejects_non_hexadecimal_span_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "16-character hexadecimal"):
            export_evaluations_to_phoenix(
                _PhoenixClient(),
                span_id="not-a-span",
                results=evaluate_workflow(self.workflow),
            )


if __name__ == "__main__":
    unittest.main()
