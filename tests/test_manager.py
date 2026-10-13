import unittest
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import cast
from unittest.mock import patch

from agents import (
    AnalysisResult,
    AnalysisStatus,
    BusinessFinding,
    BusinessOpportunity,
    BusinessReport,
    LeadScore,
    ManagerAgent,
    MarketEvidenceStatus,
    MarketResearch,
    MarketSignal,
    Recommendation,
    ReportAgent,
    ReportSourceReference,
    ResearchIssue,
    ResearchCategory,
    ResearchResult,
    ResearchSource,
    ResearchStatus,
    StageStatus,
    WorkflowResult,
    WorkflowStatus,
)
from agents.tracing import configured_tracer, traced_span
from rag import BusinessProfile


class FakeSpan:
    def __init__(self, name: str, fail_on_attribute: bool = False) -> None:
        self.name = name
        self.attributes: dict[str, bool | float | int | str] = {}
        self.fail_on_attribute = fail_on_attribute
        self.finished = False

    def set_attribute(self, key: str, value: bool | float | int | str) -> None:
        if self.fail_on_attribute:
            raise OSError("local fake exporter failure")
        self.attributes[key] = value


class FakeTracer:
    def __init__(
        self, fail_on_finish: bool = False, fail_on_attribute: bool = False
    ) -> None:
        self.spans: list[FakeSpan] = []
        self.fail_on_finish = fail_on_finish
        self.fail_on_attribute = fail_on_attribute
        self.start_options: list[dict[str, bool]] = []

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        **kwargs: bool,
    ) -> Generator[FakeSpan, None, None]:
        self.start_options.append(kwargs)
        span = FakeSpan(name, fail_on_attribute=self.fail_on_attribute)
        self.spans.append(span)
        try:
            yield span
        finally:
            span.finished = True
            if self.fail_on_finish:
                raise OSError("local fake exporter failure")


class FakeResearchAgent:
    def __init__(
        self,
        *outcomes: ResearchResult | Exception,
        events: list[str] | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.events = events
        self.calls: list[BusinessProfile] = []

    def research(self, profile: BusinessProfile) -> ResearchResult:
        if self.events is not None:
            self.events.append("research")
        self.calls.append(profile)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAnalysisAgent:
    def __init__(
        self, result: AnalysisResult | Exception, events: list[str] | None = None
    ) -> None:
        self.result = result
        self.events = events
        self.calls: list[tuple[ResearchResult, BusinessProfile]] = []

    def analyze(
        self, research_result: ResearchResult, *, profile: BusinessProfile
    ) -> AnalysisResult:
        if self.events is not None:
            self.events.append("analysis")
        self.calls.append((research_result, profile))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeReportAgent:
    def __init__(
        self,
        result: BusinessReport | Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.events = events
        self.calls: list[
            tuple[ResearchResult | None, AnalysisResult | None, BusinessProfile | None]
        ] = []

    def generate(
        self,
        research_result: ResearchResult | None = None,
        analysis_result: AnalysisResult | None = None,
        *,
        profile: BusinessProfile | None = None,
    ) -> BusinessReport:
        if self.events is not None:
            self.events.append("report")
        self.calls.append((research_result, analysis_result, profile))
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is not None:
            return self.result
        return ReportAgent().generate(
            research_result, analysis_result, profile=profile
        )


class ManagerAgentTests(unittest.TestCase):
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
            title="Local skincare report",
            url="https://example.test/market",
            snippet="A source-reported skincare finding.",
            provider="synthetic",
            source_type="web",
            metadata={},
        )
        self.business_finding = BusinessFinding(
            "Business context", "Source-reported context.", ("s1",)
        )
        self.market_finding = MarketResearch(
            "Market context", "Source-reported market context.", ("s1",)
        )
        self.research = ResearchResult(
            business_profile_id="soap-shop",
            status=ResearchStatus.SUCCESS,
            business_findings=(self.business_finding,),
            market_findings=(self.market_finding,),
            competitors=(),
            leads=(),
            sources=(self.source,),
            issues=(),
        )
        self.lead_score = LeadScore(
            "Natural Living Store",
            60,
            1,
            "Medium",
            "Evidence alignment only.",
            ("s1",),
            ("A source-reported fit.",),
            "Not a conversion probability.",
        )
        self.signal = MarketSignal(
            "A supplied market signal.",
            ("A source-reported signal.",),
            ("s1",),
            "Not independently verified.",
        )
        self.opportunity = BusinessOpportunity(
            "Validate possible retailer fit.",
            "Related to the supplied target customers.",
            "Unvalidated potential.",
            ("A source-reported fit.",),
            ("s1",),
            "Requires validation.",
        )
        self.recommendation = Recommendation(
            "Review the supplied evidence.",
            "It relates to the business profile.",
            "Medium",
            ("s1",),
            "Verify costs and budget before any commitment.",
            "Advisory only.",
        )
        self.analysis = AnalysisResult(
            status=AnalysisStatus.SUCCESS,
            lead_scores=(self.lead_score,),
            market_signals=(self.signal,),
            market_status=MarketEvidenceStatus.RESEARCH_SIGNAL,
            opportunities=(self.opportunity,),
            recommendations=(self.recommendation,),
            source_references=(self.source.url,),
        )

    def test_successful_workflow_uses_existing_order_and_handoffs(self) -> None:
        events: list[str] = []
        research_agent = FakeResearchAgent(self.research, events=events)
        analysis_agent = FakeAnalysisAgent(self.analysis, events)
        report_agent = FakeReportAgent(events=events)
        manager = ManagerAgent(research_agent, analysis_agent, report_agent)

        result = manager.run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.SUCCESS)
        self.assertEqual(events, ["research", "analysis", "report"])
        self.assertEqual(
            [stage.name for stage in result.stages],
            ["validation", "research", "analysis", "report"],
        )
        self.assertTrue(all(stage.status is StageStatus.COMPLETED for stage in result.stages))
        self.assertIs(analysis_agent.calls[0][0], self.research)
        self.assertIs(analysis_agent.calls[0][1], self.profile)
        self.assertEqual(
            report_agent.calls[0],
            (self.research, self.analysis, self.profile),
        )
        self.assertEqual(result.report.lead_scores, (self.lead_score,))
        self.assertEqual(result.report.opportunities, (self.opportunity,))
        self.assertEqual(result.report.recommendations, (self.recommendation,))
        self.assertEqual(result.report.source_references[0].source_id, "s1")

    def test_invalid_or_missing_profile_skips_every_agent(self) -> None:
        research_agent = FakeResearchAgent(self.research)
        manager = ManagerAgent(research_agent)

        result = manager.run(cast(BusinessProfile, None))

        self.assertEqual(result.status, WorkflowStatus.INVALID_INPUT)
        self.assertEqual(result.stages[0].status, StageStatus.FAILED)
        self.assertTrue(
            all(stage.status is StageStatus.SKIPPED for stage in result.stages[1:])
        )
        self.assertEqual(research_agent.calls, [])
        self.assertTrue(any("BusinessProfile" in warning for warning in result.warnings))

    def test_corrupted_profile_fields_are_rejected_before_research(self) -> None:
        object.__setattr__(self.profile, "budget", None)
        research_agent = FakeResearchAgent(self.research)

        result = ManagerAgent(research_agent).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.INVALID_INPUT)
        self.assertEqual(research_agent.calls, [])
        self.assertIn("profile.budget", result.warnings[0])

    def test_partial_research_result_is_preserved_and_reported(self) -> None:
        partial = replace(
            self.research,
            status=ResearchStatus.PARTIAL,
            issues=(
                ResearchIssue(
                    category=ResearchCategory.MARKET,
                    query="soap market",
                    message="One query was unavailable.",
                ),
            ),
        )
        result = ManagerAgent(
            FakeResearchAgent(partial), FakeAnalysisAgent(self.analysis)
        ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.PARTIAL)
        self.assertEqual(result.research_result, partial)
        self.assertEqual(self._stage(result, "research").status, StageStatus.PARTIAL)
        self.assertIsNotNone(result.report)
        self.assertTrue(any("PARTIAL" in warning or "partial" in warning.lower() for warning in result.warnings))
        self.assertEqual(result.report.research_status, ResearchStatus.PARTIAL)

    def test_failed_research_is_retried_with_a_fixed_limit_and_partial_data_kept(self) -> None:
        failed = replace(self.research, status=ResearchStatus.FAILED)
        research_agent = FakeResearchAgent(failed, self.research)
        result = ManagerAgent(
            research_agent,
            FakeAnalysisAgent(self.analysis),
            research_retry_limit=1,
        ).run(self.profile)

        self.assertEqual(len(research_agent.calls), 2)
        self.assertEqual(self._stage(result, "research").attempts, 2)
        self.assertEqual(result.research_result, self.research)
        self.assertEqual(result.status, WorkflowStatus.SUCCESS)

    def test_retry_limit_bounds_persistent_failed_research(self) -> None:
        failed = replace(self.research, status=ResearchStatus.FAILED)
        research_agent = FakeResearchAgent(failed)

        result = ManagerAgent(
            research_agent,
            FakeAnalysisAgent(self.analysis),
            research_retry_limit=2,
        ).run(self.profile)

        self.assertEqual(len(research_agent.calls), 3)
        self.assertEqual(self._stage(result, "research").attempts, 3)
        self.assertEqual(self._stage(result, "research").status, StageStatus.FAILED)
        self.assertIsNotNone(result.analysis_result)
        self.assertIsNotNone(result.report)
        self.assertEqual(result.status, WorkflowStatus.PARTIAL)

    def test_structurally_invalid_failed_research_is_not_retried(self) -> None:
        failed = replace(
            self.research,
            business_profile_id="different-business",
            status=ResearchStatus.FAILED,
        )
        research_agent = FakeResearchAgent(failed, self.research)

        result = ManagerAgent(
            research_agent, FakeAnalysisAgent(self.analysis), research_retry_limit=2
        ).run(self.profile)

        self.assertEqual(len(research_agent.calls), 1)
        self.assertEqual(self._stage(result, "research").status, StageStatus.FAILED)
        self.assertTrue(
            any("does not match" in warning for warning in result.warnings)
        )

    def test_transient_research_exception_is_retried_once(self) -> None:
        research_agent = FakeResearchAgent(OSError("temporary"), self.research)
        result = ManagerAgent(
            research_agent, FakeAnalysisAgent(self.analysis)
        ).run(self.profile)

        self.assertEqual(len(research_agent.calls), 2)
        self.assertEqual(self._stage(result, "research").attempts, 2)
        self.assertEqual(self._stage(result, "research").status, StageStatus.COMPLETED)
        self.assertEqual(result.status, WorkflowStatus.SUCCESS)

    def test_validation_exception_is_not_retried(self) -> None:
        research_agent = FakeResearchAgent(ValueError("invalid input"), self.research)
        result = ManagerAgent(research_agent).run(self.profile)

        self.assertEqual(len(research_agent.calls), 1)
        self.assertEqual(self._stage(result, "research").status, StageStatus.FAILED)
        self.assertEqual(self._stage(result, "research").attempts, 1)
        self.assertEqual(self._stage(result, "analysis").status, StageStatus.SKIPPED)
        self.assertEqual(self._stage(result, "report").status, StageStatus.SKIPPED)

    def test_research_failure_exception_is_bounded_and_skips_dependents(self) -> None:
        research_agent = FakeResearchAgent(OSError("offline"))
        result = ManagerAgent(research_agent, research_retry_limit=1).run(self.profile)

        self.assertEqual(len(research_agent.calls), 2)
        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertEqual(self._stage(result, "research").status, StageStatus.FAILED)
        self.assertEqual(
            [self._stage(result, name).status for name in ("analysis", "report")],
            [StageStatus.SKIPPED, StageStatus.SKIPPED],
        )
        self.assertIsNone(result.report)

    def test_empty_research_is_not_promoted_to_success_or_invented_findings(self) -> None:
        empty = replace(
            self.research,
            business_findings=(),
            market_findings=(),
            sources=(),
            leads=(),
        )
        analysis = AnalysisResult(status=AnalysisStatus.INSUFFICIENT_EVIDENCE)
        result = ManagerAgent(
            FakeResearchAgent(empty), FakeAnalysisAgent(analysis)
        ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.PARTIAL)
        self.assertEqual(result.report.business_findings, ())
        self.assertEqual(result.report.market_findings, ())
        self.assertEqual(result.report.opportunities, ())
        self.assertEqual(result.report.recommendations, ())
        self.assertTrue(any("no source" in warning.lower() for warning in result.warnings))

    def test_analysis_failure_is_visible_and_report_keeps_research(self) -> None:
        analysis_agent = FakeAnalysisAgent(RuntimeError("analysis unavailable"))
        report_agent = FakeReportAgent()
        result = ManagerAgent(
            FakeResearchAgent(self.research), analysis_agent, report_agent
        ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.PARTIAL)
        self.assertEqual(self._stage(result, "analysis").status, StageStatus.FAILED)
        self.assertIsNone(result.analysis_result)
        self.assertEqual(report_agent.calls[0][0], self.research)
        self.assertIsNone(report_agent.calls[0][1])
        self.assertIsNotNone(result.report)
        self.assertTrue(any("analysis unavailable" in warning for warning in result.warnings))

    def test_analysis_failed_status_is_not_treated_as_success(self) -> None:
        failed_analysis = AnalysisResult(status=AnalysisStatus.FAILED)
        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(failed_analysis),
        ).run(self.profile)

        self.assertEqual(self._stage(result, "analysis").status, StageStatus.FAILED)
        self.assertEqual(result.analysis_result, failed_analysis)
        self.assertEqual(result.report.analysis_status, AnalysisStatus.FAILED)
        self.assertEqual(result.status, WorkflowStatus.PARTIAL)

    def test_report_failure_is_visible_without_discarding_prior_results(self) -> None:
        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(self.analysis),
            FakeReportAgent(RuntimeError("render unavailable")),
        ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.PARTIAL)
        self.assertEqual(self._stage(result, "report").status, StageStatus.FAILED)
        self.assertIsNotNone(result.research_result)
        self.assertIsNotNone(result.analysis_result)
        self.assertIsNone(result.report)
        self.assertTrue(any("render unavailable" in warning for warning in result.warnings))

    def test_invalid_research_source_references_fail_quality_check_and_are_not_reported(self) -> None:
        invalid = replace(
            self.research,
            leads=(),
            business_findings=(
                BusinessFinding("Unlinked", "Unsupported.", ("invented-source",)),
            ),
        )
        result = ManagerAgent(
            FakeResearchAgent(invalid), FakeAnalysisAgent(self.analysis)
        ).run(self.profile)

        self.assertEqual(self._stage(result, "research").status, StageStatus.PARTIAL)
        self.assertTrue(any("absent from the result" in warning for warning in result.warnings))
        self.assertEqual(result.report.business_findings[0].source_ids, ())
        self.assertNotIn(
            "invented-source",
            {reference.source_id for reference in result.report.source_references},
        )

    def test_invalid_analysis_citation_fails_quality_check_before_report_handoff(self) -> None:
        invalid_analysis = replace(
            self.analysis,
            source_references=("https://invented.example.test/evidence",),
        )
        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(invalid_analysis),
        ).run(self.profile)

        self.assertEqual(self._stage(result, "analysis").status, StageStatus.PARTIAL)
        self.assertTrue(any("absent from the research sources" in warning for warning in result.warnings))
        self.assertEqual(result.report.source_references[0].url, self.source.url)
        self.assertNotIn("invented.example.test", result.report.to_markdown())

    def test_report_quality_check_flags_invented_source_reference(self) -> None:
        bad_report = replace(
            ReportAgent().generate(self.research, self.analysis, profile=self.profile),
            source_references=(),
        )
        bad_report = replace(
            bad_report,
            source_references=(
                ReportSourceReference(
                    "invented", "Made-up", "https://invented.example.test/"
                ),
            ),
        )
        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(self.analysis),
            FakeReportAgent(bad_report),
        ).run(self.profile)

        self.assertEqual(self._stage(result, "report").status, StageStatus.FAILED)
        self.assertIsNone(result.report)
        self.assertTrue(any("absent from research" in warning for warning in result.warnings))

    def test_report_quality_check_flags_unlinked_finding_even_with_valid_references(self) -> None:
        valid_report = ReportAgent().generate(
            self.research, self.analysis, profile=self.profile
        )
        bad_report = replace(
            valid_report,
            business_findings=(
                BusinessFinding("Unsupported", "Not linked.", ("invented",)),
            ),
        )

        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(self.analysis),
            FakeReportAgent(bad_report),
        ).run(self.profile)

        self.assertEqual(self._stage(result, "report").status, StageStatus.FAILED)
        self.assertIsNone(result.report)
        self.assertTrue(
            any("finding references a source absent" in warning for warning in result.warnings)
        )

    def test_output_is_deterministic(self) -> None:
        def run_once():
            return ManagerAgent(
                FakeResearchAgent(self.research),
                FakeAnalysisAgent(self.analysis),
            ).run(self.profile)

        first = run_once()
        second = run_once()

        self.assertEqual(first.status, second.status)
        self.assertEqual(first.stages, second.stages)
        self.assertEqual(first.report.to_markdown(), second.report.to_markdown())

    def test_invalid_retry_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ManagerAgent(FakeResearchAgent(self.research), research_retry_limit=-1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ManagerAgent(FakeResearchAgent(self.research), research_retry_limit=True)

    def test_tracing_disabled_without_credentials_preserves_workflow(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("phoenix.otel.register") as register:
                result = ManagerAgent(
                    FakeResearchAgent(self.research),
                    FakeAnalysisAgent(self.analysis),
                ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.SUCCESS)
        register.assert_not_called()

    def test_invalid_tracing_configuration_does_not_initialize_phoenix(self) -> None:
        invalid_configuration = (
            {"BIZAGENTS_TRACING_ENABLED": "sometimes"},
            {"BIZAGENTS_TRACING_ENABLED": "true"},
            {
                "BIZAGENTS_TRACING_ENABLED": "true",
                "PHOENIX_COLLECTOR_ENDPOINT": "https://user@collector.invalid",
            },
        )
        for environment in invalid_configuration:
            with self.subTest(environment=environment):
                with patch.dict("os.environ", environment, clear=True):
                    with patch("phoenix.otel.register") as register:
                        self.assertIsNone(configured_tracer())
                register.assert_not_called()

    def test_phoenix_initialization_failure_does_not_change_workflow(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BIZAGENTS_TRACING_ENABLED": "true",
                "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:6006/v1/traces",
            },
            clear=True,
        ):
            with patch("phoenix.otel.register", side_effect=OSError("secret endpoint")):
                result = ManagerAgent(
                    FakeResearchAgent(self.research),
                    FakeAnalysisAgent(self.analysis),
                ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.SUCCESS)
        self.assertNotIn("secret endpoint", repr(result))

    def test_attribute_export_failure_does_not_change_workflow(self) -> None:
        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(self.analysis),
            tracer=FakeTracer(fail_on_attribute=True),
        ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.SUCCESS)

    def test_phoenix_registration_uses_environment_and_disables_autoinstrumentation(
        self,
    ) -> None:
        tracer = FakeTracer()
        environment = {
            "BIZAGENTS_TRACING_ENABLED": "true",
            "PHOENIX_PROJECT_NAME": "local-project",
            "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:6006/v1/traces",
        }
        with patch.dict("os.environ", environment, clear=True):
            with patch("phoenix.otel.register") as register:
                register.return_value.get_tracer.return_value = tracer
                configured = configured_tracer()

        self.assertIs(configured, tracer)
        register.assert_called_once_with(
            endpoint="http://localhost:6006/v1/traces",
            project_name="local-project",
            api_key=None,
            batch=True,
            auto_instrument=False,
            verbose=False,
        )

    def test_successful_workflow_records_safe_workflow_and_stage_metadata(self) -> None:
        tracer = FakeTracer()
        result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(self.analysis),
            tracer=tracer,
        ).run(self.profile)

        self.assertEqual(result.status, WorkflowStatus.SUCCESS)
        spans_by_name = {span.name: span for span in tracer.spans}
        self.assertEqual(
            set(spans_by_name),
            {
                "bizagents.workflow",
                "bizagents.stage.validation",
                "bizagents.stage.research",
                "bizagents.stage.analysis",
                "bizagents.stage.report",
            },
        )
        self.assertEqual(
            spans_by_name["bizagents.workflow"].attributes[
                "bizagents.workflow.status"
            ],
            "success",
        )
        self.assertTrue(
            all(
                options
                == {
                    "record_exception": False,
                    "set_status_on_exception": False,
                }
                for options in tracer.start_options
            )
        )
        for stage_name in ("validation", "research", "analysis", "report"):
            attributes = spans_by_name[f"bizagents.stage.{stage_name}"].attributes
            self.assertEqual(attributes["bizagents.stage.status"], "completed")
            self.assertIsInstance(attributes["bizagents.duration_ms"], float)
        all_attributes = repr(
            [span.attributes for span in tracer.spans]
        )
        self.assertNotIn(self.profile.business_profile_id, all_attributes)
        self.assertNotIn(self.profile.offering, all_attributes)

    def test_partial_and_failed_stage_outcomes_are_recorded(self) -> None:
        partial = replace(self.research, status=ResearchStatus.PARTIAL)
        partial_tracer = FakeTracer()
        partial_result = ManagerAgent(
            FakeResearchAgent(partial),
            FakeAnalysisAgent(self.analysis),
            tracer=partial_tracer,
        ).run(self.profile)
        research_span = next(
            span
            for span in partial_tracer.spans
            if span.name == "bizagents.stage.research"
        )
        self.assertEqual(partial_result.status, WorkflowStatus.PARTIAL)
        partial_workflow_span = next(
            span for span in partial_tracer.spans if span.name == "bizagents.workflow"
        )
        self.assertEqual(
            partial_workflow_span.attributes["bizagents.workflow.status"],
            "partial",
        )
        self.assertEqual(research_span.attributes["bizagents.stage.status"], "partial")
        self.assertEqual(
            research_span.attributes["bizagents.stage.attempts"],
            1,
        )

        failed_tracer = FakeTracer()
        failed_result = ManagerAgent(
            FakeResearchAgent(OSError("research offline")),
            tracer=failed_tracer,
        ).run(self.profile)
        failed_research_span = next(
            span
            for span in failed_tracer.spans
            if span.name == "bizagents.stage.research"
        )
        self.assertEqual(failed_result.status, WorkflowStatus.FAILED)
        self.assertEqual(
            failed_research_span.attributes["bizagents.stage.status"],
            "failed",
        )
        self.assertTrue(failed_research_span.attributes["bizagents.error"])
        self.assertEqual(
            failed_research_span.attributes["bizagents.stage.attempts"],
            2,
        )
        self.assertTrue(all(span.finished for span in failed_tracer.spans))
        skipped_stages = {
            span.name: span
            for span in failed_tracer.spans
            if ".stage." in span.name and span.name.rsplit(".", 1)[-1] in {"analysis", "report"}
        }
        self.assertEqual(
            {
                name: span.attributes["bizagents.stage.status"]
                for name, span in skipped_stages.items()
            },
            {
                "bizagents.stage.analysis": "skipped",
                "bizagents.stage.report": "skipped",
            },
        )
        self.assertTrue(all(span.finished for span in partial_tracer.spans))

    def test_invalid_profile_traces_skipped_stages(self) -> None:
        tracer = FakeTracer()
        result = ManagerAgent(FakeResearchAgent(self.research), tracer=tracer).run(
            cast(BusinessProfile, None)
        )

        self.assertEqual(result.status, WorkflowStatus.INVALID_INPUT)
        spans_by_name = {span.name: span for span in tracer.spans}
        for stage_name in ("research", "analysis", "report"):
            span = spans_by_name[f"bizagents.stage.{stage_name}"]
            self.assertEqual(span.attributes["bizagents.stage.status"], "skipped")
            self.assertEqual(span.attributes["bizagents.stage.attempts"], 0)
            self.assertTrue(span.finished)

    def test_tracing_export_failure_does_not_change_business_failure(self) -> None:
        failed_result = ManagerAgent(
            FakeResearchAgent(OSError("research offline")),
            tracer=FakeTracer(fail_on_finish=True),
        ).run(self.profile)
        successful_result = ManagerAgent(
            FakeResearchAgent(self.research),
            FakeAnalysisAgent(self.analysis),
            tracer=FakeTracer(fail_on_finish=True),
        ).run(self.profile)

        self.assertEqual(failed_result.status, WorkflowStatus.FAILED)
        self.assertEqual(
            self._stage(failed_result, "research").status,
            StageStatus.FAILED,
        )
        self.assertEqual(successful_result.status, WorkflowStatus.SUCCESS)

    def test_finish_failure_does_not_mask_original_exception(self) -> None:
        original_error = RuntimeError("sensitive original error")
        tracer = FakeTracer(fail_on_finish=True)

        with self.assertRaises(RuntimeError) as raised:
            with traced_span(tracer, "bizagents.test"):
                raise original_error

        self.assertIs(raised.exception, original_error)
        self.assertTrue(tracer.spans[0].finished)
        self.assertNotIn("sensitive original error", repr(tracer.spans[0].attributes))

    @staticmethod
    def _stage(result: WorkflowResult, name: str):
        return next(stage for stage in result.stages if stage.name == name)


if __name__ == "__main__":
    unittest.main()
