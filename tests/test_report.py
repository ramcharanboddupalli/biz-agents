import unittest

from agents import (
    AnalysisResult,
    AnalysisStatus,
    BusinessFinding,
    BusinessOpportunity,
    CompetitorResearch,
    LeadResearch,
    LeadScore,
    MarketEvidenceStatus,
    MarketGap,
    MarketResearch,
    MarketSignal,
    Recommendation,
    ReportAgent,
    ReportSourceReference,
    ResearchCategory,
    ResearchIssue,
    ResearchResult,
    ResearchSource,
    ResearchStatus,
)
from rag import BusinessProfile


class ReportAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = BusinessProfile(
            offering="Handmade organic soaps",
            target_customers="natural skincare customers",
            location="Hyderabad",
            budget="ask finance first",
            goals="grow local sales",
            business_profile_id="soap-shop",
        )
        self.source = ResearchSource(
            source_id="s1",
            title="Local skincare survey",
            url="https://example.test/survey",
            snippet="A survey reports an unmet need for natural skincare customers.",
            provider="synthetic",
            source_type="web",
            metadata={},
        )
        self.business = BusinessFinding(
            "Business context",
            "A source-reported description of handmade soap buyers.",
            ("s1",),
        )
        self.market = MarketResearch(
            "Market need",
            self.source.snippet,
            ("s1",),
        )
        self.competitor = CompetitorResearch(
            "Local Soap Co",
            "Sells handmade soaps in the area.",
            "Matched the supplied competitor query.",
            ("s1",),
        )
        self.lead = LeadResearch(
            "Natural Living Store",
            "A local retailer of natural skincare products.",
            "Potentially relevant to the supplied target customers.",
            ("s1",),
        )
        self.score = LeadScore(
            lead_name="Natural Living Store",
            score=60,
            rank=1,
            priority="Medium",
            rationale="Score 60/100 = target customers +30, business location +20.",
            source_ids=("s1",),
            evidence=(self.source.snippet,),
            uncertainty="This heuristic is not a conversion probability.",
        )
        self.signal = MarketSignal(
            "Source-reported market signal: " + self.source.snippet,
            (self.source.snippet,),
            ("s1",),
            "Not independently verified.",
        )
        self.gap = MarketGap(
            "Potential gap to investigate.",
            "The signal overlaps the target customers.",
            (self.source.snippet,),
            ("s1",),
            "unvalidated-potential-gap",
            "Demand and viability remain unknown.",
        )
        self.opportunity = BusinessOpportunity(
            "Validate a possible retailer fit.",
            "Could relate to the supplied target customers.",
            "Qualitative; requires validation.",
            (self.source.snippet,),
            ("s1",),
            "Research alignment does not establish a relationship.",
        )
        self.recommendation = Recommendation(
            "Review source-backed retailer fit",
            "The supplied evidence mentions natural skincare customers.",
            "Medium",
            ("s1",),
            "Read the linked source and verify any proposed cost against the budget.",
            "Advisory only; no budget fit is assumed.",
        )
        self.research = self._research()
        self.analysis = self._analysis()
        self.agent = ReportAgent()

    def test_successful_report_preserves_supplied_sections_and_context(self) -> None:
        report = self.agent.generate(
            self.research,
            self.analysis,
            profile=self.profile,
        )

        self.assertEqual(report.business_profile, self.profile)
        self.assertEqual(report.business_profile_id, "soap-shop")
        self.assertEqual(report.research_status, ResearchStatus.SUCCESS)
        self.assertEqual(report.analysis_status, AnalysisStatus.SUCCESS)
        self.assertEqual(report.business_findings, (self.business,))
        self.assertEqual(report.market_findings, (self.market,))
        self.assertEqual(report.competitors, (self.competitor,))
        self.assertEqual(report.leads, (self.lead,))
        self.assertEqual(report.lead_scores, (self.score,))
        self.assertEqual(report.market_gaps, (self.gap,))
        self.assertEqual(report.opportunities, (self.opportunity,))
        self.assertEqual(report.recommendations, (self.recommendation,))

    def test_executive_summary_uses_only_statuses_and_supplied_counts(self) -> None:
        report = self.agent.generate(self.research, self.analysis)

        self.assertIn("Research status: **success**", report.executive_summary)
        self.assertIn("Analysis status: **success**", report.executive_summary)
        self.assertIn("1 business, 1 market, 1 competitor, and 1 lead", report.executive_summary)
        self.assertIn("1 lead score, 1 market signal, and 1 potential gap", report.executive_summary)
        self.assertNotIn("guarantee", report.executive_summary.casefold())
        self.assertNotIn("conversion rate", report.executive_summary.casefold())

    def test_valid_source_references_and_citations_are_preserved(self) -> None:
        report = self.agent.generate(self.research, self.analysis)
        self.assertEqual(
            report.source_references,
            (ReportSourceReference("s1", "Local skincare survey", "https://example.test/survey"),),
        )
        for section in (
            report.business_findings,
            report.market_findings,
            report.competitors,
            report.leads,
            report.lead_scores,
            report.market_signals,
            report.market_gaps,
            report.opportunities,
            report.recommendations,
        ):
            self.assertEqual(section[0].source_ids, ("s1",))
        self.assertIn("[s1](https://example.test/survey)", report.to_markdown())

    def test_partial_research_and_warnings_are_reported(self) -> None:
        partial = self._research(
            status=ResearchStatus.PARTIAL,
            issues=(
                ResearchIssue(
                    category=ResearchCategory.MARKET,
                    query="soap market",
                    message="provider unavailable",
                ),
            ),
        )
        analysis = self._analysis(warnings=("Market evidence is limited.",))

        report = self.agent.generate(partial, analysis)

        self.assertIn("partial", report.executive_summary)
        self.assertTrue(any("partial" in item.lower() for item in report.warnings))
        self.assertTrue(any("provider unavailable" in item for item in report.warnings))
        self.assertTrue(any("Market evidence is limited." in item for item in report.warnings))
        self.assertTrue(report.next_steps)

    def test_failed_and_empty_research_are_not_reported_as_successful_findings(self) -> None:
        failed = self.agent.generate(
            self._research(status=ResearchStatus.FAILED, findings=False),
            self._analysis(status=AnalysisStatus.FAILED, empty=True),
        )
        empty = self.agent.generate(
            self._research(findings=False),
            self._analysis(status=AnalysisStatus.INSUFFICIENT_EVIDENCE, empty=True),
        )

        self.assertIn("Research failed", " ".join(failed.warnings))
        self.assertEqual(failed.business_findings, ())
        self.assertIn("no usable findings", " ".join(failed.missing_data).lower())
        self.assertIn("Research status: **success**", empty.executive_summary)
        self.assertEqual(empty.business_findings, ())
        self.assertTrue(any("no usable findings" in item.lower() for item in empty.missing_data))
        markdown = empty.to_markdown()
        self.assertIn("No business findings were supplied.", markdown)
        self.assertIn("No market findings were supplied.", markdown)

    def test_missing_failed_and_empty_analysis_are_explicit(self) -> None:
        missing = self.agent.generate(self.research)
        failed = self.agent.generate(
            self.research,
            self._analysis(status=AnalysisStatus.FAILED, empty=True),
        )
        empty = self.agent.generate(
            self.research,
            self._analysis(status=AnalysisStatus.INSUFFICIENT_EVIDENCE, empty=True),
        )

        self.assertIsNone(missing.analysis_status)
        self.assertTrue(any("No analysis result" in item for item in missing.missing_data))
        self.assertIn("Analysis was not supplied", " ".join(missing.warnings))
        self.assertEqual(failed.analysis_status, AnalysisStatus.FAILED)
        self.assertIn("Analysis failed", " ".join(failed.warnings))
        self.assertEqual(empty.analysis_status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("no usable scored leads" in item.lower() for item in empty.missing_data))

    def test_absent_data_does_not_create_findings_or_claims(self) -> None:
        report = self.agent.generate()

        self.assertIsNone(report.research_status)
        self.assertIsNone(report.analysis_status)
        self.assertEqual(report.source_references, ())
        self.assertEqual(report.business_findings, ())
        self.assertEqual(report.market_findings, ())
        self.assertEqual(report.competitors, ())
        self.assertEqual(report.leads, ())
        self.assertEqual(report.lead_scores, ())
        self.assertEqual(report.market_gaps, ())
        self.assertEqual(report.opportunities, ())
        self.assertEqual(report.recommendations, ())
        self.assertIn("No research result was supplied", " ".join(report.missing_data))
        self.assertIn("No analysis result was supplied", " ".join(report.missing_data))
        self.assertIn("not supplied or invalid", report.executive_summary)

    def test_leads_scores_opportunities_and_recommendations_render_accurately(self) -> None:
        markdown = self.agent.generate(
            self.research,
            self.analysis,
            profile=self.profile,
        ).to_markdown()

        self.assertIn("Natural Living Store - 60/100 (rank 1, Medium)", markdown)
        self.assertIn("Validate a possible retailer fit.", markdown)
        self.assertIn("Potential only:", markdown)
        self.assertIn("Review source-backed retailer fit.", markdown)
        self.assertIn("verify any proposed cost against the budget", markdown)
        self.assertIn("no action is performed", markdown.casefold())

    def test_markdown_is_deterministic_and_has_plain_missing_sections(self) -> None:
        first = self.agent.generate(self.research, self.analysis).to_markdown()
        second = self.agent.generate(self.research, self.analysis).to_markdown()

        self.assertEqual(first, second)
        self.assertIn("# Business Report", first)
        self.assertIn("## Source References", first)
        self.assertIn("## Warnings", first)
        self.assertIn("No warnings reported.", first)

    def test_invalid_and_conflicting_sources_are_excluded_from_all_citations(self) -> None:
        conflict = ResearchSource(
            "s1", "Conflicting", "https://example.test/other", "Other snippet",
            "synthetic", "web", {},
        )
        bad_url = ResearchSource(
            "bad", "Invalid URL", "javascript:alert(1)", "Unsupported",
            "synthetic", "web", {},
        )
        report = self.agent.generate(
            self._research(sources=(self.source, conflict, bad_url)),
            self.analysis,
        )

        self.assertEqual(report.source_references, ())
        self.assertEqual(report.business_findings[0].source_ids, ())
        self.assertEqual(report.lead_scores[0].source_ids, ())
        self.assertTrue(any("conflicting records" in item for item in report.warnings))
        self.assertTrue(any("invalid URL" in item for item in report.warnings))
        self.assertNotIn("[s1](", report.to_markdown())

    def test_missing_source_ids_are_filtered_and_warned(self) -> None:
        bad_analysis = self._analysis(
            opportunities=(
                BusinessOpportunity(
                    "Unverified prospect.",
                    "Potential.",
                    "Unknown.",
                    (),
                    ("missing-id", "s1"),
                ),
            )
        )
        report = self.agent.generate(self.research, bad_analysis)

        self.assertEqual(report.opportunities[0].source_ids, ("s1",))
        self.assertTrue(any("invalid source IDs" in item for item in report.warnings))

    def test_source_less_findings_remain_unverified_and_are_marked(self) -> None:
        unreferenced = BusinessFinding(
            "Unreferenced claim",
            "No supporting source was supplied.",
            (),
        )
        research = self._research()
        object.__setattr__(research, "business_findings", (unreferenced,))

        report = self.agent.generate(research, self._analysis())

        self.assertEqual(report.business_findings, (unreferenced,))
        self.assertTrue(
            any("cannot be traced to supplied research" in item for item in report.warnings)
        )
        self.assertNotIn("Sources:", report.to_markdown().split("## Business Findings", 1)[1].split("## Market Findings", 1)[0])

    def test_analysis_url_not_in_research_is_not_added_as_a_reference(self) -> None:
        bad_analysis = self._analysis(
            source_references=(
                "https://example.test/survey",
                "https://invented.example.test/source",
            )
        )
        report = self.agent.generate(self.research, bad_analysis)

        self.assertEqual(len(report.source_references), 1)
        self.assertTrue(any("not present among valid research sources" in item for item in report.warnings))
        self.assertNotIn("invented.example.test", report.to_markdown())

    def test_mismatched_profile_and_research_ids_are_visible_and_warned(self) -> None:
        mismatched = BusinessProfile(
            offering=self.profile.offering,
            target_customers=self.profile.target_customers,
            location=self.profile.location,
            budget=self.profile.budget,
            goals=self.profile.goals,
            business_profile_id="different-business",
        )

        report = self.agent.generate(self.research, self.analysis, profile=mismatched)

        self.assertEqual(report.business_profile.business_profile_id, "different-business")
        self.assertEqual(report.business_profile_id, "soap-shop")
        self.assertTrue(any("does not match" in item for item in report.warnings))

    def test_malformed_collections_and_finding_fields_are_omitted_with_warnings(self) -> None:
        malformed_market = MarketResearch("Market", "Summary", ("s1",))
        object.__setattr__(malformed_market, "summary", None)
        research = self._research()
        object.__setattr__(research, "market_findings", (malformed_market,))
        analysis = self._analysis()
        object.__setattr__(analysis, "market_signals", "not a sequence")

        report = self.agent.generate(research, analysis)

        self.assertEqual(report.market_findings, ())
        self.assertEqual(report.market_signals, ())
        self.assertTrue(any("malformed fields" in item for item in report.warnings))
        self.assertTrue(any("collection is malformed" in item for item in report.warnings))
        self.assertIn("No market findings were supplied.", report.to_markdown())

    def test_wrong_top_level_types_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "ResearchResult"):
            self.agent.generate(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "AnalysisResult"):
            self.agent.generate(self.research, object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "BusinessProfile"):
            self.agent.generate(self.research, profile=object())  # type: ignore[arg-type]

    def _research(
        self,
        *,
        status: ResearchStatus = ResearchStatus.SUCCESS,
        sources: tuple[ResearchSource, ...] | None = None,
        issues: tuple[ResearchIssue, ...] = (),
        findings: bool = True,
    ) -> ResearchResult:
        return ResearchResult(
            business_profile_id="soap-shop",
            status=status,
            business_findings=(self.business,) if findings else (),
            market_findings=(self.market,) if findings else (),
            competitors=(self.competitor,) if findings else (),
            leads=(self.lead,) if findings else (),
            sources=(self.source,) if sources is None and findings else (sources or ()),
            issues=issues,
        )

    def _analysis(
        self,
        *,
        status: AnalysisStatus = AnalysisStatus.SUCCESS,
        empty: bool = False,
        warnings: tuple[str, ...] = (),
        opportunities: tuple[BusinessOpportunity, ...] | None = None,
        source_references: tuple[str, ...] = ("https://example.test/survey",),
    ) -> AnalysisResult:
        if empty:
            return AnalysisResult(
                status=status,
                market_status=MarketEvidenceStatus.UNKNOWN,
                source_references=source_references,
                warnings=warnings,
            )
        return AnalysisResult(
            status=status,
            lead_scores=(self.score,),
            market_signals=(self.signal,),
            market_gaps=(self.gap,),
            market_status=MarketEvidenceStatus.POTENTIAL_GAP,
            opportunities=(self.opportunity,) if opportunities is None else opportunities,
            recommendations=(self.recommendation,),
            source_references=source_references,
            warnings=warnings,
        )

if __name__ == "__main__":
    unittest.main()
