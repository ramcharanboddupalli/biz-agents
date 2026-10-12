import unittest

from agents import (
    AnalysisAgent,
    AnalysisStatus,
    MarketEvidenceStatus,
    ResearchStatus,
)
from agents.research import (
    LeadResearch,
    MarketResearch,
    ResearchResult,
    ResearchSource,
)
from rag import BusinessProfile


class AnalysisAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = BusinessProfile(
            offering="Handmade organic soaps",
            target_customers="natural skincare customers",
            location="Hyderabad",
            budget="ask finance first",
            goals="grow local sales",
            business_profile_id="soap-shop",
        )

    def test_exact_documented_scoring_formula_and_source_corroboration(self) -> None:
        source = self._source(
            "s1",
            "https://example.test/lead",
            "Handmade organic soaps for natural skincare customers in Hyderabad "
            "to grow local sales.",
        )
        lead = LeadResearch(
            "Soap buyer",
            source.snippet,
            source.snippet,
            ("s1",),
        )

        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )

        score = result.lead_scores[0]
        self.assertEqual(score.score, 100)
        self.assertEqual(
            score.evidence,
            ("Handmade organic soaps for natural skincare customers in Hyderabad to grow local sales.",),
        )
        self.assertIn("offering +30", score.rationale)
        self.assertIn("target customers +30", score.rationale)
        self.assertIn("business location +20", score.rationale)
        self.assertIn("business goal +20", score.rationale)
        self.assertIn("not a measured probability", score.uncertainty)

    def test_generic_keyword_presence_and_name_alone_do_not_score_fit(self) -> None:
        source = self._source(
            "s1", "https://example.test/generic", "A local customer market opportunity."
        )
        lead = LeadResearch(
            "Handmade organic soaps",
            "A local customer market opportunity.",
            "A general listing.",
            ("s1",),
        )

        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )

        self.assertEqual(result.lead_scores[0].score, 0)
        self.assertTrue(result.lead_scores[0].missing_evidence)
        self.assertIn("zero evidence-alignment", " ".join(result.warnings))
        self.assertEqual(result.opportunities, ())
        self.assertEqual(result.recommendations, ())

    def test_single_core_dimension_does_not_create_lead_opportunity(self) -> None:
        source = self._source(
            "s1",
            "https://example.test/customer-list",
            "Natural skincare customers appear in a general directory.",
        )
        lead = LeadResearch(
            "General directory",
            source.snippet,
            source.snippet,
            ("s1",),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )

        self.assertEqual(result.lead_scores[0].score, 30)
        self.assertEqual(result.lead_scores[0].matched_dimensions, ("target_customers",))
        self.assertIn("Weak evidence", result.lead_scores[0].uncertainty)
        self.assertEqual(result.opportunities, ())
        self.assertEqual(result.recommendations, ())
        self.assertEqual(result.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("both offering and target-customer" in warning for warning in result.warnings))

    def test_invalid_source_ids_never_award_points_or_escape_into_output(self) -> None:
        source = self._source(
            "valid", "https://example.test/valid", "Handmade organic soaps"
        )
        lead = LeadResearch(
            "Unlinked soap seller",
            "Handmade organic soaps for natural skincare customers in Hyderabad.",
            "grow local sales",
            ("missing", "valid", "missing"),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )

        self.assertEqual(result.lead_scores[0].source_ids, ("valid",))
        self.assertEqual(result.lead_scores[0].score, 30)
        self.assertIn("invalid source references", " ".join(result.warnings))
        self._assert_output_source_ids_valid(result, (source,))

    def test_missing_source_references_leave_lead_at_zero_and_report_uncertainty(self) -> None:
        source = self._source("s1", "https://example.test/source", "Unrelated content.")
        lead = LeadResearch(
            "Unsupported lead",
            "Handmade organic soaps for natural skincare customers in Hyderabad.",
            "grow local sales",
            ("unknown",),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )

        self.assertEqual(len(result.lead_scores), 1)
        self.assertEqual(result.lead_scores[0].score, 0)
        self.assertEqual(result.lead_scores[0].source_ids, ())
        self.assertTrue(result.lead_scores[0].uncertainty)
        self.assertEqual(result.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.recommendations, ())

    def test_lead_ranking_and_ties_are_deterministic(self) -> None:
        source_a = self._source(
            "a", "https://example.test/a", "natural skincare customers in Hyderabad"
        )
        source_b = self._source(
            "b", "https://example.test/b", "natural skincare customers in Hyderabad"
        )
        leads = (
            LeadResearch("Zeta", source_b.snippet, source_b.snippet, ("b",)),
            LeadResearch("Alpha", source_a.snippet, source_a.snippet, ("a",)),
            LeadResearch("Beta", source_b.snippet, source_b.snippet, ("b",)),
        )
        agent = AnalysisAgent(self.profile)
        first = agent.analyze(self._research(sources=(source_a, source_b), leads=leads))
        second = agent.analyze(
            self._research(sources=(source_b, source_a), leads=tuple(reversed(leads)))
        )

        self.assertEqual(
            [(item.lead_name, item.score, item.rank) for item in first.lead_scores],
            [(item.lead_name, item.score, item.rank) for item in second.lead_scores],
        )
        self.assertEqual([item.lead_name for item in first.lead_scores], ["Alpha", "Beta", "Zeta"])
        self.assertTrue(all(0 <= item.score <= 100 for item in first.lead_scores))

    def test_zero_score_leads_do_not_suppress_insufficient_evidence_warning(self) -> None:
        source = self._source("s1", "https://example.test/weak", "An unrelated company listing.")
        leads = (
            LeadResearch("One", "Unrelated company.", "No profile match.", ("s1",)),
            LeadResearch("Two", "Another unrelated company.", "No profile match.", ("s1",)),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=leads)
        )

        self.assertEqual([item.score for item in result.lead_scores], [0, 0])
        self.assertTrue(any("All leads have zero" in warning for warning in result.warnings))
        self.assertEqual(result.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)

    def test_location_and_goal_matches_alone_are_not_presented_as_an_opportunity(self) -> None:
        source = self._source(
            "s1",
            "https://example.test/location-only",
            "Hyderabad businesses often plan to grow local sales.",
        )
        lead = LeadResearch(
            "Generic business",
            source.snippet,
            source.snippet,
            ("s1",),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )

        self.assertEqual(result.lead_scores[0].score, 40)
        self.assertEqual(result.lead_scores[0].matched_dimensions, ("location", "goals"))
        self.assertEqual(result.opportunities, ())
        self.assertEqual(result.recommendations, ())

    def test_market_keyword_only_finding_is_not_a_market_signal_or_gap(self) -> None:
        source = self._source(
            "m1", "https://example.test/keywords", "Market opportunity and demand gap."
        )
        finding = MarketResearch("Market gap", source.snippet, ("m1",))
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), market_findings=(finding,))
        )

        self.assertEqual(result.market_signals, ())
        self.assertEqual(result.market_gaps, ())
        self.assertEqual(result.market_status, MarketEvidenceStatus.UNKNOWN)
        self.assertTrue(any("Market need is unknown" in warning for warning in result.warnings))

    def test_lack_of_market_proof_is_not_itself_an_unmet_need_signal(self) -> None:
        source = self._source(
            "m1",
            "https://example.test/no-proof",
            "There is a lack of evidence for unmet demand among natural skincare customers.",
        )
        finding = MarketResearch("Market report", source.snippet, ("m1",))
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), market_findings=(finding,))
        )

        self.assertEqual(result.market_signals, ())
        self.assertEqual(result.market_gaps, ())
        self.assertEqual(result.market_status, MarketEvidenceStatus.UNKNOWN)
        self.assertEqual(result.opportunities, ())
        self.assertEqual(result.recommendations, ())

    def test_summary_only_market_phrase_without_source_corroboration_is_unknown(self) -> None:
        source = self._source(
            "m1", "https://example.test/weak-signal", "Natural skincare trends are changing."
        )
        finding = MarketResearch(
            "Survey",
            "A survey reports an unmet need for natural skincare customers.",
            ("m1",),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), market_findings=(finding,))
        )
        self.assertEqual(result.market_signals, ())
        self.assertEqual(result.market_gaps, ())
        self.assertEqual(result.market_status, MarketEvidenceStatus.UNKNOWN)

    def test_negated_market_signal_is_not_reported_as_an_unmet_need(self) -> None:
        source = self._source(
            "m1",
            "https://example.test/negated",
            "The survey found no unmet need for natural skincare customers.",
        )
        finding = MarketResearch("Survey", source.snippet, ("m1",))
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), market_findings=(finding,))
        )
        self.assertEqual(result.market_signals, ())
        self.assertEqual(result.market_gaps, ())

    def test_source_backed_market_signal_and_profile_overlap_make_only_potential_gap(self) -> None:
        source = self._source(
            "m1",
            "https://example.test/signal",
            "A survey reports an unmet need for natural skincare customers in Hyderabad.",
        )
        finding = MarketResearch("Survey result", source.snippet, ("m1",))
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), market_findings=(finding,))
        )

        self.assertEqual(result.market_status, MarketEvidenceStatus.POTENTIAL_GAP)
        self.assertEqual(len(result.market_signals), 1)
        self.assertEqual(len(result.market_gaps), 1)
        self.assertEqual(result.market_gaps[0].confidence, "unvalidated-potential-gap")
        self.assertIn("not independently verified", result.market_signals[0].limitations)
        self.assertIn("do not establish market size", result.market_gaps[0].limitations)
        self._assert_output_source_ids_valid(result, (source,))

    def test_market_signal_without_business_relevance_is_not_a_business_gap(self) -> None:
        source = self._source(
            "m1",
            "https://example.test/unrelated",
            "A survey reports an unmet need for marine engine parts.",
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(
                sources=(source,),
                market_findings=(MarketResearch("Survey", source.snippet, ("m1",)),),
            )
        )
        self.assertEqual(result.market_status, MarketEvidenceStatus.RESEARCH_SIGNAL)
        self.assertEqual(result.market_gaps, ())
        self.assertEqual(result.opportunities, ())

    def test_no_market_evidence_does_not_infer_gap_from_profile_or_leads(self) -> None:
        source = self._source(
            "l1", "https://example.test/lead", "Handmade organic soaps for local buyers."
        )
        lead = LeadResearch(
            "Local Soap Buyer",
            "Handmade organic soaps",
            "Local natural skincare customers",
            ("l1",),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )
        self.assertEqual(result.market_status, MarketEvidenceStatus.UNKNOWN)
        self.assertEqual(result.market_gaps, ())
        self.assertTrue(any("Market need is unknown" in warning for warning in result.warnings))

    def test_market_finding_with_invalid_source_cannot_support_gap(self) -> None:
        source = self._source(
            "m1", "https://example.test/market", "An unmet need for natural skincare."
        )
        finding = MarketResearch("Survey", source.snippet, ("does-not-exist",))
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), market_findings=(finding,))
        )
        self.assertEqual(result.market_signals, ())
        self.assertEqual(result.market_gaps, ())
        self.assertEqual(result.source_references, ("https://example.test/market",))

    def test_duplicate_sources_are_deduplicated_and_conflicting_ids_excluded(self) -> None:
        duplicate = self._source(
            "same", "https://example.test/one", "natural skincare customers"
        )
        conflicting = self._source(
            "conflict", "https://example.test/two", "natural skincare customers"
        )
        conflicting_other = self._source(
            "conflict", "https://example.test/three", "different source"
        )
        lead = LeadResearch(
            "Natural skincare",
            "natural skincare customers",
            "natural skincare customers",
            ("same", "conflict"),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(
                sources=(duplicate, duplicate, conflicting, conflicting_other),
                leads=(lead,),
            )
        )

        self.assertEqual(result.source_references, ("https://example.test/one",))
        self.assertEqual(result.lead_scores[0].source_ids, ("same",))
        self.assertTrue(any("conflicting records" in warning for warning in result.warnings))
        self._assert_output_source_ids_valid(result, (duplicate,))

    def test_duplicate_source_id_with_malformed_record_invalidates_the_id(self) -> None:
        valid = self._source(
            "reused", "https://example.test/valid", "natural skincare customers"
        )
        malformed = self._source(
            "reused", "not-a-url", "natural skincare customers"
        )
        lead = LeadResearch(
            "Lead",
            "natural skincare customers",
            "natural skincare customers",
            ("reused",),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(valid, malformed), leads=(lead,))
        )
        self.assertEqual(result.source_references, ())
        self.assertEqual(result.lead_scores[0].source_ids, ())
        self.assertEqual(result.lead_scores[0].score, 0)
        self.assertTrue(any("conflicting records" in warning for warning in result.warnings))

    def test_malformed_source_url_and_fields_never_appear_as_references(self) -> None:
        invalid = ResearchSource(
            "bad",
            "Bad",
            "javascript:alert(1)",
            "natural skincare customers",
            "provider",
            "web",
            {},
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(invalid,))
        )
        self.assertEqual(result.source_references, ())
        self.assertEqual(result.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("invalid URL" in warning for warning in result.warnings))

    def test_recommendations_are_advisory_prioritized_and_budget_cautious(self) -> None:
        high = self._source(
            "high",
            "https://example.test/high",
            "Handmade organic soaps for natural skincare customers in Hyderabad "
            "to grow local sales.",
        )
        moderate = self._source(
            "moderate",
            "https://example.test/moderate",
            "natural skincare customers in Hyderabad",
        )
        leads = (
            LeadResearch("High Fit", high.snippet, high.snippet, ("high",)),
            LeadResearch(
                "Medium Fit", moderate.snippet, moderate.snippet, ("moderate",)
            ),
        )
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(high, moderate), leads=leads)
        )

        self.assertEqual(result.recommendations[0].priority, "High")
        self.assertTrue(all(item.source_ids for item in result.recommendations))
        self.assertTrue(
            all("verify the cost and available budget" in item.risks.lower() for item in result.recommendations)
        )
        self.assertTrue(all("Review" in item.action for item in result.recommendations))
        self.assertIn("Handmade organic soaps", result.recommendations[0].why)
        self.assertTrue(all("performed" not in item.action.lower() for item in result.recommendations))

    def test_zero_score_and_empty_research_do_not_create_opportunities_or_recommendations(self) -> None:
        source = self._source("s1", "https://example.test/unrelated", "No profile match.")
        lead = LeadResearch("Weak lead", "Generic organization.", "Unrelated.", ("s1",))
        weak = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,))
        )
        empty = AnalysisAgent(self.profile).analyze(self._research())
        self.assertEqual(weak.opportunities, ())
        self.assertEqual(weak.recommendations, ())
        self.assertEqual(empty.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(empty.opportunities, ())
        self.assertEqual(empty.recommendations, ())

    def test_failed_research_status_is_distinguished(self) -> None:
        failed = self._research(status=ResearchStatus.FAILED)
        result = AnalysisAgent(self.profile).analyze(failed)
        self.assertEqual(result.status, AnalysisStatus.FAILED)
        self.assertTrue(any("Research execution failed" in warning for warning in result.warnings))

    def test_research_issues_prevent_success_status(self) -> None:
        source = self._source(
            "s1",
            "https://example.test/partial",
            "natural skincare customers in Hyderabad",
        )
        lead = LeadResearch(
            "Relevant lead",
            source.snippet,
            source.snippet,
            ("s1",),
        )
        research = self._research(sources=(source,), leads=(lead,))
        object.__setattr__(research, "issues", (object(),))

        result = AnalysisAgent(self.profile).analyze(research)

        self.assertEqual(result.status, AnalysisStatus.PARTIAL)
        self.assertTrue(any("issue(s)" in warning for warning in result.warnings))
        self.assertTrue(any("malformed" in warning for warning in result.warnings))

    def test_invalid_profile_and_malformed_research_input_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "BusinessProfile"):
            AnalysisAgent(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "ResearchResult"):
            AnalysisAgent(self.profile).analyze(None)  # type: ignore[arg-type]

        malformed = self._research()
        object.__setattr__(malformed, "sources", "not a source sequence")
        with self.assertRaisesRegex(ValueError, "sources must be a sequence"):
            AnalysisAgent(self.profile).analyze(malformed)

        invalid_profile = object.__new__(BusinessProfile)
        object.__setattr__(invalid_profile, "offering", "")
        object.__setattr__(invalid_profile, "target_customers", "customers")
        object.__setattr__(invalid_profile, "location", "Hyderabad")
        object.__setattr__(invalid_profile, "budget", "unknown")
        object.__setattr__(invalid_profile, "goals", "grow")
        object.__setattr__(invalid_profile, "business_profile_id", "soap-shop")
        with self.assertRaisesRegex(ValueError, "offering"):
            AnalysisAgent(invalid_profile).analyze(self._research())  # type: ignore[arg-type]

    def test_all_substantive_output_references_resolve_to_supplied_sources(self) -> None:
        source = self._source(
            "s1",
            "https://example.test/all",
            "Handmade organic soaps for natural skincare customers in Hyderabad "
            "to grow local sales. Survey reports an unmet need for natural skincare customers.",
        )
        lead = LeadResearch("Relevant lead", source.snippet, source.snippet, ("s1",))
        market = MarketResearch("Survey", source.snippet, ("s1",))
        result = AnalysisAgent(self.profile).analyze(
            self._research(sources=(source,), leads=(lead,), market_findings=(market,))
        )
        self._assert_output_source_ids_valid(result, (source,))
        self.assertEqual(result.source_references, ("https://example.test/all",))
        self.assertTrue(result.opportunities)
        self.assertTrue(result.recommendations)

    @staticmethod
    def _source(source_id: str, url: str, snippet: str) -> ResearchSource:
        return ResearchSource(
            source_id=source_id,
            title=f"Source {source_id}",
            url=url,
            snippet=snippet,
            provider="test",
            source_type="web",
            metadata={},
        )

    @staticmethod
    def _research(
        *,
        sources: tuple[ResearchSource, ...] = (),
        leads: tuple[LeadResearch, ...] = (),
        market_findings: tuple[MarketResearch, ...] = (),
        status: ResearchStatus = ResearchStatus.SUCCESS,
    ) -> ResearchResult:
        return ResearchResult(
            business_profile_id="soap-shop",
            status=status,
            business_findings=(),
            market_findings=market_findings,
            competitors=(),
            leads=leads,
            sources=sources,
            issues=(),
        )

    @staticmethod
    def _assert_output_source_ids_valid(result, sources) -> None:
        valid_ids = {source.source_id for source in sources}
        findings = (
            *result.lead_scores,
            *result.market_signals,
            *result.market_gaps,
            *result.opportunities,
            *result.recommendations,
        )
        for finding in findings:
            assert set(finding.source_ids) <= valid_ids


if __name__ == "__main__":
    unittest.main()
