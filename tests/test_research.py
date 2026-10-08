import unittest
from collections.abc import Sequence

from agents import (
    BusinessFinding,
    CompetitorResearch,
    LeadResearch,
    MarketResearch,
    ProviderResult,
    ResearchAgent,
    ResearchCategory,
    ResearchProvider,
    ResearchStatus,
)
from rag import BusinessProfile


class FakeResearchProvider:
    def __init__(self, responses: dict[str, Sequence[object]] | None = None) -> None:
        self.responses = responses or {}
        self.queries: list[str] = []

    def search(self, query: str) -> Sequence[object]:
        self.queries.append(query)
        response = self.responses.get(query, ())
        if isinstance(response, Exception):
            raise response
        return response


class BrokenResearchProvider:
    def search(self, query: str) -> Sequence[object]:
        raise OSError("provider unavailable")


class ResearchAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = self._profile()
        self.provider = FakeResearchProvider()
        self.agent = ResearchAgent(self.provider)

    def test_accepts_valid_business_profile(self) -> None:
        result = self.agent.research(self.profile)

        self.assertEqual(result.business_profile_id, "soap-shop")
        self.assertEqual(result.status, ResearchStatus.SUCCESS)
        self.assertEqual(len(self.provider.queries), 6)

    def test_research_queries_are_generated_from_profile(self) -> None:
        plan = self.agent.build_plan(self.profile)
        queries = [item.query for item in plan]

        self.assertTrue(any("Handmade organic soaps" in query for query in queries))
        self.assertTrue(any("natural skincare customers" in query for query in queries))
        self.assertTrue(any("Hyderabad" in query for query in queries))
        self.assertTrue(any("grow local sales" in query for query in queries))
        self.assertFalse(any("₹50,000" in query for query in queries))

    def test_plan_covers_all_research_dimensions(self) -> None:
        plan = self.agent.build_plan(self.profile)

        self.assertEqual(
            {item.category for item in plan},
            {
                ResearchCategory.BUSINESS,
                ResearchCategory.MARKET,
                ResearchCategory.COMPETITORS,
                ResearchCategory.LEADS,
            },
        )

    def test_provider_is_used_through_injected_abstraction(self) -> None:
        self.agent.research(self.profile)

        self.assertEqual(len(self.provider.queries), 6)
        self.assertTrue(all(query.strip() for query in self.provider.queries))
        self.assertNotIn("Tavily", type(self.agent).__name__)
        self.assertNotIn("Serper", type(self.agent).__name__)

    def test_provider_results_become_source_aware_structured_findings(self) -> None:
        first_query = self.agent.build_plan(self.profile)[0].query
        self.provider.responses[first_query] = (
            self._result("Soap category report", "https://example.test/market"),
        )

        result = self.agent.research(self.profile)

        self.assertEqual(result.status, ResearchStatus.SUCCESS)
        self.assertEqual(len(result.business_findings), 1)
        finding = result.business_findings[0]
        self.assertIsInstance(finding, BusinessFinding)
        self.assertEqual(finding.summary, "A source-backed summary.")
        source = result.sources[0]
        self.assertEqual(source.url, "https://example.test/market")
        self.assertEqual(finding.source_ids, (source.source_id,))

    def test_competitor_market_and_lead_categories_are_structured(self) -> None:
        plan = self.agent.build_plan(self.profile)
        self.provider.responses[plan[2].query] = (
            self._result("Regional market report", "https://example.test/market"),
        )
        self.provider.responses[plan[4].query] = (
            self._result("Local Soap Co", "https://example.test/competitor"),
        )
        self.provider.responses[plan[5].query] = (
            self._result("Natural Living Store", "https://example.test/lead"),
        )

        result = self.agent.research(self.profile)

        self.assertIsInstance(result.market_findings[0], MarketResearch)
        self.assertEqual(result.market_findings[0].summary, "A source-backed summary.")
        self.assertIsInstance(result.competitors[0], CompetitorResearch)
        self.assertEqual(result.competitors[0].name, "Local Soap Co")
        self.assertTrue(result.competitors[0].relevance.startswith("Matched research query:"))
        self.assertIsInstance(result.leads[0], LeadResearch)
        self.assertEqual(result.leads[0].name, "Natural Living Store")
        source_ids = {source.source_id for source in result.sources}
        self.assertTrue(set(result.market_findings[0].source_ids) <= source_ids)
        self.assertTrue(set(result.competitors[0].source_ids) <= source_ids)
        self.assertTrue(set(result.leads[0].source_ids) <= source_ids)

    def test_duplicate_urls_are_normalized_and_deduplicated(self) -> None:
        plan = self.agent.build_plan(self.profile)
        self.provider.responses[plan[0].query] = (
            self._result("First title", "https://EXAMPLE.test/page/"),
        )
        self.provider.responses[plan[1].query] = (
            self._result("Duplicate title", "https://example.test/page#section"),
        )

        result = self.agent.research(self.profile)

        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].url, "https://example.test/page")
        self.assertEqual(len(result.business_findings), 2)
        self.assertEqual(
            result.business_findings[0].source_ids,
            result.business_findings[1].source_ids,
        )

    def test_empty_provider_results_are_successful_without_findings(self) -> None:
        result = self.agent.research(self.profile)

        self.assertEqual(result.status, ResearchStatus.SUCCESS)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.business_findings, ())
        self.assertEqual(result.issues, ())

    def test_provider_failure_is_reported_as_failed_not_fabricated(self) -> None:
        result = ResearchAgent(BrokenResearchProvider()).research(self.profile)

        self.assertEqual(result.status, ResearchStatus.FAILED)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.business_findings, ())
        self.assertEqual(len(result.issues), 6)
        self.assertTrue(all("provider unavailable" in issue.message for issue in result.issues))

    def test_mixed_provider_failure_is_reported_as_partial(self) -> None:
        plan = self.agent.build_plan(self.profile)
        self.provider.responses[plan[0].query] = (
            self._result("Business source", "https://example.test/business"),
        )
        self.provider.responses[plan[1].query] = OSError("temporary failure")

        result = self.agent.research(self.profile)

        self.assertEqual(result.status, ResearchStatus.PARTIAL)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(len(result.issues), 1)

    def test_malformed_result_and_missing_url_are_reported_and_discarded(self) -> None:
        query = self.agent.build_plan(self.profile)[0].query
        self.provider.responses[query] = (
            self._result("Missing URL", ""),
            "not a provider result",
        )

        result = self.agent.research(self.profile)

        self.assertEqual(result.status, ResearchStatus.PARTIAL)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.business_findings, ())
        self.assertEqual(len(result.issues), 2)
        self.assertTrue(any("source URL" in issue.message for issue in result.issues))
        self.assertTrue(any("malformed result" in issue.message for issue in result.issues))

    def test_invalid_profile_and_provider_contract_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "BusinessProfile"):
            self.agent.research(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "search"):
            ResearchAgent(object())  # type: ignore[arg-type]

        invalid_profile = object.__new__(BusinessProfile)
        for name, value in (
            ("offering", ""),
            ("target_customers", "customers"),
            ("location", "Hyderabad"),
            ("budget", "₹50,000"),
            ("goals", "grow"),
            ("business_profile_id", "soap-shop"),
        ):
            object.__setattr__(invalid_profile, name, value)
        with self.assertRaisesRegex(ValueError, "offering"):
            self.agent.research(invalid_profile)

    def test_missing_source_url_does_not_get_fabricated(self) -> None:
        query = self.agent.build_plan(self.profile)[0].query
        self.provider.responses[query] = (
            ProviderResult(
                title="Unlinked result",
                url="",
                snippet="No source URL was supplied.",
                provider="fake",
            ),
        )

        result = self.agent.research(self.profile)

        self.assertEqual(result.sources, ())
        self.assertEqual(result.business_findings, ())
        self.assertTrue(any("source URL" in issue.message for issue in result.issues))

    def test_invalid_url_and_malformed_metadata_are_reported(self) -> None:
        query = self.agent.build_plan(self.profile)[0].query
        malformed_metadata = ProviderResult(
            title="Malformed metadata",
            url="https://example.test/metadata",
            snippet="A snippet.",
            provider="fake",
            metadata={"count": 3},  # type: ignore[dict-item]
        )
        self.provider.responses[query] = (
            self._result("Invalid port", "https://example.test:bad/source"),
            malformed_metadata,
        )

        result = self.agent.research(self.profile)

        self.assertEqual(result.status, ResearchStatus.PARTIAL)
        self.assertEqual(result.sources, ())
        self.assertTrue(any("source URL" in issue.message for issue in result.issues))
        self.assertTrue(any("metadata keys and values" in issue.message for issue in result.issues))

    def test_malformed_provider_response_is_a_failure_issue(self) -> None:
        class InvalidResponseProvider:
            def search(self, query: str) -> Sequence[ProviderResult]:
                return None  # type: ignore[return-value]

        result = ResearchAgent(InvalidResponseProvider()).research(self.profile)

        self.assertEqual(result.status, ResearchStatus.FAILED)
        self.assertEqual(len(result.issues), 6)
        self.assertTrue(
            all("expected a sequence" in issue.message for issue in result.issues)
        )

    def test_complete_end_to_end_deterministic_research_flow(self) -> None:
        plan = self.agent.build_plan(self.profile)
        self.provider.responses = {
            item.query: (
                self._result(
                    f"{item.category.value.title()} source",
                    f"https://example.test/{item.category.value}/{index}",
                ),
            )
            for index, item in enumerate(plan)
        }

        result = self.agent.research(self.profile)

        self.assertEqual(result.status, ResearchStatus.SUCCESS)
        self.assertEqual(len(result.sources), 6)
        self.assertEqual(len(result.business_findings), 2)
        self.assertEqual(len(result.market_findings), 2)
        self.assertEqual(len(result.competitors), 1)
        self.assertEqual(len(result.leads), 1)
        self.assertEqual(result.issues, ())
        self.assertTrue(
            all(
                source_id in {source.source_id for source in result.sources}
                for finding in (
                    *result.business_findings,
                    *result.market_findings,
                    *result.competitors,
                    *result.leads,
                )
                for source_id in finding.source_ids
            )
        )

    @staticmethod
    def _profile() -> BusinessProfile:
        return BusinessProfile(
            offering="Handmade organic soaps",
            target_customers="natural skincare customers",
            location="Hyderabad",
            budget="₹50,000",
            goals="grow local sales",
            business_profile_id="soap-shop",
        )

    @staticmethod
    def _result(title: str, url: str) -> ProviderResult:
        return ProviderResult(
            title=title,
            url=url,
            snippet="A source-backed summary.",
            provider="fake-test-provider",
            metadata={"kind": "test"},
        )


if __name__ == "__main__":
    unittest.main()
