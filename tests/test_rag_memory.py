import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb

from agents import IntakeAgent, IntakeState
from rag import BusinessMemory, BusinessProfile, MemoryResult


class BusinessMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_directory.name) / "chroma"
        self.memory = BusinessMemory(self.storage_path)

    def tearDown(self) -> None:
        self.memory.close()
        self.temp_directory.cleanup()

    def test_rag_module_imports_and_chromadb_initializes(self) -> None:
        self.assertTrue(chromadb.__version__)
        self.assertIsInstance(self.memory, BusinessMemory)

    def test_default_storage_path_is_project_local_and_overridable_in_tests(self) -> None:
        expected_path = Path(self.temp_directory.name) / "default-chroma"
        with patch("rag.memory._DEFAULT_STORAGE_PATH", expected_path):
            memory = BusinessMemory()
        try:
            self.assertEqual(memory.storage_path, expected_path)
            self.assertTrue(expected_path.is_dir())
        finally:
            memory.close()

    def test_save_and_retrieve_business_profile(self) -> None:
        saved = self.memory.save_business_profile(self._profile())
        results = self.memory.retrieve(
            "What products does this business sell?",
            business_profile_id="soap-shop",
            limit=10,
        )

        self.assertEqual(saved.business_profile_id, "soap-shop")
        self.assertEqual(saved.documents_written, 6)
        self.assertTrue(results)
        self.assertIsInstance(results[0], MemoryResult)
        profile_document = next(
            result for result in results if result.metadata["field"] == "profile"
        )
        for detail in (
            "Handmade organic soaps",
            "People looking for natural skincare products",
            "Hyderabad",
            "₹50,000",
            "Increase local customers and monthly sales",
        ):
            self.assertIn(detail, profile_document.content)

    def test_retrieval_is_semantically_relevant_and_ranked(self) -> None:
        self.memory.save_business_profile(self._profile())

        results = self.memory.retrieve(
            "Who is most likely to buy the products?",
            business_profile_id="soap-shop",
        )

        self.assertTrue(results)
        self.assertEqual(results[0].metadata["field"], "target_customers")
        self.assertIn("natural skincare", results[0].content)

    def test_metadata_is_preserved(self) -> None:
        self.memory.save_business_profile(self._profile(), source="confirmed_intake")
        results = self.memory.retrieve(
            "Where is the company based?",
            business_profile_id="soap-shop",
        )

        location = next(result for result in results if result.metadata["field"] == "location")
        self.assertEqual(location.metadata["memory_type"], "business_profile")
        self.assertEqual(location.metadata["business_profile_id"], "soap-shop")
        self.assertEqual(location.metadata["source"], "confirmed_intake")
        self.assertTrue(location.metadata["updated_at"])

    def test_updating_profile_replaces_existing_field_documents(self) -> None:
        self.memory.save_business_profile(self._profile())
        updated = self._profile(location="Secunderabad")
        result = self.memory.save_business_profile(updated)

        self.assertEqual(result.documents_written, 6)
        locations = self.memory.retrieve(
            "business offering customers location budget goals",
            business_profile_id="soap-shop",
            limit=20,
        )
        self.assertEqual(len(locations), 6)
        self.assertTrue(any("Secunderabad" in item.content for item in locations))
        self.assertFalse(any("Hyderabad" in item.content for item in locations))

    def test_empty_database_returns_no_results(self) -> None:
        self.assertEqual(self.memory.retrieve("business customers"), [])

    def test_empty_query_returns_no_results(self) -> None:
        self.memory.save_business_profile(self._profile())
        self.assertEqual(self.memory.retrieve("  "), [])

    def test_unrelated_query_returns_no_relevant_results(self) -> None:
        self.memory.save_business_profile(self._profile())

        results = self.memory.retrieve(
            "What is Jupiter's atmospheric composition?",
            business_profile_id="soap-shop",
        )

        self.assertEqual(results, [])

    def test_invalid_query_arguments_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(TypeError, "query must be a string"):
            self.memory.retrieve(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.memory.retrieve("business", limit=0)
        with self.assertRaisesRegex(ValueError, "business_profile_id"):
            self.memory.retrieve("business", business_profile_id=" ")

    def test_persistence_survives_memory_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            storage_path = Path(temporary_directory) / "chroma"
            original_memory = BusinessMemory(storage_path)
            original_memory.save_business_profile(self._profile())
            original_memory.close()
            del original_memory

            reopened_memory = BusinessMemory(storage_path)
            try:
                results = reopened_memory.retrieve(
                    "monthly budget",
                    business_profile_id="soap-shop",
                )
                self.assertTrue(results)
                self.assertTrue(any("₹50,000" in item.content for item in results))
            finally:
                reopened_memory.close()

    def test_profile_can_be_created_only_from_confirmed_intake(self) -> None:
        agent = IntakeAgent()
        turn = agent.start()
        self.assertEqual(turn.session.state, IntakeState.COLLECTING)
        with self.assertRaisesRegex(ValueError, "completed intake"):
            BusinessProfile.from_intake_session(turn.session)

        for answer in (
            "Handmade organic soaps",
            "People interested in natural skincare",
            "Hyderabad",
            "₹50,000",
            "Increase local customers and monthly sales",
        ):
            turn = agent.respond(turn.session, answer)
        turn = agent.respond(turn.session, "Yes")

        profile = BusinessProfile.from_intake_session(
            turn.session,
            business_profile_id="confirmed-soap-shop",
        )
        self.assertEqual(profile.business_profile_id, "confirmed-soap-shop")
        self.assertEqual(profile.offering, "Handmade organic soaps")
        self.memory.save_business_profile(profile)
        results = self.memory.retrieve(
            "What do they sell?",
            business_profile_id="confirmed-soap-shop",
            limit=10,
        )
        self.assertTrue(any("Handmade organic soaps" in result.content for result in results))

    def test_invalid_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "offering"):
            BusinessProfile(
                offering=" ",
                target_customers="customers",
                location="city",
                budget="100",
                goals="grow",
            )
        with self.assertRaisesRegex(TypeError, "BusinessProfile"):
            self.memory.save_business_profile(None)  # type: ignore[arg-type]

    @staticmethod
    def _profile(*, location: str = "Hyderabad") -> BusinessProfile:
        return BusinessProfile(
            offering="Handmade organic soaps",
            target_customers="People looking for natural skincare products",
            location=location,
            budget="₹50,000",
            goals="Increase local customers and monthly sales",
            business_profile_id="soap-shop",
        )


if __name__ == "__main__":
    unittest.main()
