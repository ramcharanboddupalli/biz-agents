import unittest

from agents import (
    ConfirmationStatus,
    IntakeAgent,
    IntakeField,
    IntakeSession,
    IntakeState,
)


class IntakeAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = IntakeAgent()

    def test_new_session_starts_in_new_then_asks_one_initial_question(self) -> None:
        session = IntakeSession()
        self.assertEqual(session.state, IntakeState.NEW)
        self.assertEqual(len(session.missing_information), 5)

        turn = self.agent.start()

        self.assertEqual(turn.session.state, IntakeState.COLLECTING)
        self.assertEqual(turn.session.pending_field, IntakeField.OFFERING)
        self.assertEqual(turn.message, "What does your business sell or offer?")
        self.assertEqual(turn.message.count("?"), 1)

    def test_collects_each_required_field_one_question_at_a_time(self) -> None:
        turn = self.agent.start()
        answers = (
            "Handmade organic soaps",
            "People seeking natural skincare",
            "Hyderabad",
            "₹50,000",
            "Increase local customers and monthly sales",
        )
        expected_questions = (
            "Who are your main target customers?",
            "Where is your business located?",
            "What budget are you planning to use for your business goals?",
            "What is your main business goal?",
        )

        for index, answer in enumerate(answers):
            turn = self.agent.respond(turn.session, answer)
            if index < len(expected_questions):
                self.assertEqual(turn.message, expected_questions[index])
                self.assertEqual(turn.message.count("?"), 1)
                self.assertEqual(turn.session.state, IntakeState.COLLECTING)

        self.assertEqual(turn.session.state, IntakeState.CONFIRMATION_PENDING)
        self.assertEqual(turn.session.pending_field, None)
        self.assertEqual(turn.session.missing_information, ())
        self.assertEqual(
            turn.session.collected_information,
            {
                IntakeField.OFFERING: answers[0],
                IntakeField.TARGET_CUSTOMERS: answers[1],
                IntakeField.LOCATION: answers[2],
                IntakeField.BUDGET: answers[3],
                IntakeField.GOALS: answers[4],
            },
        )

    def test_unclear_answer_is_clarified_without_skipping_or_overwriting(self) -> None:
        turn = self.agent.start()
        turn = self.agent.respond(turn.session, "not sure")

        self.assertEqual(turn.session.state, IntakeState.COLLECTING)
        self.assertEqual(turn.session.pending_field, IntakeField.OFFERING)
        self.assertIsNone(turn.session.data.offering)
        self.assertEqual(turn.session.missing_information[0], IntakeField.OFFERING)
        self.assertIn("clarify", turn.message.casefold())
        self.assertEqual(turn.message.count("?"), 1)

        turn = self.agent.respond(turn.session, "Handmade candles")
        self.assertEqual(turn.session.data.offering, "Handmade candles")
        self.assertEqual(turn.session.pending_field, IntakeField.TARGET_CUSTOMERS)

    def test_incomplete_answer_is_clarified_without_being_recorded(self) -> None:
        turn = self.agent.start()
        turn = self.agent.respond(turn.session, "We sell")

        self.assertEqual(turn.session.pending_field, IntakeField.OFFERING)
        self.assertIsNone(turn.session.data.offering)
        self.assertEqual(turn.session.state, IntakeState.COLLECTING)
        self.assertIn("clarify", turn.message.casefold())

    def test_field_specific_vague_answers_require_clarification(self) -> None:
        turn = self.agent.start()
        for answer, expected_field in (
            ("Handmade candles", IntakeField.TARGET_CUSTOMERS),
            ("People", IntakeField.TARGET_CUSTOMERS),
        ):
            turn = self.agent.respond(turn.session, answer)
            self.assertEqual(turn.session.pending_field, expected_field)
        self.assertIsNone(turn.session.data.target_customers)
        self.assertIn("clarify", turn.message.casefold())

    def test_explicitly_undecided_budget_is_recorded_without_invention(self) -> None:
        turn = self.agent.start()
        for answer in (
            "Handmade candles",
            "Local gift shoppers",
            "Pune",
            "I haven't decided",
        ):
            turn = self.agent.respond(turn.session, answer)

        self.assertEqual(turn.session.data.budget, "not decided yet")
        self.assertEqual(turn.session.pending_field, IntakeField.GOALS)

    def test_summary_is_only_generated_after_all_required_fields(self) -> None:
        turn = self.agent.start()
        for answer in (
            "Custom furniture",
            "Homeowners",
            "Chennai",
            "₹80,000",
        ):
            turn = self.agent.respond(turn.session, answer)
            self.assertIsNone(turn.session.summary)

        turn = self.agent.respond(turn.session, "Grow online sales")
        self.assertIn("You offer Custom furniture", turn.session.summary)
        self.assertIn("to Homeowners in Chennai", turn.session.summary)
        self.assertIn("budget of ₹80,000", turn.session.summary)
        self.assertIn("main goal is Grow online sales", turn.session.summary)
        self.assertIn(turn.session.summary, turn.message)
        self.assertIn("Is this information correct?", turn.message)
        self.assertEqual(
            turn.session.confirmation_status,
            ConfirmationStatus.PENDING,
        )

    def test_rejection_requests_correction_and_does_not_complete(self) -> None:
        turn = self._complete_information()
        turn = self.agent.respond(turn.session, "No")

        self.assertEqual(turn.session.state, IntakeState.CONFIRMATION_PENDING)
        self.assertEqual(turn.session.confirmation_status, ConfirmationStatus.REJECTED)
        self.assertTrue(turn.session.awaiting_correction)
        self.assertIsNotNone(turn.session.summary)
        self.assertIn("What should I correct?", turn.message)

    def test_invalid_correction_value_is_not_saved(self) -> None:
        turn = self._complete_information()
        original_budget = turn.session.data.budget
        turn = self.agent.respond(turn.session, "No, the budget is wrong.")

        self.assertEqual(turn.session.data.budget, original_budget)
        self.assertEqual(turn.session.state, IntakeState.CONFIRMATION_PENDING)
        self.assertEqual(turn.session.confirmation_status, ConfirmationStatus.REJECTED)
        self.assertTrue(turn.session.awaiting_correction)

    def test_uncertain_confirmation_does_not_complete_or_mark_rejected(self) -> None:
        turn = self._complete_information()
        turn = self.agent.respond(turn.session, "Maybe")

        self.assertEqual(turn.session.state, IntakeState.CONFIRMATION_PENDING)
        self.assertEqual(turn.session.confirmation_status, ConfirmationStatus.PENDING)
        self.assertFalse(turn.session.awaiting_correction)
        self.assertIn("Please confirm", turn.message)

    def test_correction_updates_field_and_regenerates_summary(self) -> None:
        turn = self._complete_information()
        original_summary = turn.session.summary

        turn = self.agent.respond(turn.session, "No, the budget is ₹75,000.")

        self.assertEqual(turn.session.state, IntakeState.CONFIRMATION_PENDING)
        self.assertEqual(turn.session.data.budget, "₹75,000")
        self.assertNotEqual(turn.session.summary, original_summary)
        self.assertIn("budget of ₹75,000", turn.session.summary)
        self.assertIn("Here is the updated summary:", turn.message)
        self.assertEqual(turn.session.confirmation_status, ConfirmationStatus.PENDING)
        self.assertFalse(turn.session.awaiting_correction)

    def test_correction_can_be_provided_after_rejection_prompt(self) -> None:
        turn = self._complete_information()
        turn = self.agent.respond(turn.session, "No, that is wrong.")
        turn = self.agent.respond(turn.session, "The location is Pune.")

        self.assertEqual(turn.session.data.location, "Pune")
        self.assertIn("in Pune", turn.session.summary)
        self.assertIn("Is this information correct?", turn.message)

    def test_goal_correction_is_applied_to_the_correct_field(self) -> None:
        turn = self._complete_information()
        turn = self.agent.respond(
            turn.session,
            "No, my main goal is to expand into two new cities.",
        )

        self.assertEqual(turn.session.data.goals, "to expand into two new cities")
        self.assertIn("main goal is to expand into two new cities", turn.session.summary)

    def test_only_explicit_confirmation_completes_intake(self) -> None:
        turn = self._complete_information()

        turn = self.agent.respond(turn.session, "I think so, but the location is Pune.")
        self.assertNotEqual(turn.session.state, IntakeState.COMPLETED)
        self.assertEqual(turn.session.data.location, "Pune")
        self.assertEqual(turn.session.confirmation_status, ConfirmationStatus.PENDING)

        turn = self.agent.respond(turn.session, "Yes, that's correct.")
        self.assertEqual(turn.session.state, IntakeState.COMPLETED)
        self.assertEqual(turn.session.confirmation_status, ConfirmationStatus.CONFIRMED)
        self.assertFalse(turn.session.awaiting_correction)
        self.assertEqual(turn.message, "Thanks. Your business intake is complete.")

    def test_completed_or_unstarted_sessions_reject_turns(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond(IntakeSession(), "Answer")

        turn = self._complete_information()
        self.agent.respond(turn.session, "Yes")
        with self.assertRaises(RuntimeError):
            self.agent.respond(turn.session, "More information")

    def test_confirmation_cannot_complete_an_intake_with_missing_information(self) -> None:
        session = IntakeSession(state=IntakeState.CONFIRMATION_PENDING)
        turn = self.agent.respond(session, "Yes")

        self.assertEqual(turn.session.state, IntakeState.COLLECTING)
        self.assertEqual(turn.session.pending_field, IntakeField.OFFERING)
        self.assertIsNone(turn.session.summary)
        self.assertNotEqual(turn.session.state, IntakeState.COMPLETED)
        self.assertEqual(turn.message, "What does your business sell or offer?")

    def _complete_information(self):
        turn = self.agent.start()
        for answer in (
            "Handmade soaps",
            "Natural skincare shoppers",
            "Hyderabad",
            "₹50,000",
            "Increase local sales",
        ):
            turn = self.agent.respond(turn.session, answer)
        return turn


if __name__ == "__main__":
    unittest.main()
