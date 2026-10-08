"""Deterministic, text-based business intake conversation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class IntakeState(str, Enum):
    NEW = "NEW"
    COLLECTING = "COLLECTING"
    CONFIRMATION_PENDING = "CONFIRMATION_PENDING"
    COMPLETED = "COMPLETED"


class IntakeField(str, Enum):
    OFFERING = "offering"
    TARGET_CUSTOMERS = "target_customers"
    LOCATION = "location"
    BUDGET = "budget"
    GOALS = "goals"


class ConfirmationStatus(str, Enum):
    REJECTED = "REJECTED"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"


@dataclass
class IntakeData:
    offering: str | None = None
    target_customers: str | None = None
    location: str | None = None
    budget: str | None = None
    goals: str | None = None


@dataclass
class IntakeSession:
    """All state for one intake; create a separate instance per conversation."""

    data: IntakeData = field(default_factory=IntakeData)
    state: IntakeState = IntakeState.NEW
    pending_field: IntakeField | None = None
    summary: str | None = None
    confirmation_status: ConfirmationStatus | None = None
    awaiting_correction: bool = False

    @property
    def collected_information(self) -> dict[IntakeField, str]:
        return {
            field_name: value
            for field_name in _FIELD_ORDER
            if (value := getattr(self.data, field_name.value)) is not None
        }

    @property
    def missing_information(self) -> tuple[IntakeField, ...]:
        return tuple(
            field_name
            for field_name in _FIELD_ORDER
            if getattr(self.data, field_name.value) is None
        )


@dataclass(frozen=True)
class IntakeTurn:
    session: IntakeSession
    message: str


class IntakeStateError(RuntimeError):
    """Raised when a turn is submitted before start or after completion."""


_FIELD_ORDER = (
    IntakeField.OFFERING,
    IntakeField.TARGET_CUSTOMERS,
    IntakeField.LOCATION,
    IntakeField.BUDGET,
    IntakeField.GOALS,
)

_QUESTIONS = {
    IntakeField.OFFERING: "What does your business sell or offer?",
    IntakeField.TARGET_CUSTOMERS: "Who are your main target customers?",
    IntakeField.LOCATION: "Where is your business located?",
    IntakeField.BUDGET: "What budget are you planning to use for your business goals?",
    IntakeField.GOALS: "What is your main business goal?",
}

_UNCLEAR_ANSWERS = {
    "dunno",
    "incorrect",
    "i dont know",
    "i do not know",
    "i have not decided",
    "i havent decided",
    "idk",
    "maybe",
    "no idea",
    "not sure",
    "not decided",
    "not decided yet",
    "not correct",
    "not right",
    "no",
    "something",
    "somewhere",
    "tbd",
    "unknown",
    "wrong",
    "nope",
}

_UNDECIDED_BUDGET_ANSWERS = {
    "havent decided",
    "have not decided",
    "i havent decided",
    "i have not decided",
    "i havent decided yet",
    "i have not decided yet",
    "not decided",
    "not decided yet",
    "no budget yet",
    "budget is undecided",
}

_VAGUE_ANSWERS = {
    IntakeField.OFFERING: {"business", "products", "services", "stuff", "things"},
    IntakeField.TARGET_CUSTOMERS: {
        "anybody",
        "anyone",
        "clients",
        "customers",
        "everyone",
        "people",
    },
    IntakeField.LOCATION: {"here", "there"},
    IntakeField.BUDGET: {"a lot", "a little", "some", "whatever"},
    IntakeField.GOALS: {"do better", "grow", "growth", "success"},
}

_INCOMPLETE_ANSWERS = {
    IntakeField.OFFERING: {
        "i offer",
        "i provide",
        "i sell",
        "we offer",
        "we provide",
        "we sell",
    },
    IntakeField.TARGET_CUSTOMERS: {
        "my target customers",
        "our target customers",
        "target audience",
        "target customer",
        "target customers",
    },
    IntakeField.LOCATION: {"at", "based in", "city of", "in", "located in"},
    IntakeField.BUDGET: {
        "about",
        "approximately",
        "around",
        "budget",
        "budget is",
        "my budget",
        "our budget",
    },
    IntakeField.GOALS: {
        "goal",
        "i hope to",
        "i plan to",
        "i want to",
        "main goal",
        "my goal",
        "my main goal",
        "our goal",
        "our main goal",
        "we hope to",
        "we plan to",
        "we want to",
    },
}

_CORRECTION_PATTERNS = (
    (
        IntakeField.OFFERING,
        re.compile(
            r"\b(?:business offering|offering|products?|services?|"
            r"what (?:i|we) (?:sell|offer))\b"
            r"\s*(?:(?:is|are)\s+|[:=]\s*)?(.+)",
            re.IGNORECASE,
        ),
    ),
    (
        IntakeField.TARGET_CUSTOMERS,
        re.compile(
            r"\b(?:target customers?|target audience|customers?|clients?)\b"
            r"\s*(?:(?:is|are)\s+|[:=]\s*)?(.+)",
            re.IGNORECASE,
        ),
    ),
    (
        IntakeField.LOCATION,
        re.compile(
            r"\b(?:business location|location|based in|located in|city|country)\b"
            r"\s*(?:(?:is|are)\s+|[:=]\s*)?(.+)",
            re.IGNORECASE,
        ),
    ),
    (
        IntakeField.BUDGET,
        re.compile(
            r"\b(?:business )?budget\b\s*(?:(?:is|are)\s+|[:=]\s*)?(.+)",
            re.IGNORECASE,
        ),
    ),
    (
        IntakeField.GOALS,
        re.compile(
            r"\b(?:(?:business )?goals?|main goal)\b"
            r"\s*(?:(?:is|are)\s+|[:=]\s*)?(.+)",
            re.IGNORECASE,
        ),
    ),
)

_LEADING_CORRECTION_WORDS = re.compile(
    r"^\s*(?:(?:actually|correction|instead|rather)\b[:,]?\s*"
    r"|(?:it|that)\s+(?:should|needs to)\s+be\s+)",
    re.IGNORECASE,
)


class IntakeAgent:
    """Runs an in-memory intake conversation without an LLM or external service."""

    def start(self) -> IntakeTurn:
        session = IntakeSession()
        session.state = IntakeState.COLLECTING
        session.pending_field = _FIELD_ORDER[0]
        return IntakeTurn(session=session, message=_QUESTIONS[session.pending_field])

    def respond(self, session: IntakeSession, user_message: str) -> IntakeTurn:
        if session.state is IntakeState.NEW:
            raise IntakeStateError("Start the intake before submitting an answer.")
        if session.state is IntakeState.COMPLETED:
            raise IntakeStateError("This intake is already complete.")
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string.")

        answer = user_message.strip()
        if session.state is IntakeState.COLLECTING:
            return self._collect_answer(session, answer)
        return self._handle_confirmation(session, answer)

    def _collect_answer(self, session: IntakeSession, answer: str) -> IntakeTurn:
        pending_field = session.pending_field
        if pending_field is None:
            raise IntakeStateError("The intake has no question awaiting an answer.")
        if not self._is_clear_answer(pending_field, answer):
            return IntakeTurn(
                session=session,
                message=self._clarification_question(pending_field),
            )

        setattr(session.data, pending_field.value, self._normalize_answer(pending_field, answer))
        next_field = next(
            (
                field_name
                for field_name in session.missing_information
                if field_name is not pending_field
            ),
            None,
        )
        if next_field is not None:
            session.pending_field = next_field
            return IntakeTurn(session=session, message=_QUESTIONS[next_field])

        session.pending_field = None
        session.state = IntakeState.CONFIRMATION_PENDING
        session.summary = self._generate_summary(session.data)
        session.confirmation_status = ConfirmationStatus.PENDING
        return IntakeTurn(
            session=session,
            message=f"Here is what I understood:\n{session.summary}\n\n"
            "Is this information correct?",
        )

    def _handle_confirmation(self, session: IntakeSession, answer: str) -> IntakeTurn:
        missing = session.missing_information
        if missing:
            session.state = IntakeState.COLLECTING
            session.pending_field = missing[0]
            session.summary = None
            session.confirmation_status = None
            session.awaiting_correction = False
            return IntakeTurn(
                session=session,
                message=_QUESTIONS[session.pending_field],
            )

        if self._is_confirmation(answer):
            session.state = IntakeState.COMPLETED
            session.confirmation_status = ConfirmationStatus.CONFIRMED
            session.awaiting_correction = False
            return IntakeTurn(
                session=session,
                message="Thanks. Your business intake is complete.",
            )

        correction = self._parse_correction(answer)
        if correction is not None:
            field_name, value = correction
            if self._is_clear_answer(field_name, value):
                setattr(
                    session.data,
                    field_name.value,
                    self._normalize_answer(field_name, value),
                )
                session.summary = self._generate_summary(session.data)
                session.confirmation_status = ConfirmationStatus.PENDING
                session.awaiting_correction = False
                return IntakeTurn(
                    session=session,
                    message=f"Here is the updated summary:\n{session.summary}\n\n"
                    "Is this information correct?",
                )

        if session.awaiting_correction or self._is_rejection(answer):
            session.state = IntakeState.CONFIRMATION_PENDING
            session.confirmation_status = ConfirmationStatus.REJECTED
            session.awaiting_correction = True
            return IntakeTurn(
                session=session,
                message=(
                    "What should I correct? Please name one part—offering, target "
                    "customers, location, budget, or goals—and provide the correct "
                    "information."
                ),
            )

        return IntakeTurn(
            session=session,
            message=(
                "Please confirm whether this summary is correct, or tell me what "
                "to correct."
            ),
        )

    @staticmethod
    def _is_clear_answer(field_name: IntakeField, answer: str) -> bool:
        normalized = " ".join(
            re.sub(r"[^\w\s]", "", answer.casefold()).split()
        )
        if not normalized:
            return False
        if field_name is IntakeField.BUDGET and normalized in _UNDECIDED_BUDGET_ANSWERS:
            return True
        return (
            normalized not in _UNCLEAR_ANSWERS
            and normalized not in _VAGUE_ANSWERS[field_name]
            and normalized not in _INCOMPLETE_ANSWERS[field_name]
        )

    @staticmethod
    def _normalize_answer(field_name: IntakeField, answer: str) -> str:
        normalized = " ".join(answer.split())
        plain_text = " ".join(
            re.sub(r"[^\w\s]", "", normalized.casefold()).split()
        )
        if (
            field_name is IntakeField.BUDGET
            and plain_text in _UNDECIDED_BUDGET_ANSWERS
        ):
            return "not decided yet"
        return normalized

    @staticmethod
    def _clarification_question(field_name: IntakeField) -> str:
        return {
            IntakeField.OFFERING: (
                "Could you clarify what your business sells or offers?"
            ),
            IntakeField.TARGET_CUSTOMERS: (
                "Could you clarify who your target customers are?"
            ),
            IntakeField.LOCATION: (
                "Could you clarify where your business is located?"
            ),
            IntakeField.BUDGET: (
                "Could you share your planned budget, or say if it is not decided yet?"
            ),
            IntakeField.GOALS: "Could you clarify your main business goal?",
        }[field_name]

    @staticmethod
    def _is_confirmation(answer: str) -> bool:
        normalized = " ".join(
            re.sub(r"[^\w\s]", "", answer.casefold()).split()
        )
        return normalized in {
            "yes",
            "yes thats correct",
            "yes that is correct",
            "thats correct",
            "that is correct",
            "correct",
            "looks good",
            "confirmed",
            "i confirm",
            "yep",
            "yeah thats right",
            "that is right",
        }

    @staticmethod
    def _is_rejection(answer: str) -> bool:
        normalized = " ".join(
            re.sub(r"[^\w\s]", "", answer.casefold()).split()
        )
        return normalized in {
            "no",
            "no thats incorrect",
            "no thats not correct",
            "no that is not correct",
            "no thats wrong",
            "not correct",
            "not right",
            "incorrect",
            "wrong",
            "nope",
            "not quite",
        } or normalized.startswith(("no ", "nope "))

    @staticmethod
    def _parse_correction(answer: str) -> tuple[IntakeField, str] | None:
        matches = [
            (field_name, match.group(1).strip())
            for field_name, pattern in _CORRECTION_PATTERNS
            if (match := pattern.search(answer)) is not None
        ]
        distinct_fields = {field_name for field_name, _ in matches}
        if len(distinct_fields) != 1:
            return None

        field_name, value = matches[0]
        value = _LEADING_CORRECTION_WORDS.sub("", value).strip(" \t\r\n:,.")
        if field_name is IntakeField.GOALS and value.casefold().startswith("is "):
            value = value[3:].strip()
        return field_name, value

    @staticmethod
    def _generate_summary(data: IntakeData) -> str:
        values = (
            data.offering,
            data.target_customers,
            data.location,
            data.budget,
            data.goals,
        )
        if any(value is None for value in values):
            raise IntakeStateError("Cannot summarize an intake with missing information.")
        offering, customers, location, budget, goals = values
        return (
            f"You offer {offering} to {customers} in {location}, with a budget "
            f"of {budget}. Your main goal is {goals}."
        )
