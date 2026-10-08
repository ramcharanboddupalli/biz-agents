"""Public data models for business-profile memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from agents import ConfirmationStatus, IntakeSession, IntakeState


@dataclass(frozen=True, slots=True)
class BusinessProfile:
    offering: str
    target_customers: str
    location: str
    budget: str
    goals: str
    business_profile_id: str = "default-business"

    def __post_init__(self) -> None:
        for field_name in (
            "offering",
            "target_customers",
            "location",
            "budget",
            "goals",
            "business_profile_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())

    @classmethod
    def from_intake_session(
        cls,
        session: IntakeSession,
        *,
        business_profile_id: str = "default-business",
    ) -> BusinessProfile:
        if session.state is not IntakeState.COMPLETED:
            raise ValueError("Business profile can only be created from a completed intake.")
        if session.confirmation_status is not ConfirmationStatus.CONFIRMED:
            raise ValueError("Business profile requires explicit owner confirmation.")

        data = session.data
        values = (
            data.offering,
            data.target_customers,
            data.location,
            data.budget,
            data.goals,
        )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in values
        ):
            raise ValueError("Completed intake is missing required business information.")

        offering, target_customers, location, budget, goals = values
        return cls(
            offering=offering,
            target_customers=target_customers,
            location=location,
            budget=budget,
            goals=goals,
            business_profile_id=business_profile_id,
        )


@dataclass(frozen=True, slots=True)
class MemoryResult:
    content: str
    metadata: Mapping[str, str]
    distance: float


@dataclass(frozen=True, slots=True)
class BusinessProfileSaveResult:
    business_profile_id: str
    documents_written: int
    updated_at: datetime
