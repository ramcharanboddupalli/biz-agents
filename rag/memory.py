"""Persistent ChromaDB-backed business-profile memory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from .models import BusinessProfile, BusinessProfileSaveResult, MemoryResult

_COLLECTION_NAME = "business_profiles"
_DEFAULT_STORAGE_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma"
_MAX_RELEVANT_COSINE_DISTANCE = 0.9
_PROFILE_FIELDS = (
    ("offering", "Business offering"),
    ("target_customers", "Target customers"),
    ("location", "Business location"),
    ("budget", "Business budget"),
    ("goals", "Business goals"),
)


class BusinessMemory:
    """Store and semantically retrieve business profile facts locally."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        path = (
            Path(storage_path).expanduser()
            if storage_path is not None
            else _DEFAULT_STORAGE_PATH
        )
        self.storage_path = path
        try:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection: Collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=DefaultEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize ChromaDB at '{path}': {exc}"
            ) from exc

    def close(self) -> None:
        """Close the persistent client and release its local database handles."""
        self._client.close()

    def save_business_profile(
        self,
        profile: BusinessProfile,
        *,
        source: str = "intake_agent",
    ) -> BusinessProfileSaveResult:
        if not isinstance(profile, BusinessProfile):
            raise TypeError("profile must be a BusinessProfile.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string.")

        updated_at = datetime.now(UTC)
        timestamp = updated_at.isoformat()
        ids = [f"{profile.business_profile_id}:profile"]
        documents = [self._format_profile(profile)]
        metadatas = [
            {
                "memory_type": "business_profile",
                "business_profile_id": profile.business_profile_id,
                "source": source.strip(),
                "field": "profile",
                "updated_at": timestamp,
            }
        ]

        for field_name, label in _PROFILE_FIELDS:
            value = getattr(profile, field_name)
            ids.append(f"{profile.business_profile_id}:{field_name}")
            documents.append(f"{label}:\n{value}")
            metadatas.append(
                {
                    "memory_type": "business_profile",
                    "business_profile_id": profile.business_profile_id,
                    "source": source.strip(),
                    "field": field_name,
                    "updated_at": timestamp,
                }
            )

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return BusinessProfileSaveResult(
            business_profile_id=profile.business_profile_id,
            documents_written=len(ids),
            updated_at=updated_at,
        )

    def retrieve(
        self,
        query: str,
        *,
        business_profile_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        if business_profile_id is not None and (
            not isinstance(business_profile_id, str)
            or not business_profile_id.strip()
        ):
            raise ValueError("business_profile_id must be a non-empty string when provided.")

        normalized_query = query.strip()
        if not normalized_query:
            return []
        collection_size = self._collection.count()
        if collection_size == 0:
            return []

        options: dict[str, object] = {
            "query_texts": [normalized_query],
            "n_results": min(limit, collection_size),
            "include": ["documents", "metadatas", "distances"],
        }
        if business_profile_id is not None:
            options["where"] = {
                "business_profile_id": business_profile_id.strip()
            }

        response = self._collection.query(**options)
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        results = [
            MemoryResult(
                content=document,
                metadata=metadata,
                distance=float(distance),
            )
            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]
        return [
            result
            for result in results
            if result.distance <= _MAX_RELEVANT_COSINE_DISTANCE
        ]

    @staticmethod
    def _format_profile(profile: BusinessProfile) -> str:
        return "\n\n".join(
            (
                f"Business offering:\n{profile.offering}",
                f"Target customers:\n{profile.target_customers}",
                f"Business location:\n{profile.location}",
                f"Business budget:\n{profile.budget}",
                f"Business goals:\n{profile.goals}",
            )
        )
