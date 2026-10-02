"""Deterministically remove older candidates within the same document family."""

from datetime import date

from app.retrieval.retriever import RetrievalResult


def resolve_versions(candidates: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep the highest-version candidates in each family, preserving order."""
    preferred_by_family: dict[str, tuple[tuple[int, ...], date]] = {}
    for candidate in candidates:
        preference = _preference_key(candidate)
        current_preference = preferred_by_family.get(candidate.family_id)
        if current_preference is None or preference > current_preference:
            preferred_by_family[candidate.family_id] = preference

    return [
        candidate
        for candidate in candidates
        if _preference_key(candidate) == preferred_by_family[candidate.family_id]
    ]


def _preference_key(candidate: RetrievalResult) -> tuple[tuple[int, ...], date]:
    return (_version_key(candidate.version), _published_at_key(candidate.published_at))


def _version_key(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()

    try:
        parts = tuple(int(part) for part in version.split("."))
    except ValueError as error:
        raise ValueError(f"Geçersiz noktalı sayısal sürüm: {version}") from error
    if any(part < 0 for part in parts):
        raise ValueError(f"Geçersiz noktalı sayısal sürüm: {version}")

    return _remove_trailing_zeroes(parts)


def _remove_trailing_zeroes(parts: tuple[int, ...]) -> tuple[int, ...]:
    while parts and parts[-1] == 0:
        parts = parts[:-1]
    return parts


def _published_at_key(published_at: str | None) -> date:
    return date.fromisoformat(published_at) if published_at else date.min
