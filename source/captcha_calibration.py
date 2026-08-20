"""Supervised OCR calibration learned from labelled HeroesWM samples.

The generic ddddocr model remains the primary recognizer. Labelled samples are
used only to derive advisory character-confusion candidates; image lookup is
deliberately not used.
"""

from __future__ import annotations

TRAINING_SAMPLE_COUNT = 20
TRAINING_LABELS = (
    "DQQQ9N",
    "226KQS",
    "AP4QXP",
    "PYVGQY",
    "S428V3",
    "CA9K6N",
    "468AH4",
    "N9SDE6",
    "H43U54",
    "VHDYV2",
    "293258",
    "HH3HUE",
    "83DHYC",
    "NXHE2S",
    "4SAPUG",
    "2PQAEX",
    "EQD4YD",
    "56KYPP",
    "X3E32S",
    "QSU9NU",
)

# position -> OCR character -> calibrated HeroesWM character
POSITION_CORRECTIONS: dict[int, dict[str, str]] = {
    0: {"T": "P"},
    3: {"C": "G", "5": "8", "4": "A"},
    4: {"B": "6", "7": "H"},
}

# Additional observations from samples 8–20. Unlike the original seven-sample
# correction these are emitted as separate alternatives, one change at a time
# and then in combinations. A legitimate S, D, T or 7 is never overwritten in
# the recognizer's original answer.
ADVISORY_POSITION_CORRECTIONS: dict[int, dict[str, str]] = {
    0: {"D": "4", "S": "8"},
    2: {"5": "S", "Q": "D"},
    3: {"T": "P"},
    4: {"T": "P"},
    5: {"C": "G", "7": "P"},
}


def calibrated_candidate(code: str) -> str:
    if len(code) != 6:
        return code
    characters = list(code)
    for position, mapping in POSITION_CORRECTIONS.items():
        characters[position] = mapping.get(characters[position], characters[position])
    return "".join(characters)


def calibrated_candidates(code: str) -> list[str]:
    """Return advisory learned variants without replacing the base answer."""
    if len(code) != 6:
        return [code]
    primary = calibrated_candidate(code)
    results = [primary]
    alternatives: list[tuple[int, str]] = []
    for position, mapping in ADVISORY_POSITION_CORRECTIONS.items():
        replacement = mapping.get(primary[position])
        if replacement and replacement != primary[position]:
            alternatives.append((position, replacement))

    # Breadth-first combinations put single observed confusions before more
    # speculative multi-character changes.
    frontier = [primary]
    for position, replacement in alternatives:
        additions: list[str] = []
        for current in frontier:
            characters = list(current)
            characters[position] = replacement
            candidate = "".join(characters)
            if candidate not in results and candidate not in additions:
                additions.append(candidate)
        results.extend(additions)
        frontier.extend(additions)
    return results
