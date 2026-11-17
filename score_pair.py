"""Customer-record pair scoring for the Greyquill FDE task."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from functools import lru_cache
from difflib import SequenceMatcher
import re
import unicodedata


@dataclass
class FieldComparison:
    """How to compare one field."""
    attribute: str
    method: str
    weight: float


@dataclass
class MatchResult:
    confidence: float
    algorithm: str
    field_scores: dict[str, float]


PLAN = [
    FieldComparison("name", "phonetic", 0.25),
    FieldComparison("email", "edit_distance", 0.20),
    FieldComparison("phone", "normalized_phone", 0.20),
    FieldComparison("date_of_birth", "date", 0.20),
    FieldComparison("pin_code", "exact", 0.10),
    FieldComparison("city", "exact", 0.05),
]


def _blank(value: object) -> bool:
    if value is None:
        return True
    try:
        # Handles float NaN and pandas.NA without importing pandas.
        if value != value:
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "nat"}


def _normal_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _name_tokens(value: object) -> list[str]:
    # "Surname, Given" and "Given Surname" become order-independent tokens.
    return _normal_text(value).split()


def _soundex(word: str) -> str:
    """Return a four-character Soundex key for a normalized name token."""
    if not word:
        return ""
    groups = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4", "m": "5", "n": "5", "r": "6",
    }
    first = word[0].upper()
    digits: list[str] = []
    previous = groups.get(word[0], "")
    for char in word[1:]:
        code = groups.get(char, "")
        if code and code != previous:
            digits.append(code)
        # Vowels and separators reset adjacent-consonant suppression.
        previous = code
    return (first + "".join(digits) + "000")[:4]


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    # A matching initial is meaningful, but slightly weaker than a full token.
    if len(left) == 1 or len(right) == 1:
        short, long = (left, right) if len(left) == 1 else (right, left)
        return 0.88 if long.startswith(short) else 0.0
    character_score = SequenceMatcher(None, left, right).ratio()
    if _soundex(left) == _soundex(right):
        return max(0.86, character_score)
    return character_score


def _phonetic_name_score(left: object, right: object) -> float:
    """Order-independent token matching with an optimal one-to-one assignment."""
    a, b = _name_tokens(left), _name_tokens(right)
    if not a or not b:
        return 0.0

    # Dynamic programming avoids the order-dependent mistakes of greedy matching.
    # The shorter side is assigned into the longer side; unmatched tokens count as 0.
    if len(a) > len(b):
        a, b = b, a

    @lru_cache(maxsize=None)
    def best(index: int, used_mask: int) -> float:
        if index == len(a):
            return 0.0
        best_total = best(index + 1, used_mask)  # leave this token unmatched
        for j, token_b in enumerate(b):
            bit = 1 << j
            if not used_mask & bit:
                candidate = _token_similarity(a[index], token_b) + best(index + 1, used_mask | bit)
                best_total = max(best_total, candidate)
        return best_total

    return best(0, 0) / max(len(a), len(b))


def _levenshtein_ratio(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    # Memory-efficient dynamic-programming edit distance.
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right))


def _normalize_phone(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    if digits.startswith("91") and len(digits) > 10:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) > 10:
        digits = digits[1:]
    return digits


def _parse_date(value: object) -> date | None:
    text = str(value).strip()
    formats = (
        "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d.%m.%y", "%d-%m-%Y",
        "%d-%m-%y", "%d-%b-%Y", "%d-%B-%Y",
        "%d-%b-%y", "%d-%B-%y",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt).date()
            if "%y" in fmt:
                year = parsed.year % 100
                parsed = parsed.replace(year=(2000 + year if year <= 26 else 1900 + year))
            return parsed
        except ValueError:
            continue
    return None


def score_pair(a: dict[str, object], b: dict[str, object], plan: list[FieldComparison]) -> MatchResult:
    """Return a weighted similarity score, ignoring fields missing on either side."""
    field_scores: dict[str, float] = {}
    weighted_total = 0.0
    compared_weight = 0.0
    methods: list[str] = []

    for comparison in plan:
        left = a.get(comparison.attribute)
        right = b.get(comparison.attribute)
        if _blank(left) or _blank(right):
            continue

        method = comparison.method
        if method == "phonetic":
            score = _phonetic_name_score(left, right)
        elif method == "edit_distance":
            score = _levenshtein_ratio(_normal_text(left).replace(" ", ""),
                                       _normal_text(right).replace(" ", ""))
        elif method == "normalized_phone":
            score = float(_normalize_phone(left) == _normalize_phone(right))
        elif method == "date":
            left_date, right_date = _parse_date(left), _parse_date(right)
            score = float(left_date is not None and left_date == right_date)
        elif method == "exact":
            score = float(str(left).strip().casefold() == str(right).strip().casefold())
        else:
            raise ValueError(f"Unsupported comparison method: {method}")

        field_scores[comparison.attribute] = max(0.0, min(1.0, float(score)))
        weighted_total += field_scores[comparison.attribute] * comparison.weight
        compared_weight += comparison.weight
        if method not in methods:
            methods.append(method)

    confidence = weighted_total / compared_weight if compared_weight else 0.0
    return MatchResult(
        confidence=max(0.0, min(1.0, confidence)),
        algorithm="+".join(methods),
        field_scores=field_scores,
    )
