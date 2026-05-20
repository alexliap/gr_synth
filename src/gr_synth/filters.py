"""Output filters from guide §7.

Apply order (matters): lang-ID → preamble strip → format → length → near-dup.
Each function is independent; ``apply_all`` chains them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import fasttext
import fasttext.FastText as _ft_module
import numpy as np
import regex as re
from datasketch import MinHash, MinHashLSH
from .types import Record
from .config import Settings


# fasttext-wheel is unmaintained and calls ``np.array(probs, copy=False)``
# inside ``predict()``, which NumPy 2 banned. Monkey-patch with ``np.asarray``
# at import time so callers don't need to know about it.
def _patched_predict(
    self, text, k=1, threshold=0.0, on_unicode_error="strict"
):
    def _check(entry: str) -> str:
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        return entry + "\n"

    if isinstance(text, list):
        return self.f.multilinePredict(
            [_check(e) for e in text], k, threshold, on_unicode_error
        )
    predictions = self.f.predict(_check(text), k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], ())
    return labels, np.asarray(probs)


_ft_module._FastText.predict = _patched_predict


# ---------- language ID -------------------------------------------------

# fastText's predict() raises on embedded newlines.
_NEWLINES = re.compile(r"\s+")

_LID_SLICE_LEN = 500
_LID_SLICE_COUNT = 3


def load_lid_model(settings: Settings):
    """Load the fastText ``lid.176.bin`` model. Returns the opaque model object."""
    fasttext.FastText.eprint = lambda *_a, **_k: None  # silence "Warning: ..." spam
    return fasttext.load_model(str(settings.lid_model_path))


def _predict_greek(lid_model, chunk: str) -> float:
    """Return the predicted probability that ``chunk`` is Greek (``__label__el``)."""
    chunk = _NEWLINES.sub(" ", chunk).strip()
    if not chunk:
        return 0.0
    labels, probs = lid_model.predict(chunk, k=1)
    return float(probs[0]) if labels and labels[0] == "__label__el" else 0.0


def is_greek(text: str, lid_model, threshold: float) -> tuple[bool, float]:
    """Return ``(passes, mean_confidence)``.

    For long outputs, sample 3 random ``_LID_SLICE_LEN``-char slices and require
    all of them to predict ``__label__el`` above ``threshold``. ``mean_confidence``
    is the average Greek probability across slices (or the single full-text prob
    for short inputs).
    """
    if len(text) <= _LID_SLICE_LEN:
        prob = _predict_greek(lid_model, text)
        return prob >= threshold, prob

    rng = random.Random(len(text))
    probs: list[float] = []
    max_start = len(text) - _LID_SLICE_LEN
    for _ in range(_LID_SLICE_COUNT):
        start = rng.randint(0, max_start)
        probs.append(_predict_greek(lid_model, text[start : start + _LID_SLICE_LEN]))
    mean = sum(probs) / len(probs)
    return all(p >= threshold for p in probs), mean


# ---------- preamble strip ---------------------------------------------

_PREAMBLE_PATTERNS = [
    re.compile(r"^(Φυσικά|Βεβαίως|Εντάξει|Ορίστε)[,!.\s]+", re.IGNORECASE),
    re.compile(
        r"^(Εδώ είναι|Παρακάτω είναι|Παρακάτω παρουσιάζω|Ακολουθεί)[^\n]{0,80}[:.]\s*\n+",
        re.IGNORECASE,
    ),
    re.compile(r"^\*\*[^\n*]{1,200}\*\*\s*\n+"),  # bolded title line
]

_REFUSAL_PREFIX = re.compile(
    r"^\s*(Δεν μπορώ|Λυπάμαι|Ως γλωσσικό μοντέλο|Ως μοντέλο)",
    re.IGNORECASE,
)


def strip_preamble(text: str) -> str:
    """Strip common Greek model preambles ("Φυσικά,", "Εδώ είναι...", bolded titles).

    If what's left is mostly a refusal, return "" so the length filter drops it.
    """
    out = text.lstrip()
    # Loop because a model may stack a "Φυσικά." line and a "Εδώ είναι..." line.
    while True:
        before = out
        for pat in _PREAMBLE_PATTERNS:
            out = pat.sub("", out, count=1)
        out = out.lstrip()
        if out == before:
            break
    if _REFUSAL_PREFIX.match(out):
        return ""
    return out


# ---------- format validation ------------------------------------------

_DIGIT = re.compile(r"\d")
_MATH_KEYWORDS = re.compile(r"(λύση|βήμα)", re.IGNORECASE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TUTORIAL_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_TUTORIAL_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)


def _has_table(text: str) -> bool:
    run = 0
    for line in text.splitlines():
        if _TABLE_ROW.match(line):
            run += 1
            if run >= 3:
                return True
        else:
            run = 0
    return False


def validate_format(text: str, prompt_name: str) -> bool:
    """Per-prompt structural sanity."""
    if prompt_name == "faq":
        if text.count("?") + text.count(";") < 3:
            return False
        # FAQ should be multi-paragraph, not one wall of prose.
        return text.count("\n") >= 4
    if prompt_name == "math":
        return bool(_DIGIT.search(text)) and bool(_MATH_KEYWORDS.search(text))
    if prompt_name == "table":
        return _has_table(text)
    if prompt_name == "tutorial":
        if len(text) <= 300:
            return False
        has_numbered = _TUTORIAL_NUMBERED.search(text) is not None
        has_bullets = len(_TUTORIAL_BULLET.findall(text)) >= 3
        return has_numbered or has_bullets
    return False


# ---------- length sanity ----------------------------------------------

def length_ok(text: str, settings: Settings) -> bool:
    return settings.min_output_chars <= len(text) <= settings.max_output_chars


# ---------- near-duplicate filtering -----------------------------------

_TOKEN = re.compile(r"\p{L}+\p{M}*|\p{N}+")
_SHINGLE_SIZE = 5


def _shingles(text: str) -> set[bytes]:
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < _SHINGLE_SIZE:
        # Fallback: treat the whole token sequence as one shingle.
        return {" ".join(tokens).encode("utf-8")} if tokens else set()
    return {
        " ".join(tokens[i : i + _SHINGLE_SIZE]).encode("utf-8")
        for i in range(len(tokens) - _SHINGLE_SIZE + 1)
    }


class MinHashDeduper:
    """MinHash LSH near-duplicate filter, scoped to a single prompt.

    Use one instance per prompt name; share across the whole run so dedup
    is global within that prompt's output.
    """

    def __init__(self, threshold: float, num_perm: int) -> None:
        self._num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._next_id = 0

    def _minhash(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self._num_perm)
        for shingle in _shingles(text):
            mh.update(shingle)
        return mh

    def add_and_check(self, text: str) -> bool:
        """Return True if this text is novel (and add it). False if near-duplicate."""
        mh = self._minhash(text)
        if self._lsh.query(mh):
            return False
        key = f"d{self._next_id}"
        self._next_id += 1
        self._lsh.insert(key, mh)
        return True


# ---------- orchestration ---------------------------------------------------

@dataclass
class FilterStats:
    """Drop counters, printed at shutdown so we can compare against guide §7 expectations."""
    seen: int = 0
    dropped_lang: int = 0
    dropped_format: int = 0
    dropped_length: int = 0
    dropped_dup: int = 0
    kept: int = 0
    lid_confidences: list[float] = field(default_factory=list)


def apply_all(
    record: Record,
    *,
    lid_model,
    deduper: MinHashDeduper,
    stats: FilterStats,
    settings: Settings,
) -> Record | None:
    """Run the full filter chain. Mutates ``record`` (adds ``language_confidence``)
    and ``stats``. Returns the surviving record or ``None``."""
    stats.seen += 1
    text = record.text or ""

    # 1. Language ID — gate before any string surgery so we judge raw output.
    passes_lang, confidence = is_greek(text, lid_model, settings.lid_threshold)
    if not passes_lang:
        stats.dropped_lang += 1
        return None

    # 2. Preamble strip — mutates the text we'll save.
    text = strip_preamble(text)

    # 3. Length sanity (before format so we don't run regex on empty / huge strings).
    if not length_ok(text, settings):
        stats.dropped_length += 1
        return None

    # 4. Format validation.
    prompt_name = record.prompt or ""
    if not validate_format(text, prompt_name):
        stats.dropped_format += 1
        return None

    # 5. Dedup against everything kept so far for this prompt.
    if not deduper.add_and_check(text):
        stats.dropped_dup += 1
        return None

    record.text = text
    record.language_confidence = confidence
    stats.lid_confidences.append(confidence)
    stats.kept += 1
    return record
