# Greyquill customer-record pair scoring

A dependency-free implementation of the `score_pair` function from the internship task.

## Run the example

With Python 3.10 or newer, run this single command from this directory:

```bash
python demo.py
```

The demo uses two synthetic customer records embedded in `demo.py`; it does not require the full CSV or any third-party packages. No customer dataset or real customer records are included in this repository.

Run the automated tests with:

```bash
python -m unittest -v
```

## Scoring approach

- **Missing values:** if either side is blank or missing, that field is omitted from `field_scores` and from the weighted average. The remaining weights are re-normalized.
- **phonetic (name):** normalize case, accents and punctuation; split names into tokens; compare tokens independent of order; and find the best one-to-one token assignment. Exact token matches score 1.0, a matching initial scores 0.88, Soundex-equivalent tokens receive a phonetic floor of 0.86, and other tokens use `SequenceMatcher` character similarity. Unmatched tokens lower the overall name score.
- **edit_distance (email):** normalize case and punctuation, then use normalized Levenshtein similarity: `1 - edit_distance / max_length`.
- **normalized_phone:** remove non-digits, remove an initial country code `91` and a leading trunk `0`, then compare the remaining digits exactly.
- **date:** parse the date representations used in the supplied CSV (ISO, slash, dot, hyphen and abbreviated/full month forms) and compare calendar dates. Two-digit years use a pivot: `00`–`26` map to 2000–2026 and `27`–`99` map to 1927–1999.
- **exact:** trim surrounding whitespace and compare case-insensitively.

The `algorithm` field lists the comparison methods actually used. `field_scores` includes only fields with evidence on both sides.

## Dependencies

Python 3.10 or newer. No third-party libraries are required.
