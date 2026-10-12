"""Shared text normalisation helpers."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_unicode_text(text: str) -> str:
    """Combine UTF-16 surrogate pairs while preserving ordinary Unicode text.

    Some MNER records contain escaped emoji as two surrogate code points.  Fast
    tokenizer backends reject those Python strings, even though the pair denotes
    a valid Unicode character.  A UTF-16LE round trip combines valid pairs and
    replaces any genuinely unpaired surrogate with the Unicode replacement
    character.
    """
    return text.encode("utf-16-le", errors="surrogatepass").decode(
        "utf-16-le",
        errors="replace",
    )


def normalize_words(words: Iterable[str]) -> list[str]:
    """Return words with malformed surrogate representations normalised."""
    return [normalize_unicode_text(str(word)) for word in words]
