"""Age-dependent steering label.

The manifest `pattern` is the class at formation. A storm is erratic at age t
when `turned_erratic_at` is set and t is at least that age. A storm born
erratic has a null flip and is erratic from age 0.
"""

from __future__ import annotations


def label_at(pattern: str, turned_erratic_at: float | None, age: float) -> str:
    if turned_erratic_at is not None and age >= float(turned_erratic_at):
        return "erratic"
    return pattern
