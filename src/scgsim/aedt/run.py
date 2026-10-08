"""Formal AEDT CLI entry point delegating to the transaction owner."""

from __future__ import annotations

from .runtime.transaction import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
