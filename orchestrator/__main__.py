"""Enables `python -m orchestrator "query"`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
