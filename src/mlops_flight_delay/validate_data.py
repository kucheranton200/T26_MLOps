"""Compatibility entrypoint forwarding to mlops_project.validate_data."""

import sys

from mlops_project.validate_data import calculate_sha256, main, validate_data

__all__ = ["calculate_sha256", "main", "validate_data"]

if __name__ == "__main__":
    sys.exit(main())
