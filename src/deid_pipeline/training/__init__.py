"""Training and dataset tooling (dev-only).

This package is intentionally separated from runtime code paths:
- Runtime de-identification must be local-only and should not require dataset tooling.
- Training/data preparation may require network access and large dependencies.
"""

