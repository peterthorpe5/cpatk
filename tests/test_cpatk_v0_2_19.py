"""Regression tests for CPATK v0.2.19 review hardening."""

from __future__ import annotations

import inspect

import cpatk.batch as batch


def test_batch_module_does_not_expose_test_prefixed_helpers() -> None:
    """Production helpers should not be discovered as unittest test cases."""
    public_callables = [
        name
        for name, value in inspect.getmembers(batch)
        if callable(value) and not name.startswith("__")
    ]
    assert "test_metadata_association_with_pcs" not in public_callables
    assert "calculate_metadata_association_with_pcs" in public_callables
