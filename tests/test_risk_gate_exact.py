from __future__ import annotations

import pytest

from control_evidence.pipeline import _clopper_pearson_upper


def test_exact_upper_bound_matches_zero_error_closed_form():
    expected = 1.0 - 0.05 ** (1.0 / 60.0)
    assert _clopper_pearson_upper(0, 60) == pytest.approx(expected, abs=1e-12)


def test_exact_upper_bound_is_monotone_and_conservative_for_nonzero_errors():
    zero = _clopper_pearson_upper(0, 60)
    one = _clopper_pearson_upper(1, 60)
    two = _clopper_pearson_upper(2, 60)
    assert 0 < zero < one < two < 1
    assert one > 1 / 60


def test_exact_upper_bound_validates_inputs():
    assert _clopper_pearson_upper(0, 0) == 1.0
    assert _clopper_pearson_upper(5, 5) == 1.0
    with pytest.raises(ValueError):
        _clopper_pearson_upper(-1, 5)
    with pytest.raises(ValueError):
        _clopper_pearson_upper(6, 5)
