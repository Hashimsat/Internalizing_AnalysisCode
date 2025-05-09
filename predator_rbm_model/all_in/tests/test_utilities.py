from all_in import safe_div


def test_safe_div():
    """This function tests the safe_div function."""

    # Division by zero (int), so return 0
    assert safe_div(1, 0) == 0

    # Division by 2, so return regular result
    assert safe_div(1, 2) == 0.5

    # Division by 2, numerator is 0, expect 0
    assert safe_div(0, 2) == 0

    # Division by 2, numerator is 0.0 (float), expect 0
    assert safe_div(1, 0.0) == 0
