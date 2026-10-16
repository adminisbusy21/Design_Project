from backend.priority import calculate_priority


def test_high_priority_damage():
    result = calculate_priority("Positive", 0.95)

    assert result["priority"] == "HIGH"
    assert result["review_required"] is False


def test_medium_priority_low_confidence_damage():
    result = calculate_priority("Positive", 0.60)

    assert result["priority"] == "MEDIUM"
    assert result["review_required"] is True


def test_low_priority_no_damage():
    result = calculate_priority("Negative", 0.95)

    assert result["priority"] == "LOW"
    assert result["review_required"] is False