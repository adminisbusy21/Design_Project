
import pytest

from backend.priority import calculate_priority


def test_high_confidence_crack_gets_highest_priority():
    result = calculate_priority("Crack", 0.95)
    assert result["priority"] == "P1"
    assert result["review_required"] is False


def test_high_confidence_rust_corrosion_gets_high_priority():
    result = calculate_priority("Rust_Corrosion", 0.95)
    assert result["priority"] == "P2"
    assert result["review_required"] is False


def test_high_confidence_road_pothole_gets_priority():
    result = calculate_priority("Road_Pothole", 0.95)
    assert result["priority"] == "P3"
    assert result["review_required"] is False


def test_high_confidence_surface_erosion_gets_priority():
    result = calculate_priority("Road_Surface_Erosion", 0.95)
    assert result["priority"] == "P4"
    assert result["review_required"] is False


def test_high_confidence_non_crack_gets_low_priority():
    result = calculate_priority("Non-Crack", 0.95)
    assert result["priority"] == "P10"
    assert result["review_required"] is False


def test_medium_confidence_damage_gets_lower_priority():
    result = calculate_priority("Crack", 0.90)
    assert result["priority"] == "P2"
    assert result["review_required"] is False


def test_low_confidence_damage_requires_review():
    result = calculate_priority("Crack", 0.60)
    assert result["priority"] == "P5"
    assert result["review_required"] is True


def test_low_confidence_non_crack_requires_review():
    result = calculate_priority("Non-Crack", 0.60)
    assert result["priority"] == "P5"
    assert result["review_required"] is True


def test_invalid_prediction_is_rejected():
    with pytest.raises(ValueError):
        calculate_priority("Positive", 0.95)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        calculate_priority("Crack", 1.1)
