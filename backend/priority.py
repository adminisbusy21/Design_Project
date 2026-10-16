def calculate_priority(prediction: str, confidence: float) -> dict:
    """
    Calculate preliminary inspection priority from AI prediction confidence.

    This is an inspection-prioritization aid, not a structural safety assessment.
    """

    prediction = prediction.strip().lower()

    if prediction not in {"positive", "negative"}:
        raise ValueError("Prediction must be 'Positive' or 'Negative'.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")

    # High-confidence detected damage
    if prediction == "positive" and confidence >= 0.80:
        return {
            "priority": "HIGH",
            "reason": "Damage detected with high model confidence.",
            "review_required": False,
        }

    # Detected damage, but model is uncertain
    if prediction == "positive" and confidence < 0.80:
        return {
            "priority": "MEDIUM",
            "reason": "Damage detected, but model confidence is below the high-confidence threshold.",
            "review_required": True,
        }

    # No damage detected, but model is uncertain
    if prediction == "negative" and confidence < 0.80:
        return {
            "priority": "MEDIUM",
            "reason": "No damage detected, but model confidence is below the high-confidence threshold.",
            "review_required": True,
        }

    # High-confidence no-damage result
    return {
        "priority": "LOW",
        "reason": "No damage detected with high model confidence.",
        "review_required": False,
    }