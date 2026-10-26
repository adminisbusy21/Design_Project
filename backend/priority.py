DAMAGE_CLASSES = {
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
}


def calculate_priority(prediction: str, confidence: float) -> dict:
    """
    Calculate preliminary inspection priority for the 5-class
    structural damage classifier.

    Priority is an inspection-ordering aid, not a structural
    safety assessment. Model confidence represents prediction
    certainty and must not be interpreted as physical severity.
    """

    prediction = prediction.strip()

    if prediction not in DAMAGE_CLASSES:
        raise ValueError(
            f"Unsupported prediction '{prediction}'. "
            f"Expected one of: {sorted(DAMAGE_CLASSES)}."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")

    # Low-confidence predictions require human review.
    if confidence < 0.80:
        return {
            "priority": "P5",
            "reason": (
                f"{prediction} predicted, but model confidence is below "
                "the high-confidence threshold; human review required."
            ),
            "review_required": True,
        }

    # High-confidence non-damage result.
    if prediction == "Non-Crack":
        if confidence >= 0.95:
            priority = "P10"
        elif confidence >= 0.90:
            priority = "P9"
        else:
            priority = "P8"

        return {
            "priority": priority,
            "reason": (
                "Non-Crack predicted with high model confidence; "
                "lower inspection urgency."
            ),
            "review_required": False,
        }

    # High-confidence damage classifications.
    priority_ranges = {
        "Crack": ("P1", "P2", "P3"),
        "Rust_Corrosion": ("P2", "P3", "P4"),
        "Road_Pothole": ("P3", "P4", "P5"),
        "Road_Surface_Erosion": ("P4", "P5", "P6"),
    }

    high, medium, low = priority_ranges[prediction]

    if confidence >= 0.95:
        priority = high
    elif confidence >= 0.90:
        priority = medium
    else:
        priority = low

    return {
        "priority": priority,
        "reason": (
            f"{prediction} predicted with high model confidence; "
            "inspection priority assigned according to the preliminary "
            "damage-category prioritization policy."
        ),
        "review_required": False,
    }