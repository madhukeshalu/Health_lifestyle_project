"""Business rule driven recommendations (ported from EDA/Notebook helpers)."""
from typing import Dict


def generate_recommendations(bmi: float, wellness_score: float, risk_category: str, inputs: Dict) -> Dict:
    recs = {"risk_category": risk_category, "actions": []}
    if risk_category == "High":
        recs["actions"].append("Consult a physician for comprehensive assessment")
        if bmi is not None and bmi >= 30:
            recs["actions"].append("Targeted weight management program")
    if wellness_score < 60:
        recs["actions"].append("Increase physical activity and improve diet")
    if "smoking" in inputs and str(inputs.get("smoking", "")).lower() in ("yes", "smoker", "true"):
        recs["actions"].append("Smoking cessation program")
    if not recs["actions"]:
        recs["actions"].append("Maintain healthy lifestyle")
    return recs
