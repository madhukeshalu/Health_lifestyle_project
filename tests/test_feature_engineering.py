import math
from src.components import feature_engineering as fe


def test_calculate_bmi():
    bmi = fe.calculate_bmi(80, 160)  # 80kg, 160cm -> 31.25
    assert bmi is not None
    assert math.isclose(bmi, 31.25, rel_tol=1e-3)


def test_get_risk_category_high():
    cat = fe.get_risk_category(31.25, 55)
    assert cat == "High"


def test_get_risk_category_low():
    cat = fe.get_risk_category(22.0, 80)
    assert cat == "Low"
