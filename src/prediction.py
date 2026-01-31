# =====================================================
# IMPORTS
# =====================================================
import os
import sys
import joblib
import numpy as np
import pandas as pd

# =====================================================
# LOAD TRAINED ARTIFACTS
# =====================================================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_file_dir)
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "health_risk_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {str(e)}")
    raise

# =====================================================
# HEALTH THRESHOLDS
# =====================================================
HIGH_RISK_THRESHOLD = 0.25

# =====================================================
# RAW → NUMERIC ENCODING
# =====================================================
def encode_raw_inputs(user_input: dict) -> dict:
    user_input["gender"] = str(user_input["gender"]).lower().strip()
    user_input["gender"] = 1 if user_input["gender"] == "male" else 0

    diet_map = {"poor": 1, "average": 2, "good": 3}
    user_input["diet_quality"] = str(user_input["diet_quality"]).lower().strip()
    user_input["diet_quality"] = diet_map.get(user_input["diet_quality"], 2)

    stress_map = {"low": 1, "medium": 2, "high": 3}
    user_input["stress_level"] = str(user_input["stress_level"]).lower().strip()
    user_input["stress_level"] = stress_map.get(user_input["stress_level"], 2)

    smoke_map = {"non-smoker": 0, "occasional": 1, "regular": 2}
    user_input["smoking_status"] = str(user_input["smoking_status"]).lower().strip()
    user_input["smoking_status"] = smoke_map.get(user_input["smoking_status"], 0)

    alcohol_map = {"moderate": 1, "high": 2}
    user_input["alcohol_consumption"] = str(user_input["alcohol_consumption"]).lower().strip()
    user_input["alcohol_consumption"] = alcohol_map.get(user_input["alcohol_consumption"], 1)

    user_input["family_history"] = str(user_input["family_history"]).lower().strip()
    user_input["family_history"] = 1 if user_input["family_history"] == "yes" else 0

    user_input["chronic_conditions"] = str(user_input["chronic_conditions"]).lower().strip()
    user_input["chronic_conditions"] = (
        0 if user_input["chronic_conditions"] in ["none", "nan"] else 1
    )

    return user_input

# =====================================================
# DERIVED FEATURES
# =====================================================
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_wellness_score(inputs: dict) -> float:
    score = 100

    score -= inputs["stress_level"] * 5
    score -= max(0, 8 - inputs["sleep_hours_per_day"]) * 4
    score -= inputs["smoking_status"] * 5
    score -= inputs["alcohol_consumption"] * 5
    score -= inputs["daily_screen_time_hours"] * 2
    score -= inputs["fast_food_frequency_per_week"] * 3

    score += inputs["physical_activity_hours_per_week"] * 2
    score += inputs["diet_quality"] * 3
    score += inputs["water_intake_liters"] * 2

    if "mental_wellbeing_score" in inputs:
        score += inputs["mental_wellbeing_score"] / 10

    return round(max(0, min(score, 100)), 1)

def get_wellness_category(score: float) -> str:
    if score < 40:
        return "Poor"
    elif score < 60:
        return "Fair"
    elif score < 80:
        return "Good"
    else:
        return "Excellent"

# =====================================================
# BUSINESS RULE ENGINE (UPDATED)
# =====================================================
def generate_recommendations(bmi: float, wellness_score: float, risk_category: str, user_input: dict) -> dict:
    """Generate recommendations based on specific lifestyle factors"""
    
    bmi_category = get_bmi_category(bmi)
    wellness_category = get_wellness_category(wellness_score)
    
    # Collect specific lifestyle risk factors
    risk_factors = []
    
    # BMI-related factors
    if bmi_category == "Obese":
        risk_factors.append("Obesity (BMI ≥ 30)")
    elif bmi_category == "Overweight":
        risk_factors.append("Overweight (BMI 25-30)")
    elif bmi_category == "Underweight":
        risk_factors.append("Underweight (BMI < 18.5)")
    
    # Wellness-related factors
    if wellness_category == "Poor":
        risk_factors.append("Low wellness score (< 40)")
    elif wellness_category == "Fair":
        risk_factors.append("Below average wellness score (40-59)")
    
    # Specific lifestyle factors
    if user_input.get("physical_activity_hours_per_week", 0) < 2.5:
        risk_factors.append("Insufficient physical activity")
    
    if user_input.get("diet_quality", 2) == 1:  # Poor diet
        risk_factors.append("Poor diet quality")
    elif user_input.get("diet_quality", 2) == 2:  # Average diet
        risk_factors.append("Average diet (needs improvement)")
    
    if user_input.get("sleep_hours_per_day", 8) < 6:
        risk_factors.append("Inadequate sleep (< 6 hours)")
    elif user_input.get("sleep_hours_per_day", 8) < 7:
        risk_factors.append("Suboptimal sleep (6-7 hours)")
    
    if user_input.get("stress_level", 1) == 3:  # High stress
        risk_factors.append("High stress level")
    elif user_input.get("stress_level", 1) == 2:  # Medium stress
        risk_factors.append("Moderate stress level")
    
    if user_input.get("smoking_status", 0) == 2:  # Regular smoker
        risk_factors.append("Regular smoking")
    elif user_input.get("smoking_status", 0) == 1:  # Occasional smoker
        risk_factors.append("Occasional smoking")
    
    if user_input.get("alcohol_consumption", 1) == 2:  # High alcohol
        risk_factors.append("High alcohol consumption")
    
    if user_input.get("daily_screen_time_hours", 0) > 8:
        risk_factors.append("Excessive screen time (> 8 hours)")
    elif user_input.get("daily_screen_time_hours", 0) > 6:
        risk_factors.append("High screen time (6-8 hours)")
    
    if user_input.get("water_intake_liters", 0) < 1.5:
        risk_factors.append("Inadequate water intake")
    
    if user_input.get("fast_food_frequency_per_week", 0) > 3:
        risk_factors.append("High fast food consumption")
    
    if user_input.get("chronic_conditions", 0) == 1:
        risk_factors.append("Chronic health condition(s)")
    
    if user_input.get("family_history", 0) == 1:
        risk_factors.append("Family history of disease")
    
    # Health status based on risk category
    if risk_category == "High Risk":
        health_status = "Poor"
    elif risk_category == "Medium Risk":
        health_status = "Needs Improvement"
    else:  # Low Risk
        health_status = "Good" if wellness_score >= 60 else "Fair"
    
    # Priority focus based on risk category
    if risk_category == "High Risk":
        priority_focus = "Immediate lifestyle changes and medical consultation"
    elif risk_category == "Medium Risk":
        priority_focus = "Preventive lifestyle improvements"
    else:
        priority_focus = "Maintain healthy habits with monitoring"
    
    # Risk reason - list specific factors
    if risk_factors:
        if len(risk_factors) <= 3:
            risk_reason = f"Based on: {', '.join(risk_factors)}"
        else:
            risk_reason = f"Based on {len(risk_factors)} lifestyle factors including {', '.join(risk_factors[:3])}"
    else:
        risk_reason = "No significant risk factors identified"
    
    # Specific suggestions for each risk factor
    risk_factor_suggestions = {
        # BMI-related
        "Obesity (BMI ≥ 30)": "Focus on weight loss through calorie control and regular exercise. Aim for 1-2 lbs per week loss.",
        "Overweight (BMI 25-30)": "Maintain healthy weight through balanced diet and 150+ minutes of exercise weekly.",
        "Underweight (BMI < 18.5)": "Increase calorie intake with nutrient-dense foods to reach healthy weight range.",
        
        # Wellness-related
        "Low wellness score (< 40)": "Improve overall wellness by addressing multiple lifestyle factors simultaneously.",
        "Below average wellness score (40-59)": "Focus on improving key areas like sleep, stress, and physical activity.",
        
        # Physical activity
        "Insufficient physical activity": "Aim for at least 150 minutes of moderate exercise or 75 minutes of vigorous exercise weekly.",
        
        # Diet-related
        "Poor diet quality": "Increase fruits, vegetables, whole grains. Reduce processed foods, sugar, and saturated fats.",
        "Average diet (needs improvement)": "Improve diet by adding more vegetables, lean proteins, and reducing portion sizes.",
        
        # Sleep-related
        "Inadequate sleep (< 6 hours)": "Prioritize 7-9 hours of sleep nightly. Establish consistent sleep schedule.",
        "Suboptimal sleep (6-7 hours)": "Aim for 7-9 hours of quality sleep for better health outcomes.",
        
        # Stress-related
        "High stress level": "Practice stress management: meditation, deep breathing, yoga, or regular exercise.",
        "Moderate stress level": "Incorporate daily relaxation techniques to manage stress effectively.",
        
        # Smoking-related
        "Regular smoking": "Quit smoking completely. Seek support from healthcare provider or smoking cessation programs.",
        "Occasional smoking": "Eliminate smoking completely to reduce health risks.",
        
        # Alcohol-related
        "High alcohol consumption": "Limit alcohol to moderate levels (≤1 drink/day for women, ≤2 for men).",
        
        # Screen time
        "Excessive screen time (> 8 hours)": "Reduce screen time, take regular breaks, and practice the 20-20-20 rule.",
        "High screen time (6-8 hours)": "Take frequent breaks from screens and limit recreational screen use.",
        
        # Hydration
        "Inadequate water intake": "Drink at least 2 liters of water daily. Carry a water bottle as a reminder.",
        
        # Food habits
        "High fast food consumption": "Limit fast food to once a week. Prepare healthy meals at home instead.",
        
        # Medical
        "Chronic health condition(s)": "Follow treatment plan, monitor regularly, and maintain open communication with doctor.",
        "Family history of disease": "Get regular screenings and maintain healthy lifestyle to mitigate genetic risks."
    }
    
    # Generate health suggestions based on risk factors
    if risk_factors:
        # Get unique suggestions for each risk factor
        suggestions_list = []
        for factor in risk_factors:
            if factor in risk_factor_suggestions:
                suggestions_list.append(risk_factor_suggestions[factor])
        
        # Add general recommendations based on risk category
        if risk_category == "High Risk":
            suggestions_list.append("Schedule a comprehensive health check-up with your doctor.")
            suggestions_list.append("Consider working with a nutritionist or health coach.")
        elif risk_category == "Medium Risk":
            suggestions_list.append("Monitor your progress monthly and adjust lifestyle as needed.")
            suggestions_list.append("Consider joining a wellness program or support group.")
        else:
            suggestions_list.append("Continue regular health check-ups and maintain healthy habits.")
        
        # Create numbered suggestions
        health_suggestion = "\n".join([f"{i+1}. {suggestion}" for i, suggestion in enumerate(suggestions_list[:10])])  # Limit to 10 suggestions
    else:
        health_suggestion = "1. Continue balanced diet with plenty of fruits and vegetables\n2. Maintain regular physical activity (150+ minutes/week)\n3. Get 7-9 hours of quality sleep nightly\n4. Stay hydrated (2+ liters of water daily)\n5. Schedule annual health check-ups"
    
    return {
        "health_status": health_status,
        "risk_reason": risk_reason,
        "priority_focus": priority_focus,
        "health_suggestion": health_suggestion,
        "risk_factors": risk_factors
    }

# =====================================================
# MAIN PREDICTION FUNCTION
# =====================================================
def predict_health_risk(raw_user_input: dict) -> dict:
    # STEP 1: Encode RAW inputs
    user_input = encode_raw_inputs(raw_user_input)
    
    # STEP 2: Calculate derived features
    bmi = calculate_bmi(user_input["weight_kg"], user_input["height_cm"])
    wellness_score = calculate_wellness_score(user_input)
    
    bmi_category = get_bmi_category(bmi)
    wellness_category = get_wellness_category(wellness_score)
    
    # STEP 3: Model prediction
    user_input["bmi"] = bmi
    user_input["wellness_score"] = wellness_score
    
    # Build DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Feature alignment
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_columns]
    
    # Scaling and prediction
    try:
        input_scaled = scaler.transform(input_df)
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get class indices
        class_names = list(target_encoder.classes_)
        if "High Risk" in class_names:
            high_risk_index = class_names.index("High Risk")
            high_risk_prob = probabilities[high_risk_index]
        else:
            high_risk_index = 0
            high_risk_prob = 0.0
        
        # Apply threshold
        if high_risk_prob >= HIGH_RISK_THRESHOLD:
            predicted_class = high_risk_index
        else:
            predicted_class = np.argmax(probabilities)
        
        model_risk_category = target_encoder.inverse_transform([predicted_class])[0]
        
    except Exception as e:
        print(f"Model prediction error: {e}. Using business rules.")
        # If model fails, determine risk based on business rules
        if bmi >= 30 or wellness_score <= 40:
            model_risk_category = "Medium Risk"
        else:
            model_risk_category = "Low Risk"
        high_risk_prob = 0.0
    
    # STEP 4: Apply business rule overrides
    final_risk_category = model_risk_category
    
    # Critical override rules
    critical_factors = 0
    if bmi_category == "Obese":
        critical_factors += 2
    if wellness_category == "Poor":
        critical_factors += 2
    if user_input.get("chronic_conditions", 0) == 1:
        critical_factors += 1
    if user_input.get("family_history", 0) == 1:
        critical_factors += 1
    if user_input.get("smoking_status", 0) == 2:
        critical_factors += 2
    
    # Override based on critical factors
    if critical_factors >= 4 and final_risk_category == "Low Risk":
        final_risk_category = "High Risk"
    elif critical_factors >= 2 and final_risk_category == "Low Risk":
        final_risk_category = "Medium Risk"
    
    # Override based on BMI and wellness
    if (bmi_category == "Obese" or wellness_category == "Poor") and final_risk_category == "Low Risk":
        final_risk_category = "Medium Risk"
    
    # STEP 5: Generate recommendations
    rules = generate_recommendations(bmi, wellness_score, final_risk_category, user_input)
    
    # FINAL OUTPUT
    return {
        "bmi": bmi,
        "bmi_category": bmi_category,
        "wellness_score": wellness_score,
        "wellness_category": wellness_category,
        "model_risk_category": model_risk_category,
        "final_risk_category": final_risk_category,
        "health_status": rules["health_status"],
        "risk_reason": rules["risk_reason"],
        "priority_focus": rules["priority_focus"],
        "health_suggestion": rules["health_suggestion"],
        "risk_factors": rules["risk_factors"],
        "high_risk_probability": round(float(high_risk_prob), 3),
        "critical_factors_count": critical_factors,
        "warning": "Business rules applied for accurate assessment" if final_risk_category != model_risk_category else None
    }

# =====================================================
# LOCAL TEST
# =====================================================
if __name__ == "__main__":
    sample_input = {
        "age": 34,
        "gender": "Male",
        "height_cm": 170,
        "weight_kg": 78,
        "physical_activity_hours_per_week": 2,
        "diet_quality": "Good",
        "sleep_hours_per_day": 6,
        "stress_level": "High",
        "smoking_status": "Regular",
        "alcohol_consumption": "Moderate",
        "daily_screen_time_hours": 6,
        "water_intake_liters": 2,
        "fast_food_frequency_per_week": 3,
        "mental_wellbeing_score": 45,
        "chronic_conditions": "None",
        "family_history": "Yes"
    }
    
    print("Testing prediction function...")
    result = predict_health_risk(sample_input)
    
    print("\n=== PREDICTION RESULTS ===")
    print(f"BMI: {result['bmi']} ({result['bmi_category']})")
    print(f"Wellness Score: {result['wellness_score']}/100 ({result['wellness_category']})")
    print(f"Model Risk: {result['model_risk_category']}")
    print(f"Final Risk: {result['final_risk_category']}")
    print(f"Health Status: {result['health_status']}")
    print(f"\nRisk Reason: {result['risk_reason']}")
    print(f"\nPriority Focus: {result['priority_focus']}")
    print(f"\nIdentified Risk Factors ({len(result['risk_factors'])}):")
    for i, factor in enumerate(result['risk_factors'], 1):
        print(f"  {i}. {factor}")
    print(f"\nHigh Risk Probability: {result['high_risk_probability']*100:.1f}%")
    if result.get('warning'):
        print(f"\n⚠️  {result['warning']}")