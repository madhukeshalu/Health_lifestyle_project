# utils/chart_utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def create_dashboard_charts(df):
    """Create charts for insights dashboard from DataFrame"""
    
    if df.empty:
        return {
            'risk_distribution': {'labels': [], 'values': []},
            'bmi_distribution': {'labels': [], 'values': []},
            'age_distribution': {'labels': [], 'values': []},
            'wellness_trend': {'dates': [], 'values': []}
        }
    
    charts = {}
    
    # 1. Risk Distribution
    risk_counts = df['risk_category'].value_counts()
    charts['risk_distribution'] = {
        'labels': risk_counts.index.tolist(),
        'data_values': [int(x) for x in risk_counts.values],
        'colors': ['#28a745', '#ffc107', '#dc3545']
    }
    
    # 2. BMI Distribution
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    bmi_categories = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, include_lowest=True)
    bmi_counts = bmi_categories.value_counts()
    
    charts['bmi_distribution'] = {
        'labels': bmi_labels,
        'data_values': [int(bmi_counts.get(label, 0)) for label in bmi_labels],
        'colors': ['#17a2b8', '#28a745', '#ffc107', '#dc3545']
    }
    
    # 3. Age Distribution
    age_bins = [18, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    age_categories = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    age_counts = age_categories.value_counts()
    
    charts['age_distribution'] = {
        'labels': age_labels,
        'data_values': [int(age_counts.get(label, 0)) for label in age_labels],
        'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
    }
    
    # 4. Wellness Trend (last 7 days if available)
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        recent_days = df.groupby('date')['wellness_score'].mean().tail(7)
        charts['wellness_trend'] = {
            'dates': recent_days.index.astype(str).tolist(),
            'data_values': [float(x) for x in recent_days.values]
        }
    else:
        # Generate sample trend if no timestamp
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
        values = np.random.uniform(60, 80, 7).tolist()
        charts['wellness_trend'] = {
            'dates': dates,
            'data_values': values
        }
    
    # 5. Gender Distribution
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        charts['gender_distribution'] = {
            'labels': gender_counts.index.tolist(),
            'data_values': [int(x) for x in gender_counts.values],
            'colors': ['#36A2EB', '#FF6384', '#4BC0C0']
        }
    
    return charts

def create_user_charts(result, form_data=None):
    """Create charts for individual user results"""
    if form_data is None:
        form_data = {}

    charts = {}
    
    # 1. Risk Factors Distribution
    # Categorize the identified risk factors
    risk_categories = {
        'BMI Related': 0, 
        'Lifestyle': 0, 
        'Medical': 0, 
        'Habits': 0,
        'Stress': 0,
        'Nutrition': 0
    }
    
    for factor in result.get('risk_factors', []):
        factor_lower = factor.lower()
        if 'bmi' in factor_lower or 'weight' in factor_lower or 'obese' in factor_lower:
            risk_categories['BMI Related'] += 1
        elif 'stress' in factor_lower or 'mental' in factor_lower:
            risk_categories['Stress'] += 1
        elif 'diet' in factor_lower or 'food' in factor_lower or 'water' in factor_lower:
            risk_categories['Nutrition'] += 1
        elif 'sleep' in factor_lower or 'activity' in factor_lower:
            risk_categories['Lifestyle'] += 1
        elif 'chronic' in factor_lower or 'family' in factor_lower:
            risk_categories['Medical'] += 1
        elif 'smoking' in factor_lower or 'alcohol' in factor_lower or 'screen' in factor_lower:
            risk_categories['Habits'] += 1
        else:
            risk_categories['Lifestyle'] += 1

    # Ensure at least some data shows if empty
    if sum(risk_categories.values()) == 0:
        risk_categories['Lifestyle'] = 1

    charts['risk_factors'] = {
        'labels': list(risk_categories.keys()),
        'data_values': list(risk_categories.values()),
        'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
    }
    
    # 2. Health Metrics Radar
    # Calculate scores based on form_data
    
    # Nutrition Score
    diet_map = {'Poor': 30, 'Average': 60, 'Good': 90, '1': 30, '2': 60, '3': 90}
    diet_score = diet_map.get(str(form_data.get('diet_quality', 'Average')), 60)
    water_score = min(100, (float(form_data.get('water_intake_liters', 2)) / 3) * 100)
    fast_food_score = max(0, 100 - (float(form_data.get('fast_food_frequency_per_week', 2)) * 10))
    nutrition_score = (diet_score * 0.5) + (water_score * 0.3) + (fast_food_score * 0.2)
    
    # Exercise Score
    activity = float(form_data.get('physical_activity_hours_per_week', 3))
    exercise_score = min(100, (activity / 5) * 100)
    
    # Sleep Score
    sleep = float(form_data.get('sleep_hours_per_day', 7))
    sleep_score = 100 if 7 <= sleep <= 9 else max(0, 100 - abs(8 - sleep) * 20)
    
    # Stress Score (Inverse: low stress = high score)
    stress_map = {'Low': 90, 'Medium': 60, 'High': 30, '1': 90, '2': 60, '3': 30}
    stress_level_score = stress_map.get(str(form_data.get('stress_level', 'Medium')), 60)
    wellbeing = float(form_data.get('mental_wellbeing_score', 70))
    stress_score = (stress_level_score * 0.6) + (wellbeing * 0.4)
    
    # Habits Score
    smoke_map = {'Non-smoker': 100, 'Occasional': 50, 'Regular': 10, '0': 100, '1': 50, '2': 10}
    smoke_score = smoke_map.get(str(form_data.get('smoking_status', 'Non-smoker')), 100)
    alcohol_map = {'Moderate': 80, 'High': 30, '1': 80, '2': 30} # Low/None assumed filtered or Moderate
    alcohol_score = alcohol_map.get(str(form_data.get('alcohol_consumption', 'Moderate')), 80)
    screen = float(form_data.get('daily_screen_time_hours', 6))
    screen_score = max(0, 100 - max(0, screen - 4) * 10)
    habits_score = (smoke_score * 0.4) + (alcohol_score * 0.3) + (screen_score * 0.3)
    
    # Medical Score
    chronic = 0 if str(form_data.get('chronic_conditions', 'None')).lower() in ['none', 'no', '0'] else 1
    family = 1 if str(form_data.get('family_history', 'No')).lower() in ['yes', '1'] else 0
    medical_score = 100 - (chronic * 30) - (family * 20)
    
    charts['health_metrics'] = {
        'labels': ['Nutrition', 'Exercise', 'Sleep', 'Stress', 'Habits', 'Medical'],
        'user_scores': [
            round(nutrition_score), 
            round(exercise_score), 
            round(sleep_score), 
            round(stress_score), 
            round(habits_score), 
            round(medical_score)
        ],
        'target_scores': [80, 80, 80, 80, 80, 80]
    }
    
    # 3. BMI vs Wellness
    charts['bmi_wellness'] = {
        'bmi': result.get('bmi', 25),
        'wellness': result.get('wellness_score', 70),
        'bmi_category': result.get('bmi_category', 'Normal'),
        'wellness_category': result.get('wellness_category', 'Good')
    }
    
    # 4. Risk Probability
    risk_prob = result.get('high_risk_probability', 0.15) * 100
    charts['risk_probability'] = {
        'value': risk_prob,
        'remaining': 100 - risk_prob,
        'category': result.get('final_risk_category', 'Low Risk')
    }
    
    return charts

def generate_sample_charts():
    """Generate sample charts for demo when no data exists"""
    
    # Sample risk distribution
    risk_dist = {
        'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
        'data_values': [65, 25, 10],
        'colors': ['#28a745', '#ffc107', '#dc3545']
    }
    
    # Sample BMI distribution
    bmi_dist = {
        'labels': ['Underweight', 'Normal', 'Overweight', 'Obese'],
        'data_values': [5, 55, 30, 10],
        'colors': ['#17a2b8', '#28a745', '#ffc107', '#dc3545']
    }
    
    # Sample age distribution
    age_dist = {
        'labels': ['18-25', '26-35', '36-45', '46-55', '56+'],
        'data_values': [20, 35, 25, 15, 5],
        'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
    }
    
    # Sample wellness trend
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    wellness_trend = {
        'dates': dates,
        'data_values': [65, 68, 70, 72, 75, 73, 76]
    }
    
    return {
        'risk_distribution': risk_dist,
        'bmi_distribution': bmi_dist,
        'age_distribution': age_dist,
        'wellness_trend': wellness_trend
    }

# Helper function to convert to Chart.js format
def to_chartjs_format(chart_data):
    """Convert chart data to Chart.js compatible format"""
    if 'labels' in chart_data and 'data_values' in chart_data:
        return {
            'labels': chart_data['labels'],
            'datasets': [{
                'data': chart_data['data_values'],
                'backgroundColor': chart_data.get('colors', ['#667eea']),
                'borderWidth': 2
            }]
        }
    return chart_data