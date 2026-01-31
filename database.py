"""
Database module for Health Risk Assessment Dashboard
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

# Prediction model
class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # User data
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    height_cm = db.Column(db.Float, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)
    
    # Lifestyle factors
    physical_activity_hours = db.Column(db.Float, nullable=False)
    diet_quality = db.Column(db.String(20), nullable=False)
    sleep_hours = db.Column(db.Float, nullable=False)
    stress_level = db.Column(db.String(20), nullable=False)
    smoking_status = db.Column(db.String(30), nullable=False)
    alcohol_consumption = db.Column(db.String(20), nullable=False)
    screen_time_hours = db.Column(db.Float, nullable=False)
    water_intake_liters = db.Column(db.Float, nullable=False)
    fast_food_frequency = db.Column(db.Integer, nullable=False)
    mental_wellbeing_score = db.Column(db.Integer, nullable=True)
    chronic_conditions = db.Column(db.String(100), nullable=True)
    family_history = db.Column(db.String(3), nullable=False)
    
    # Prediction results
    bmi = db.Column(db.Float, nullable=False)
    bmi_category = db.Column(db.String(20), nullable=False)
    wellness_score = db.Column(db.Float, nullable=False)
    wellness_category = db.Column(db.String(20), nullable=False)
    model_risk_category = db.Column(db.String(20), nullable=False)
    final_risk_category = db.Column(db.String(20), nullable=False)
    health_status = db.Column(db.String(30), nullable=False)
    high_risk_probability = db.Column(db.Float, nullable=False)
    critical_factors_count = db.Column(db.Integer, nullable=False)
    
    # Additional data
    risk_factors_json = db.Column(db.Text, nullable=True)
    priority_focus = db.Column(db.Text, nullable=True)
    health_suggestion = db.Column(db.Text, nullable=True)
    
    # For insights
    risk_score = db.Column(db.Integer, nullable=False)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.final_risk_category}>'

# KPI model
class KPIMetric(db.Model):
    __tablename__ = 'kpi_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    metric_date = db.Column(db.Date, default=datetime.utcnow().date, index=True)
    metric_name = db.Column(db.String(50), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    metric_type = db.Column(db.String(20), nullable=False)
    
    # Daily KPIs
    total_assessments = db.Column(db.Integer, default=0)
    avg_bmi = db.Column(db.Float, default=0)
    avg_wellness = db.Column(db.Float, default=0)
    high_risk_percentage = db.Column(db.Float, default=0)
    medium_risk_percentage = db.Column(db.Float, default=0)
    low_risk_percentage = db.Column(db.Float, default=0)
    
    def __repr__(self):
        return f'<KPIMetric {self.metric_date} - {self.metric_name}>'

# Audit log model
class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    action = db.Column(db.String(100), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    details = db.Column(db.Text, nullable=True)
    
    def __repr__(self):
        return f'<AuditLog {self.timestamp} - {self.action}>'