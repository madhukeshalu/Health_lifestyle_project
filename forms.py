"""
Flask forms for user input
"""
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField, TextAreaField, RadioField
from wtforms.validators import DataRequired, NumberRange, Length, Optional

class HealthAssessmentForm(FlaskForm):
    """Form for health risk assessment"""
    
    # Personal Information
    age = IntegerField('Age', validators=[
        DataRequired(message="Age is required"),
        NumberRange(min=18, max=100, message="Age must be between 18 and 100")
    ])
    
    gender = SelectField('Gender', choices=[
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other')
    ], validators=[DataRequired()])
    
    height_cm = FloatField('Height (cm)', validators=[
        DataRequired(message="Height is required"),
        NumberRange(min=100, max=250, message="Height must be between 100cm and 250cm")
    ])
    
    weight_kg = FloatField('Weight (kg)', validators=[
        DataRequired(message="Weight is required"),
        NumberRange(min=30, max=300, message="Weight must be between 30kg and 300kg")
    ])
    
    # Lifestyle Factors
    physical_activity_hours_per_week = FloatField(
        'Physical Activity (hours/week)', 
        validators=[
            DataRequired(),
            NumberRange(min=0, max=168, message="Hours must be between 0 and 168")
        ]
    )
    
    diet_quality = SelectField('Diet Quality', choices=[
        ('Poor', 'Poor'),
        ('Average', 'Average'), 
        ('Good', 'Good')
    ], validators=[DataRequired()])
    
    sleep_hours_per_day = FloatField('Sleep (hours/day)', validators=[
        DataRequired(),
        NumberRange(min=0, max=24, message="Hours must be between 0 and 24")
    ])
    
    stress_level = SelectField('Stress Level', choices=[
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High')
    ], validators=[DataRequired()])
    
    smoking_status = SelectField('Smoking Status', choices=[
        ('Non-smoker', 'Non-smoker'),
        ('Occasional', 'Occasional'),
        ('Regular', 'Regular')
    ], validators=[DataRequired()])
    
    alcohol_consumption = SelectField('Alcohol Consumption', choices=[
        ('Moderate', 'Moderate'),
        ('High', 'High')
    ], validators=[DataRequired()])
    
    daily_screen_time_hours = FloatField('Daily Screen Time (hours)', validators=[
        DataRequired(),
        NumberRange(min=0, max=24, message="Hours must be between 0 and 24")
    ])
    
    water_intake_liters = FloatField('Water Intake (liters/day)', validators=[
        DataRequired(),
        NumberRange(min=0, max=10, message="Liters must be between 0 and 10")
    ])
    
    fast_food_frequency_per_week = IntegerField('Fast Food (times/week)', validators=[
        DataRequired(),
        NumberRange(min=0, max=21, message="Frequency must be between 0 and 21")
    ])
    
    mental_wellbeing_score = IntegerField('Mental Wellbeing Score (0-100)', validators=[
        Optional(),
        NumberRange(min=0, max=100, message="Score must be between 0 and 100")
    ], default=50)
    
    chronic_conditions = SelectField('Chronic Conditions', choices=[
        ('None', 'None'),
        ('Diabetes', 'Diabetes'),
        ('Hypertension', 'Hypertension'),
        ('Asthma', 'Asthma'),
        ('Heart Disease', 'Heart Disease'),
        ('Other', 'Other')
    ], validators=[DataRequired()])
    
    family_history = RadioField('Family History of Disease', choices=[
        ('Yes', 'Yes'),
        ('No', 'No')
    ], validators=[DataRequired()])

class FilterForm(FlaskForm):
    """Form for filtering predictions in insights page"""
    risk_category = SelectField('Risk Category', choices=[
        ('', 'All Categories'),
        ('High Risk', 'High Risk'),
        ('Medium Risk', 'Medium Risk'),
        ('Low Risk', 'Low Risk')
    ], validators=[Optional()])
    
    date_from = StringField('From Date', validators=[Optional()])
    date_to = StringField('To Date', validators=[Optional()])
    
    age_min = IntegerField('Min Age', validators=[
        Optional(),
        NumberRange(min=18, max=100)
    ])
    
    age_max = IntegerField('Max Age', validators=[
        Optional(),
        NumberRange(min=18, max=100)
    ])