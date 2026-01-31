"""
Utility functions for the Health Risk Dashboard
"""
import os
import logging
from datetime import datetime
from fpdf import FPDF
import pandas as pd

def setup_logging():
    """Configure logging for the application"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'health_dashboard_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_pdf_report(result, patient_name, doctor_name, include_details=True, prediction_id=None):
    """Create a PDF health report"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Health Risk Assessment Report', 0, 1, 'C')
        pdf.ln(5)
        
        # Header information
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Patient: {patient_name}', 0, 1)
        if doctor_name:
            pdf.cell(0, 8, f'Doctor/Clinician: {doctor_name}', 0, 1)
        pdf.cell(0, 8, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        if prediction_id:
            pdf.cell(0, 8, f'Assessment ID: {prediction_id}', 0, 1)
        pdf.ln(10)
        
        # Assessment Results
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Assessment Results', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        results = [
            f'Overall Risk Category: {result["final_risk_category"]}',
            f'Health Status: {result["health_status"]}',
            f'BMI: {result["bmi"]} ({result["bmi_category"]})',
            f'Wellness Score: {result["wellness_score"]}/100 ({result["wellness_category"]})',
            f'High Risk Probability: {result["high_risk_probability"]*100:.1f}%',
            f'Critical Factors Identified: {result["critical_factors_count"]}'
        ]
        
        for res in results:
            pdf.cell(0, 8, res, 0, 1)
        
        pdf.ln(10)
        
        # Risk Factors
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Identified Risk Factors ({len(result["risk_factors"])})', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        for i, factor in enumerate(result['risk_factors'][:10], 1):
            pdf.cell(0, 8, f'{i}. {factor}', 0, 1)
        
        pdf.ln(10)
        
        # Recommendations
        if include_details:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Health Recommendations', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            # Split suggestions into lines
            suggestions = result['health_suggestion'].split('\n')
            for suggestion in suggestions[:15]:
                if suggestion.strip():
                    pdf.multi_cell(0, 8, suggestion.strip())
            
            pdf.ln(10)
        
        # Priority Focus
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Priority Focus', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8, result['priority_focus'])
        pdf.ln(10)
        
        # Disclaimer
        pdf.set_font('Arial', 'I', 10)
        pdf.multi_cell(0, 8, 
            'Disclaimer: This report is for informational purposes only and does not constitute medical advice. '
            'Please consult with a healthcare professional for medical concerns.'
        )
        
        # Save PDF
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f'health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        filepath = os.path.join(reports_dir, filename)
        pdf.output(filepath)
        
        return filepath
        
    except Exception as e:
        logging.error(f"Error creating PDF report: {str(e)}")
        raise

def export_to_csv(predictions):
    """Export predictions to CSV format"""
    try:
        # Create data dictionary
        data = []
        for pred in predictions:
            data.append({
                'ID': pred.id,
                'Timestamp': pred.timestamp.isoformat() if pred.timestamp else '',
                'Age': pred.age,
                'Gender': pred.gender,
                'Height (cm)': pred.height_cm,
                'Weight (kg)': pred.weight_kg,
                'BMI': pred.bmi,
                'BMI Category': pred.bmi_category,
                'Wellness Score': pred.wellness_score,
                'Wellness Category': pred.wellness_category,
                'Final Risk Category': pred.final_risk_category,
                'Health Status': pred.health_status,
                'High Risk Probability': pred.high_risk_probability,
                'Physical Activity (hrs/week)': pred.physical_activity_hours,
                'Diet Quality': pred.diet_quality,
                'Sleep (hrs/day)': pred.sleep_hours,
                'Stress Level': pred.stress_level,
                'Smoking Status': pred.smoking_status,
                'Alcohol Consumption': pred.alcohol_consumption,
                'Screen Time (hrs/day)': pred.screen_time_hours,
                'Water Intake (L/day)': pred.water_intake_liters,
                'Fast Food Frequency': pred.fast_food_frequency,
                'Mental Wellbeing Score': pred.mental_wellbeing_score,
                'Chronic Conditions': pred.chronic_conditions,
                'Family History': pred.family_history,
                'Critical Factors Count': pred.critical_factors_count
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        export_dir = 'reports/exports'
        os.makedirs(export_dir, exist_ok=True)
        
        filename = f'health_data_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        filepath = os.path.join(export_dir, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
        
    except Exception as e:
        logging.error(f"Error exporting to CSV: {str(e)}")
        raise