# utils/report_utils.py
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

def generate_report_data(form_data, result):
    """Generate comprehensive report data"""
    
    report_data = {
        'timestamp': datetime.now().strftime("%B %d, %Y %I:%M %p"),
        'user_info': {
            'age': form_data.get('age', 'N/A'),
            'gender': form_data.get('gender', 'N/A'),
            'bmi': f"{result.get('bmi', 0):.1f}",
            'bmi_category': result.get('bmi_category', 'N/A')
        },
        'assessment': {
            'risk_category': result.get('final_risk_category', 'N/A'),
            'health_status': result.get('health_status', 'N/A'),
            'wellness_score': result.get('wellness_score', 0),
            'wellness_category': result.get('wellness_category', 'N/A'),
            'high_risk_probability': f"{result.get('high_risk_probability', 0) * 100:.1f}%"
        },
        'risk_factors': result.get('risk_factors', []),
        'recommendations': result.get('health_suggestion', '').split('\n'),
        'summary': {
            'critical_factors': result.get('critical_factors_count', 0),
            'priority_focus': result.get('priority_focus', 'N/A'),
            'risk_reason': result.get('risk_reason', 'N/A')
        }
    }
    
    # Add lifestyle metrics
    report_data['lifestyle_metrics'] = {
        'physical_activity': f"{form_data.get('physical_activity_hours_per_week', 0)} hrs/week",
        'sleep_hours': f"{form_data.get('sleep_hours_per_day', 0)} hrs/night",
        'screen_time': f"{form_data.get('daily_screen_time_hours', 0)} hrs/day",
        'water_intake': f"{form_data.get('water_intake_liters', 0)} L/day",
        'fast_food': f"{form_data.get('fast_food_frequency_per_week', 0)} times/week"
    }
    
    # Add health habits
    report_data['health_habits'] = {
        'diet_quality': form_data.get('diet_quality', 'N/A'),
        'stress_level': form_data.get('stress_level', 'N/A'),
        'smoking_status': form_data.get('smoking_status', 'N/A'),
        'alcohol_consumption': form_data.get('alcohol_consumption', 'N/A')
    }
    
    # Add medical history
    report_data['medical_history'] = {
        'chronic_conditions': form_data.get('chronic_conditions', 'N/A'),
        'family_history': form_data.get('family_history', 'N/A'),
        'mental_wellbeing': form_data.get('mental_wellbeing_score', 'N/A')
    }
    
    return report_data

def format_recommendations(recommendations):
    """Format recommendations for display"""
    formatted = []
    for i, rec in enumerate(recommendations, 1):
        if rec.strip():
            formatted.append(f"{i}. {rec.strip()}")
    return formatted

def generate_pdf_report(form_data, result, filename="health_report.pdf"):
    """Generate PDF health report"""
    
    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Health Risk Assessment Report", title_style))
    
    # Date
    date_style = ParagraphStyle(
        'CustomDate',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray,
        alignment=1
    )
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", date_style))
    story.append(Spacer(1, 20))
    
    # Summary Section
    story.append(Paragraph("Assessment Summary", styles['Heading2']))
    
    # Summary table
    summary_data = [
        ["Metric", "Value", "Category"],
        ["Age", str(form_data.get('age', 'N/A')), ""],
        ["Gender", form_data.get('gender', 'N/A'), ""],
        ["BMI", f"{result.get('bmi', 0):.1f}", result.get('bmi_category', 'N/A')],
        ["Wellness Score", f"{result.get('wellness_score', 0)}/100", result.get('wellness_category', 'N/A')],
        ["Risk Category", result.get('final_risk_category', 'N/A'), result.get('health_status', 'N/A')]
    ]
    
    summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    # Risk Factors
    if result.get('risk_factors'):
        story.append(Paragraph("Identified Risk Factors", styles['Heading2']))
        for factor in result['risk_factors']:
            story.append(Paragraph(f"• {factor}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Health Recommendations", styles['Heading2']))
    
    recommendations = result.get('health_suggestion', '').split('\n')
    for rec in recommendations:
        if rec.strip():
            story.append(Paragraph(f"• {rec.strip()}", styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    # Disclaimer
    story.append(Paragraph("Disclaimer", styles['Heading3']))
    disclaimer_text = """
    This report is for informational purposes only and is not a substitute for 
    professional medical advice, diagnosis, or treatment. Always seek the advice 
    of your physician or other qualified health provider with any questions you 
    may have regarding a medical condition.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    return filename