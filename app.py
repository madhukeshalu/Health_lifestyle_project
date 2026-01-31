# app.py - Complete Flask Application for Health Risk Dashboard
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from functools import wraps
import io

# Try to import utils with fallbacks
try:
    from utils.chart_utils import create_dashboard_charts, create_user_charts
except ImportError:
    print("Warning: Could not import chart_utils. Using fallback functions.")
    
    # Fallback functions
    def create_dashboard_charts(df):
        """Fallback function when chart_utils is not available"""
        return {
            'risk_distribution': {'labels': ['Low Risk', 'Medium Risk', 'High Risk'], 'values': [60, 30, 10], 'colors': ['#28a745', '#ffc107', '#dc3545']},
            'bmi_distribution': {'labels': ['Underweight', 'Normal', 'Overweight', 'Obese'], 'values': [5, 50, 30, 15], 'colors': ['#17a2b8', '#28a745', '#ffc107', '#dc3545']},
            'age_distribution': {'labels': ['18-25', '26-35', '36-45', '46-55', '56+'], 'values': [20, 35, 25, 15, 5], 'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']},
            'wellness_trend': {'dates': ['Day 1', 'Day 2', 'Day 3'], 'values': [65, 70, 68]}
        }
    
    def create_user_charts(result, form_data=None):
        """Fallback function when chart_utils is not available"""
        return {
            'risk_factors': {'labels': ['BMI Related', 'Lifestyle', 'Medical', 'Habits'], 'values': [30, 40, 20, 10], 'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']},
            'health_metrics': {'labels': ['Nutrition', 'Exercise', 'Sleep', 'Stress'], 'user_scores': [75, 60, 80, 40], 'target_scores': [80, 80, 80, 80]}
        }

try:
    from utils.report_utils import generate_report_data
except ImportError:
    print("Warning: Could not import report_utils. Using fallback function.")
    
    def generate_report_data(form_data, result):
        """Fallback function when report_utils is not available"""
        return {
            'timestamp': datetime.now().strftime("%B %d, %Y %I:%M %p"),
            'summary': 'Complete health assessment report',
            'user_info': {
                'age': form_data.get('age', 'N/A'),
                'gender': form_data.get('gender', 'N/A'),
                'bmi': f"{result.get('bmi', 0):.1f}",
                'bmi_category': result.get('bmi_category', 'N/A')
            }
        }

from prediction import predict_health_risk

# =====================================================
# INITIALIZE FLASK APP
# =====================================================

from flask_sqlalchemy import SQLAlchemy

# =====================================================
# INITIALIZE FLASK APP
# =====================================================

app = Flask(__name__)
app.secret_key = 'health_dashboard_secret_key_2024'

# Database Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'health_risk.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.String(30), default=datetime.now().isoformat)
    # Store core metrics for easy querying
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    bmi = db.Column(db.Float)
    wellness_score = db.Column(db.Float)
    risk_category = db.Column(db.String(20))
    health_status = db.Column(db.String(20))
    # Store full JSON data for flexibility
    form_data_json = db.Column(db.Text)
    result_json = db.Column(db.Text)

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'form_data': json.loads(self.form_data_json),
            'result': json.loads(self.result_json),
            'user_id': self.user_id
        }

# Create tables
with app.app_context():
    db.create_all()

# =====================================================
# ROUTES - MAKE SURE EACH HAS UNIQUE ENDPOINT NAME
# =====================================================

@app.route('/')
def home():
    """Home page - Project overview"""
    total_predictions = Prediction.query.count()
    
    # Calculate stats for home page
    stats = {
        'total_assessments': total_predictions,
        'low_risk_count': Prediction.query.filter_by(risk_category='Low Risk').count(),
        'medium_risk_count': Prediction.query.filter_by(risk_category='Medium Risk').count(),
        'high_risk_count': Prediction.query.filter_by(risk_category='High Risk').count()
    }
    
    return render_template('index.html', 
                         total_predictions=total_predictions,
                         stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Risk prediction form - ONLY ONE predict FUNCTION"""
    if request.method == 'POST':
        try:
            print("=" * 50)
            print("FORM SUBMISSION STARTED")
            print("=" * 50)
            
            # Collect form data with safe defaults
            form_data = {
                'age': request.form.get('age', '30').strip(),
                'gender': request.form.get('gender', 'Male').strip(),
                'height_cm': request.form.get('height_cm', '170').strip(),
                'weight_kg': request.form.get('weight_kg', '70').strip(),
                'physical_activity_hours_per_week': request.form.get('physical_activity_hours_per_week', '3').strip(),
                'diet_quality': request.form.get('diet_quality', 'Average').strip(),
                'sleep_hours_per_day': request.form.get('sleep_hours_per_day', '7').strip(),
                'stress_level': request.form.get('stress_level', 'Medium').strip(),
                'smoking_status': request.form.get('smoking_status', 'Non-smoker').strip(),
                'alcohol_consumption': request.form.get('alcohol_consumption', 'Moderate').strip(),
                'daily_screen_time_hours': request.form.get('daily_screen_time_hours', '6').strip(),
                'water_intake_liters': request.form.get('water_intake_liters', '2').strip(),
                'fast_food_frequency_per_week': request.form.get('fast_food_frequency_per_week', '2').strip(),
                'mental_wellbeing_score': request.form.get('mental_wellbeing_score', '70').strip(),
                'chronic_conditions': request.form.get('chronic_conditions', 'None').strip(),
                'family_history': request.form.get('family_history', 'No').strip()
            }
            
            print("Raw form data received:", form_data)
            
            # Convert numeric fields
            numeric_fields = ['age', 'height_cm', 'weight_kg', 'physical_activity_hours_per_week',
                            'sleep_hours_per_day', 'daily_screen_time_hours', 'water_intake_liters',
                            'fast_food_frequency_per_week', 'mental_wellbeing_score']
            
            for field in numeric_fields:
                if field in form_data:
                    try:
                        form_data[field] = float(form_data[field])
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert {field} to float")
                        # Set sensible defaults
                        defaults = {
                            'age': 30, 'height_cm': 170, 'weight_kg': 70,
                            'physical_activity_hours_per_week': 3, 'sleep_hours_per_day': 7,
                            'daily_screen_time_hours': 6, 'water_intake_liters': 2,
                            'fast_food_frequency_per_week': 2, 'mental_wellbeing_score': 70
                        }
                        form_data[field] = defaults.get(field, 0)
            
            print("Processed form data:", form_data)
            
            # Get prediction
            result = predict_health_risk(form_data)
            print("Prediction result:", result)
            
            # Create database record
            new_prediction = Prediction(
                user_id='user_' + datetime.now().strftime("%Y%m%d%H%M%S"),
                timestamp=datetime.now().isoformat(),
                age=int(form_data.get('age', 0)),
                gender=form_data.get('gender'),
                bmi=result.get('bmi'),
                wellness_score=result.get('wellness_score'),
                risk_category=result.get('final_risk_category'),
                health_status=result.get('health_status'),
                form_data_json=json.dumps(form_data),
                result_json=json.dumps(result)
            )
            
            db.session.add(new_prediction)
            db.session.commit()
            
            # Use the DB object info
            prediction_record = new_prediction.to_dict()
            
            # Store in session
            session['current_prediction'] = result
            session['prediction_data'] = form_data
            session['prediction_id'] = prediction_record['user_id']
            
            print("=" * 50)
            print("FORM SUBMISSION COMPLETE")
            print("=" * 50)
            
            flash('Health risk assessment completed successfully!', 'success')
            return redirect(url_for('show_results'))
            
        except Exception as e:
            print(f"ERROR in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'Error: {str(e)}', 'danger')
            return render_template('predict.html')
    
    # GET request - show the form
    return render_template('predict.html')

# NOTE: Changed from 'results' to 'show_results' to avoid conflict
@app.route('/results')
def show_results():
    """Display prediction results"""
    if 'current_prediction' not in session:
        flash('No prediction found. Please complete the assessment first.', 'warning')
        return redirect(url_for('predict'))
    
    result = session['current_prediction']
    form_data = session.get('prediction_data', {})
    
    # Create charts for results page
    result_charts = create_user_charts(result, form_data)
    
    return render_template('results.html', 
                         result=result, 
                         charts=result_charts,
                         prediction_id=session.get('prediction_id', 'N/A'))

@app.route('/insights')
def show_insights():
    """Data insights dashboard"""
    total_predictions = Prediction.query.count()
    if total_predictions == 0:
        return render_template('insights.html', 
                              has_data=False, 
                              total_predictions=0,
                              stats={})
    
    
    # query database
    predictions = Prediction.query.order_by(Prediction.timestamp.asc()).all()
    
    # Convert history to DataFrame
    df_data = []
    for pred in predictions:
        df_data.append({
            'risk_category': pred.risk_category or 'Low Risk',
            'bmi': pred.bmi or 25,
            'wellness_score': pred.wellness_score or 70,
            'age': pred.age or 30,
            'gender': pred.gender or 'Male',
            'timestamp': pred.timestamp
        })
    
    df = pd.DataFrame(df_data)
    
    # Create dashboard charts
    charts = create_dashboard_charts(df)
    
    # Calculate statistics
    total = len(predictions)
    risk_counts = df['risk_category'].value_counts()
    
    stats = {
        'total_assessments': total,
        'avg_bmi': round(df['bmi'].mean(), 1),
        'avg_wellness': round(df['wellness_score'].mean(), 1),
        'high_risk_pct': round((risk_counts.get('High Risk', 0) / total * 100), 1) if total > 0 else 0,
        'medium_risk_pct': round((risk_counts.get('Medium Risk', 0) / total * 100), 1) if total > 0 else 0,
        'low_risk_pct': round((risk_counts.get('Low Risk', 0) / total * 100), 1) if total > 0 else 0
    }
    
    # Get recent history for table (last 10)
    recent_history = [p.to_dict() for p in Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()]
    
    return render_template('insights.html', 
                          charts=charts, 
                          stats=stats, 
                          has_data=True,
                          predictions_history=recent_history)

@app.route('/recommendations')
def show_recommendations():
    """Health recommendations page"""
    if 'current_prediction' not in session:
        flash('No prediction found. Please complete the assessment first.', 'warning')
        return redirect(url_for('predict'))
    
    result = session['current_prediction']
    
    return render_template('recommendations.html', result=result)

@app.route('/report')
def generate_report():
    """Generate and display report"""
    if 'current_prediction' not in session:
        flash('No prediction found. Please complete the assessment first.', 'warning')
        return redirect(url_for('predict'))
    
    result = session['current_prediction']
    form_data = session.get('prediction_data', {})
    
    # Generate comprehensive report data
    report_data = generate_report_data(form_data, result)
    
    return render_template('report.html', 
                         risk_results=result, 
                         user_data=form_data,
                         report_data=report_data)

@app.route('/download-report')
def download_report():
    """Download PDF report"""
    if 'current_prediction' not in session:
        flash('No prediction found. Please complete the assessment first.', 'warning')
        return redirect(url_for('predict'))
    
    # Create a simple text report
    result = session['current_prediction']
    form_data = session.get('prediction_data', {})
    
    report_text = f"""HEALTH RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PERSONAL INFORMATION:
Age: {form_data.get('age', 'N/A')}
Gender: {form_data.get('gender', 'N/A')}
Height: {form_data.get('height_cm', 'N/A')} cm
Weight: {form_data.get('weight_kg', 'N/A')} kg

ASSESSMENT RESULTS:
BMI: {result.get('bmi', 'N/A')} ({result.get('bmi_category', 'N/A')})
Wellness Score: {result.get('wellness_score', 'N/A')}/100 ({result.get('wellness_category', 'N/A')})
Risk Category: {result.get('final_risk_category', 'N/A')}
Health Status: {result.get('health_status', 'N/A')}

RECOMMENDATIONS:
{result.get('health_suggestion', 'No recommendations available')}
"""
    
    # Create file in memory
    report_bytes = report_text.encode('utf-8')
    report_io = io.BytesIO(report_bytes)
    report_io.seek(0)
    
    filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    return send_file(
        report_io,
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain'
    )

# =====================================================
# API ENDPOINTS
# =====================================================

@app.route('/api/chart-data')
def api_chart_data():
    """API endpoint for chart data"""
    predictions = Prediction.query.all()
    if not predictions:
        return jsonify({'error': 'No data available', 'sample': True})
    
    df_data = []
    for pred in predictions:
        df_data.append({
            'risk_category': pred.risk_category or 'Low Risk',
            'bmi': pred.bmi or 25,
            'wellness_score': pred.wellness_score or 70,
            'age': pred.age or 30
        })
    
    df = pd.DataFrame(df_data)
    
    # Risk distribution
    risk_dist = df['risk_category'].value_counts().to_dict()
    
    # BMI categories
    bmi_categories = pd.cut(df['bmi'], 
                           bins=[0, 18.5, 25, 30, 100],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    bmi_dist = bmi_categories.value_counts().to_dict()
    
    return jsonify({
        'risk_distribution': risk_dist,
        'bmi_distribution': bmi_dist,
        'total_assessments': len(predictions),
        'sample': False
    })

@app.route('/api/predict', methods=['POST'])
def api_predict_endpoint():
    """API endpoint for predictions - DIFFERENT NAME from main predict"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = predict_health_risk(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# =====================================================
# UTILITY ROUTES
# =====================================================

@app.route('/clear-session')
def clear_session():
    """Clear current session data"""
    session.pop('current_prediction', None)
    session.pop('prediction_data', None)
    session.pop('prediction_id', None)
    flash('Session cleared. You can start a new assessment.', 'info')
    return redirect(url_for('home'))

@app.route('/reset-all')
def reset_all_data():
    """Reset all data (for development)"""
    db.session.query(Prediction).delete()
    db.session.commit()
    session.clear()
    flash('All data has been reset.', 'info')
    return redirect(url_for('home'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'predictions_count': Prediction.query.count()
    })

# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("Health Risk Dashboard Starting...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("Available Routes:")
    print("  /                    - Home page")
    print("  /predict             - Risk assessment form")
    print("  /results             - View results")
    print("  /insights            - Data insights")
    print("  /recommendations     - Health recommendations")
    print("  /report              - Generate report")
    print("  /download-report     - Download report")
    print("  /clear-session       - Clear current session")
    print("  /reset-all           - Reset all data")
    print("=" * 60)
    print(f"Access the dashboard at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)