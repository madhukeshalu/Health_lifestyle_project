from flask import Flask, render_template
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

# Mock data
mock_stats = {
    'total_assessments': 100,
    'avg_wellness': 75.5,
    'avg_bmi': 24.5,
    'high_risk_pct': 15.0,
    'low_risk_pct': 50.0,
    'medium_risk_pct': 35.0
}

mock_charts = {
    'risk_distribution': {'labels': ['Low', 'Medium', 'High'], 'values': [50, 35, 15]},
    'bmi_distribution': {'labels': ['Normal', 'Overweight'], 'values': [60, 40]},
    'wellness_trend': {'dates': ['2023-01', '2023-02'], 'data_values': [70, 75]},
    'risk_factors': {'labels': ['Factor A'], 'data_values': [10]},
    'health_metrics': {'labels': ['Metric A'], 'user_scores': [80], 'target_scores': [90]}
}

mock_result = {
    'final_risk_category': 'Low Risk',
    'health_status': 'Good',
    'risk_reason': 'All good',
    'high_risk_probability': 0.1,
    'bmi': 22.0,
    'bmi_category': 'Normal',
    'wellness_score': 85,
    'wellness_category': 'Excellent',
    'critical_factors_count': 0,
    'risk_factors': [],
    'priority_focus': 'Keep it up'
}

@app.route('/test_insights')
def test_insights():
    try:
        render_template('insights.html', stats=mock_stats, charts=mock_charts, has_data=True, predictions_history=[])
        return "Insights Template OK"
    except Exception as e:
        return f"Insights Error: {e}"

@app.route('/test_results')
def test_results():
    try:
        render_template('results.html', result=mock_result, charts=mock_charts)
        return "Results Template OK"
    except Exception as e:
        return f"Results Error: {e}"

if __name__ == '__main__':
    with app.test_request_context():
        print(test_insights())
        print(test_results())
