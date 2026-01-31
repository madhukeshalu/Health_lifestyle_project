# Health & Lifestyle Risk Dashboard

A comprehensive Flask-based web application for analyzing health risks, providing personalized recommendations, and visualizing lifestyle data.

## 🚀 Features

- **Health Risk Prediction**: AI-powered prediction of health risks based on lifestyle factors.
- **Personalized Insights**: Detailed breakdown of risk factors and health scores.
- **Dynamic Dashboard**: Interactive charts and data visualizations using Chart.js.
- **Health Recommendations**: Categorized suggestions for improving health and wellness.
- **Report Generation**: Exportable health reports for tracking progress.

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Database**: SQLAlchemy (SQLite)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Visualization**: Chart.js
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Health_lifestyle_project.git
   cd Health_lifestyle_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the dashboard**:
   Open your browser and navigate to `http://localhost:5000`

## 📂 Project Structure

- `app.py`: Main application entry point and routes.
- `prediction.py`: Core logic for health risk assessment.
- `models/`: Database models.
- `utils/`: Helper functions for charts and reports.
- `templates/`: HTML templates for the web interface.
- `static/`: CSS styles and client-side JavaScript.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
