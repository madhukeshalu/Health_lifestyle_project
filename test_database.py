# test_database.py
from database import db

def test_database():
    print("Testing Database Operations...")
    print("="*50)
    
    # Test user creation
    user_id = db.create_user("test_user", "test@example.com", 30, "Male")
    print(f"✅ User created with ID: {user_id}")
    
    # Test health data saving
    health_data = {
        'height_cm': 175,
        'weight_kg': 70,
        'bmi': 22.86,
        'physical_activity_hours_per_week': 5,
        'diet_quality': 'Good',
        'sleep_hours_per_day': 7,
        'stress_level': 'Medium',
        'smoking_status': 'Non-smoker',
        'alcohol_consumption': 'Moderate',
        'daily_screen_time_hours': 4,
        'water_intake_liters': 2.5,
        'fast_food_frequency_per_week': 1,
        'mental_wellbeing_score': 80,
        'chronic_conditions': 'None',
        'family_history': 'No'
    }
    
    data_id = db.save_health_data(user_id, health_data)
    print(f"✅ Health data saved with ID: {data_id}")
    
    # Test KPI metrics
    kpi = db.get_kpi_summary(30)
    print(f"✅ KPI Summary: {kpi}")
    
    # Test export
    csv_file = db.export_to_csv()
    print(f"✅ CSV Export: {csv_file}")
    
    # Test logs
    logs = db.get_system_logs(5)
    print(f"✅ System Logs (last 5): {len(logs)} entries")
    
    db.close()
    print("\n✅ All database tests passed!")

if __name__ == "__main__":
    test_database()