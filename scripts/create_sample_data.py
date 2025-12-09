"""
Sample test data for Streamlit demo
3 user profiles with different risk levels
"""

import pandas as pd
from pathlib import Path

# Sample profiles
profiles = [
    {
        'name': 'Sarah (Healthy)',
        'email': 'sarah@example.com',
        'sleep_hours': 8.0,
        'sleep_quality': 8.5,
        'work_hours': 8.0,
        'meetings_count': 3,
        'tasks_completed': 7,
        'emails_received': 45,
        'commute_minutes': 15,
        'exercise_minutes': 45,
        'steps_count': 8500,
        'caffeine_mg': 150,
        'alcohol_units': 0.5,
        'screen_time_hours': 5.0,
        'social_interactions': 6,
        'outdoor_time_minutes': 60,
        'diet_quality': 8.0,
        'work_pressure': 4,
        'weather_mood_impact': 1,
        'job_title': 'Senior Engineer'
    },
    {
        'name': 'Alex (At-Risk)',
        'email': 'alex@example.com',
        'sleep_hours': 5.0,
        'sleep_quality': 4.0,
        'work_hours': 10.5,
        'meetings_count': 8,
        'tasks_completed': 12,
        'emails_received': 120,
        'commute_minutes': 45,
        'exercise_minutes': 5,
        'steps_count': 2000,
        'caffeine_mg': 600,
        'alcohol_units': 2.0,
        'screen_time_hours': 12.0,
        'social_interactions': 1,
        'outdoor_time_minutes': 5,
        'diet_quality': 4.0,
        'work_pressure': 8,
        'weather_mood_impact': -2,
        'job_title': 'Product Manager'
    },
    {
        'name': 'Jordan (Improving)',
        'email': 'jordan@example.com',
        'sleep_hours': 6.5,
        'sleep_quality': 6.0,
        'work_hours': 9.0,
        'meetings_count': 5,
        'tasks_completed': 8,
        'emails_received': 80,
        'commute_minutes': 30,
        'exercise_minutes': 25,
        'steps_count': 4500,
        'caffeine_mg': 350,
        'alcohol_units': 1.0,
        'screen_time_hours': 8.0,
        'social_interactions': 3,
        'outdoor_time_minutes': 20,
        'diet_quality': 6.0,
        'work_pressure': 6,
        'weather_mood_impact': 0,
        'job_title': 'Designer'
    }
]

def create_sample_data():
    """Create and save sample data."""
    
    # Expand each profile to 7 days (for time-series)
    data = []
    for profile in profiles:
        for day in range(1, 8):
            row = profile.copy()
            # Add day number (slight variations for realism)
            row['day'] = day
            
            # Add small daily variations
            if day > 1:
                row['sleep_hours'] += (0.5 if day % 2 == 0 else -0.3)
                row['sleep_quality'] += (0.3 if day % 2 == 0 else -0.2)
                row['stress_level'] = 7 - (row['sleep_hours'] - 6) + (row['work_hours'] - 8) * 0.5
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path('demo_test_profiles.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample data created: {output_path}")
    print(f"  - Profiles: 3 (Healthy, At-Risk, Improving)")
    print(f"  - Days per profile: 7")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Features: {len(df.columns)}")
    
    return output_path

if __name__ == "__main__":
    create_sample_data()
