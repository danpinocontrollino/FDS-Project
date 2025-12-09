"""
Generate Mock StudentLife Data for Local Testing
=================================================
Creates synthetic StudentLife-like data based on actual statistics
from the real dataset extraction.

This allows local testing without running on Kaggle.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Statistics from actual StudentLife extraction
STUDENT_STATS = {
    'u00': {'sleep': 40, 'exercise': 66, 'screen': 61, 'social': 85, 'work': 84, 'stress': 47, 'mood': 9},
    'u01': {'sleep': 41, 'exercise': 62, 'screen': 57, 'social': 65, 'work': 65, 'stress': 27, 'mood': 2},
    'u02': {'sleep': 40, 'exercise': 60, 'screen': 57, 'social': 66, 'work': 66, 'stress': 39, 'mood': 5},
}

def generate_mock_studentlife_data():
    """Generate mock data matching StudentLife structure."""
    
    records = []
    start_date = datetime(2024, 10, 1)
    
    for student_id, stats in STUDENT_STATS.items():
        # Generate 70 days of data
        for day_idx in range(70):
            date = start_date + timedelta(days=day_idx)
            
            # Behavioral features (based on actual variances)
            record = {
                'student_id': student_id,
                'date': date,
                # Real sensor data (high variance)
                'sleep_hours': np.random.normal(7.0, 1.4) if day_idx < stats['sleep'] else 7.0,
                'sleep_quality': 6.0,
                'work_hours': np.random.gamma(2, 2.5) if day_idx < stats['work'] else 5.0,
                'meetings_count': 2,
                'tasks_completed': 5,
                'emails_received': 15,
                'commute_minutes': 20,
                'exercise_minutes': np.random.gamma(3, 10) if day_idx < stats['exercise'] else 30.0,
                'steps_count': int(np.random.normal(5000, 2000)) if day_idx < stats['exercise'] else 5000,
                'caffeine_mg': 150,
                'alcohol_units': 0,
                'screen_time_hours': np.random.normal(6.0, 4.8) if day_idx < stats['screen'] else 6.0,
                'social_interactions': np.random.poisson(5) if day_idx < stats['social'] else 5,
                'outdoor_time_minutes': 30,
                'diet_quality': 6,
                'work_pressure': 5,
                'weather_mood_impact': 0,
                # Mental health targets (sparse - mostly defaults)
                'stress_level': np.random.uniform(4, 7) if day_idx < stats['stress'] and np.random.random() < 0.3 else 5.0,
                'mood_score': np.random.uniform(5, 8) if day_idx < stats['mood'] and np.random.random() < 0.1 else 6.0,
                'energy_level': 6.0,
                'focus_score': 6.0,
                'perceived_stress_scale': 20.0,
                'anxiety_score': 8.0,
                'depression_score': 6.0,
                'job_satisfaction': 6.5
            }
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Save
    output_path = 'data/processed/studentlife_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Generated mock StudentLife data: {df.shape}")
    print(f"✓ Saved to {output_path}")
    print(f"\nStudents: {df['student_id'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

if __name__ == '__main__':
    generate_mock_studentlife_data()
