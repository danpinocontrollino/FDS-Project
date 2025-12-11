import sys
from pathlib import Path
import pandas as pd

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.generate_profile import detect_contradictions


def run_test():
    # Build user_data: 7 days x features
    cols = [
        "sleep_hours","sleep_quality","work_hours","meetings_count","tasks_completed",
        "emails_received","commute_minutes","exercise_minutes","steps_count","caffeine_mg",
        "alcohol_units","screen_time_hours","social_interactions","outdoor_time_minutes",
        "diet_quality","work_pressure","weather_mood_impact",
    ]
    # 7 days of data
    data = []
    for _ in range(7):
        row = [7.5, 8.0, 8.0, 2, 5, 10, 20, 10, 6000, 450, 0.5, 6, 2, 30, 7, 1, 0]
        data.append(row)
    df = pd.DataFrame(data, columns=cols)

    # Predictions mock (structure expected by detect_contradictions)
    preds = {
        'stress_level': {'value': 3.0, 'at_risk': False},
        'mood_score': {'value': 8.5, 'at_risk': False},
        'energy_level': {'value': 8.5, 'at_risk': False},
        'focus_score': {'value': 7.0, 'at_risk': False},
        'perceived_stress_scale': {'value': 12.0, 'at_risk': False},
        'anxiety_score': {'value': 4.0, 'at_risk': False},
        'depression_score': {'value': 6.0, 'at_risk': False},
        'job_satisfaction': {'value': 8.0, 'at_risk': False},
    }

    contradictions = detect_contradictions(df, preds)
    print('--- Detected contradictions ---')
    for c in contradictions:
        print(c['type'], '-', c['description'])

if __name__ == '__main__':
    run_test()
