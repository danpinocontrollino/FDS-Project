"""
DEBUGGING SCRIPT: Check actual StudentLife data structure
Run this first to see what data is actually available
"""

import pandas as pd
import numpy as np
import os
import json

BASE_PATH = '/kaggle/input/student-life/dataset'

print("=" * 80)
print("DEBUGGING: ACTUAL DATA STRUCTURE")
print("=" * 80)

# Check one student's phonelock data
student_id = 'u00'

print("\n1. PHONELOCK DATA")
print("-" * 80)
phonelock_path = os.path.join(BASE_PATH, 'sensing', 'phonelock', f'phonelock_{student_id}.csv')
try:
    df = pd.read_csv(phonelock_path, encoding='utf-8-sig', nrows=10)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)
except Exception as e:
    print(f"Error: {e}")

print("\n2. ACTIVITY DATA")
print("-" * 80)
activity_path = os.path.join(BASE_PATH, 'sensing', 'activity', f'activity_{student_id}.csv')
try:
    df = pd.read_csv(activity_path, encoding='utf-8-sig', nrows=10)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    if 'activity_inference' in df.columns:
        print(f"\nUnique activity values: {df['activity_inference'].unique()}")
except Exception as e:
    print(f"Error: {e}")

print("\n3. EMA STRESS DATA")
print("-" * 80)
stress_path = os.path.join(BASE_PATH, 'EMA', 'response', 'Stress', f'Stress_{student_id}.json')
try:
    with open(stress_path, 'r') as f:
        data = json.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"\nFirst entry:")
                print(json.dumps(data[0], indent=2))
                if len(data) > 1:
                    print(f"\nSecond entry:")
                    print(json.dumps(data[1], indent=2))
except Exception as e:
    print(f"Error: {e}")

print("\n4. EMA MOOD DATA")
print("-" * 80)
mood_path = os.path.join(BASE_PATH, 'EMA', 'response', 'Mood', f'Mood_{student_id}.json')
try:
    with open(mood_path, 'r') as f:
        data = json.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"\nFirst entry:")
                print(json.dumps(data[0], indent=2))
except Exception as e:
    print(f"Error: {e}")

print("\n5. CONVERSATION DATA")
print("-" * 80)
conv_path = os.path.join(BASE_PATH, 'sensing', 'conversation', f'conversation_{student_id}.csv')
try:
    df = pd.read_csv(conv_path, encoding='utf-8-sig', nrows=10)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

print("\n6. APP USAGE DATA")
print("-" * 80)
app_path = os.path.join(BASE_PATH, 'app_usage', f'running_app_{student_id}.csv')
try:
    df = pd.read_csv(app_path, encoding='utf-8-sig', nrows=10)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

print("\n7. CALL LOG DATA")
print("-" * 80)
call_path = os.path.join(BASE_PATH, 'call_log', f'call_log_{student_id}.csv')
try:
    df = pd.read_csv(call_path, encoding='utf-8-sig', nrows=10)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE - Copy this output and share it")
print("=" * 80)
