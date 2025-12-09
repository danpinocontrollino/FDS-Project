#!/usr/bin/env python3
"""
Test the complete history-based workflow for longitudinal mental health tracking.

This script demonstrates:
1. Creating synthetic test user with 7 days of data
2. Collecting daily data using history_manager
3. Generating profile with trend analysis
4. Viewing day-over-day changes

Usage:
    python scripts/test_history_workflow.py
"""

from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from history_manager import UserHistoryManager

def generate_test_data(num_days=7, user_id="test_user@example.com"):
    """Generate realistic synthetic behavioral data for testing."""
    
    manager = UserHistoryManager()
    
    print(f"\n{'='*80}")
    print(f"Creating test data for: {user_id}")
    print(f"{'='*80}\n")
    
    # Base values (healthy baseline)
    base_values = {
        "sleep_hours": 7.5,
        "sleep_quality": 7.0,
        "work_hours": 8.0,
        "meetings_count": 3,
        "tasks_completed": 12,
        "emails_received": 45,
        "commute_minutes": 30,
        "exercise_minutes": 30,
        "steps_count": 8000,
        "caffeine_mg": 200,
        "alcohol_units": 0.5,
        "screen_time_hours": 9,
        "social_interactions": 5,
        "outdoor_time_minutes": 30,
        "diet_quality": 7,
        "work_pressure": 1,
        "weather_mood_impact": 0,
    }
    
    # Simulate gradual burnout trajectory over 7 days
    trajectory = {
        "sleep_hours": [-0.3, -0.2, -0.4, -0.5, -0.3, -0.6, -0.7],  # Declining sleep
        "work_hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],          # Increasing work
        "exercise_minutes": [-2, -5, -7, -10, -12, -15, -18],       # Declining exercise
        "caffeine_mg": [20, 50, 80, 120, 150, 180, 220],            # Rising caffeine
        "work_pressure": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],       # Rising stress
        "social_interactions": [-0.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0],  # Isolation
    }
    
    start_date = datetime.now() - timedelta(days=num_days-1)
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Build data for this day
        daily_data = {}
        for feature, base_val in base_values.items():
            if feature in trajectory:
                # Apply trajectory + noise
                trend = trajectory[feature][day]
                noise = np.random.normal(0, abs(trend) * 0.2)
                daily_data[feature] = base_val + trend + noise
            else:
                # Add small random noise
                noise = np.random.normal(0, base_val * 0.05)
                daily_data[feature] = base_val + noise
            
            # Ensure realistic bounds
            daily_data[feature] = max(0, daily_data[feature])
            
            # Integer features
            if feature in ["meetings_count", "tasks_completed", "emails_received", 
                          "commute_minutes", "exercise_minutes", "steps_count", 
                          "caffeine_mg", "outdoor_time_minutes", "social_interactions"]:
                daily_data[feature] = int(round(daily_data[feature]))
        
        # Save to history
        success = manager.save_daily_entry(
            user_id=user_id,
            data=daily_data,
            date=current_date.strftime("%Y-%m-%d"),
        )
        
        if success:
            print(f"✓ Day {day+1} ({current_date.strftime('%Y-%m-%d')}): "
                  f"Sleep {daily_data['sleep_hours']:.1f}h, "
                  f"Work {daily_data['work_hours']:.1f}h, "
                  f"Exercise {daily_data['exercise_minutes']:.0f}min")
        else:
            print(f"❌ Failed to save day {day+1}")
    
    print(f"\n✓ Created {num_days} days of test data")
    
    # Show statistics
    day_count = manager.get_user_day_count(user_id)
    if day_count > 0:
        print(f"\n📊 User Statistics:")
        print(f"   Total days logged: {day_count}")
        
        # Get date range
        df = manager.get_last_n_days(user_id, n=day_count)
        if df is not None and not df.empty:
            first_date = df.index[0].strftime("%Y-%m-%d") if hasattr(df.index[0], 'strftime') else 'Unknown'
            last_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], 'strftime') else 'Unknown'
            print(f"   Date range: {first_date} to {last_date}")
            print(f"   Job: software_engineer (simulated)")
    
    # Show day-over-day changes
    changes = manager.calculate_day_over_day_change(user_id)
    if changes:
        print(f"\n📈 Day-over-day changes (yesterday → today):")
        significant_changes = {k: v for k, v in changes.items() if abs(v) > 0.5}
        for feature, change in list(significant_changes.items())[:8]:
            arrow = "↑" if change > 0 else "↓"
            print(f"   {arrow} {feature.replace('_', ' ').title()}: {change:+.1f}")
    
    return manager


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           HISTORY-BASED WORKFLOW TEST                                      ║
║  Testing longitudinal mental health tracking with file-based storage       ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Generate test data
    test_user = "test_demo@example.com"
    manager = generate_test_data(num_days=7, user_id=test_user)
    
    # Step 2: Verify data retrieval
    print(f"\n{'='*80}")
    print("Testing data retrieval")
    print(f"{'='*80}\n")
    
    last_7_days = manager.get_last_n_days(test_user, n=7)
    if last_7_days is not None:
        print(f"✓ Retrieved {len(last_7_days)} days of data")
        print(f"\n📊 Sample (last 3 days):")
        print(last_7_days[['sleep_hours', 'work_hours', 'exercise_minutes', 'caffeine_mg']].tail(3).to_string())
    else:
        print("❌ Failed to retrieve data")
        return
    
    # Step 3: Get trend data for visualization
    print(f"\n{'='*80}")
    print("Generating trend data for charts")
    print(f"{'='*80}\n")
    
    trend_data = manager.get_trend_data(test_user, days=7)
    if trend_data:
        print(f"✓ Trend data for {len(trend_data['dates'])} days")
        print(f"   Features tracked: {', '.join(list(trend_data.keys())[:5])}...")
    
    # Step 4: Instructions for next steps
    print(f"\n{'='*80}")
    print("✓ TEST COMPLETE - Next Steps")
    print(f"{'='*80}\n")
    
    print("The test user is ready! Now generate a profile with:")
    print(f"\n  python scripts/generate_profile.py --from-history --user-id {test_user} --html\n")
    
    print("This will:")
    print("  1. Load the 7 days of behavioral data")
    print("  2. Generate mental health predictions")
    print("  3. Analyze day-over-day trends")
    print("  4. Create HTML report with trend charts")
    
    print("\n📂 Data stored in:")
    print(f"   data/user_history/{test_user}/")
    
    print("\n🔍 View raw data:")
    print(f"   ls -lh data/user_history/{test_user}/")


if __name__ == "__main__":
    main()
