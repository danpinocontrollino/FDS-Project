"""
================================================================================
DAILY DATA COLLECTION SCRIPT
================================================================================
Interactive script for collecting daily behavioral data from users.

Run this script daily to:
  - Collect 17 behavioral features
  - Store data in user history
  - Track longitudinal patterns

Usage:
    python scripts/collect_daily_data.py
    
    # Or specify user and date
    python scripts/collect_daily_data.py --user sarah@example.com --date 2024-12-09

Author: FDS Project Team
================================================================================
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from history_manager import UserHistoryManager


def collect_user_data_interactive(manager: UserHistoryManager, user_id: str = None, date: str = None):
    """
    Collect daily data through interactive prompts.
    
    Args:
        manager: UserHistoryManager instance
        user_id: User email/ID (will prompt if None)
        date: Date string YYYY-MM-DD (uses today if None)
    """
    print("=" * 60)
    print("📋 DAILY MENTAL HEALTH CHECK-IN")
    print("=" * 60)
    print()
    
    # Get user ID
    if user_id is None:
        user_id = input("👤 Email address: ").strip()
    else:
        print(f"👤 User: {user_id}")
    
    # Get date
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"📅 Date: {date}")
    print()
    
    # Check existing history
    existing_days = manager.get_user_day_count(user_id)
    if existing_days > 0:
        print(f"📊 You have {existing_days} days of history")
        recent = manager.get_last_n_days(user_id, n=1)
        if not recent.empty:
            last_date = recent.iloc[-1].get('date', 'unknown')
            print(f"   Last entry: {last_date}")
    print()
    
    print("Please enter today's behavioral data:")
    print("-" * 60)
    
    data = {}
    
    # Sleep
    print("\n💤 SLEEP")
    data['sleep_hours'] = float(input("  Hours of sleep (0-12): "))
    data['sleep_quality'] = int(input("  Sleep quality (1=Poor, 5=Excellent): "))
    
    # Work
    print("\n💼 WORK")
    data['work_hours'] = float(input("  Hours worked (0-16): "))
    data['meetings_count'] = int(input("  Number of meetings (0-20): "))
    data['tasks_completed'] = int(input("  Tasks completed (0-50): "))
    data['emails_received'] = int(input("  Emails received (0-200): "))
    data['work_pressure'] = int(input("  Work pressure (1=Low, 5=High): "))
    
    # Commute
    print("\n🚗 COMMUTE")
    data['commute_minutes'] = int(input("  Commute time in minutes (0-180): "))
    
    # Physical Activity
    print("\n🏃 PHYSICAL ACTIVITY")
    data['exercise_minutes'] = int(input("  Exercise minutes (0-180): "))
    data['steps_count'] = int(input("  Steps count (0-30000): "))
    data['outdoor_time_minutes'] = int(input("  Outdoor time minutes (0-300): "))
    
    # Consumption
    print("\n☕ CONSUMPTION")
    data['caffeine_mg'] = int(input("  Caffeine intake mg (0-800): "))
    data['alcohol_units'] = float(input("  Alcohol units (0-10): "))
    data['diet_quality'] = int(input("  Diet quality (1=Poor, 5=Excellent): "))
    
    # Screen & Social
    print("\n📱 SCREEN & SOCIAL")
    data['screen_time_hours'] = float(input("  Screen time hours (0-18): "))
    data['social_interactions'] = int(input("  Social interactions (0-20): "))
    
    # Environment
    print("\n🌤️  ENVIRONMENT")
    data['weather_mood_impact'] = int(input("  Weather mood impact (-2=Negative, 0=Neutral, +2=Positive): "))
    
    # Job info (optional, only ask first time)
    if existing_days == 0:
        print("\n👔 JOB INFO (First time only)")
        data['job_title'] = input("  Job title (e.g., Software Engineer): ").strip()
    
    print()
    print("=" * 60)
    
    # Save data
    try:
        filepath = manager.save_daily_entry(user_id, data, date=date)
        print(f"✅ Data saved successfully!")
        print(f"📁 Location: {filepath}")
        
        # Show day-over-day changes if available
        if existing_days >= 1:
            changes = manager.calculate_day_over_day_change(user_id)
            if changes:
                print("\n📊 CHANGES FROM YESTERDAY:")
                for metric, change in changes.items():
                    if change is not None:
                        emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        sign = "+" if change > 0 else ""
                        print(f"   {emoji} {metric.replace('_', ' ').title()}: {sign}{change:.1f}")
        
        # Next steps
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        if existing_days + 1 >= 7:
            print(f"   ✅ You have {existing_days + 1} days of data!")
            print("   🚀 Run: python scripts/generate_profile.py --user", user_id)
            print("      to generate your mental health profile with predictions")
        else:
            needed = 7 - (existing_days + 1)
            print(f"   📅 Keep logging! {needed} more days needed for predictions")
            print(f"      (Need 7 days total, have {existing_days + 1})")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False
    
    return True


def collect_batch_data(manager: UserHistoryManager, csv_path: str):
    """
    Import historical data from CSV file.
    
    Args:
        manager: UserHistoryManager instance
        csv_path: Path to CSV with columns: date, user_id, [17 features]
    """
    import pandas as pd
    
    print(f"📂 Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} entries")
        
        # Group by user and date
        for (user_id, date), group in df.groupby(['user_id', 'date']):
            if len(group) > 1:
                print(f"⚠️  Multiple entries for {user_id} on {date}, using last one")
                row = group.iloc[-1]
            else:
                row = group.iloc[0]
            
            # Convert row to dict
            data = row.to_dict()
            
            # Save entry
            manager.save_daily_entry(user_id, data, date=date)
        
        print(f"✅ Imported data for {len(df['user_id'].unique())} users")
        
    except Exception as e:
        print(f"❌ Error importing CSV: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect daily behavioral data for mental health tracking"
    )
    parser.add_argument(
        '--user', '-u',
        help='User email/ID (will prompt if not provided)'
    )
    parser.add_argument(
        '--date', '-d',
        help='Date in YYYY-MM-DD format (defaults to today)'
    )
    parser.add_argument(
        '--import-csv',
        help='Import historical data from CSV file'
    )
    parser.add_argument(
        '--list-users',
        action='store_true',
        help='List all users with history'
    )
    parser.add_argument(
        '--user-stats',
        help='Show statistics for specific user'
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = UserHistoryManager()
    
    # List users mode
    if args.list_users:
        users = manager.list_users()
        print(f"\n👥 Users with history ({len(users)}):")
        for user in users:
            days = manager.get_user_day_count(user)
            print(f"   • {user}: {days} days")
        return
    
    # User stats mode
    if args.user_stats:
        stats = manager.get_user_stats(args.user_stats, days=30)
        if not stats:
            print(f"❌ No data found for {args.user_stats}")
            return
        
        print(f"\n📊 Statistics for {args.user_stats} (last 30 days):")
        print("-" * 60)
        for metric, values in stats.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"   Mean: {values['mean']:.2f}")
            print(f"   Std:  {values['std']:.2f}")
            print(f"   Min:  {values['min']:.2f}")
            print(f"   Max:  {values['max']:.2f}")
        return
    
    # Import CSV mode
    if args.import_csv:
        collect_batch_data(manager, args.import_csv)
        return
    
    # Interactive collection mode
    collect_user_data_interactive(manager, user_id=args.user, date=args.date)


if __name__ == "__main__":
    main()
