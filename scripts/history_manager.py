"""
================================================================================
USER HISTORY MANAGER - File-Based Daily Data Tracking
================================================================================
Manages daily behavioral data storage and retrieval for longitudinal analysis.

Features:
  - Save daily entries as JSON files per user
  - Retrieve last N days of data
  - Calculate trends over time
  - Compare day-over-day changes

Storage Structure:
    data/
      user_history/
        sarah@example.com/
          2024-12-01.json
          2024-12-02.json
          ...

Author: FDS Project Team
================================================================================
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


class UserHistoryManager:
    """Manages user behavioral data history using file-based storage."""
    
    def __init__(self, base_dir: str = "data/user_history"):
        """
        Initialize history manager.
        
        Args:
            base_dir: Root directory for storing user history files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_daily_entry(self, user_id: str, data: Dict, date: Optional[str] = None) -> Path:
        """
        Save one day of behavioral data for a user.
        
        Args:
            user_id: User identifier (email or unique ID)
            data: Dictionary with behavioral features (17 features + predictions)
            date: Date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            Path to saved JSON file
            
        Example:
            >>> manager = UserHistoryManager()
            >>> data = {
            ...     'sleep_hours': 7.5,
            ...     'work_hours': 8.0,
            ...     'stress_prediction': 4.2,
            ...     # ... other features
            ... }
            >>> manager.save_daily_entry('sarah@example.com', data)
        """
        # Create user directory
        user_dir = self.base_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        # Determine date
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        filepath = user_dir / f"{date}.json"
        
        # Add metadata
        data['timestamp'] = datetime.now().isoformat()
        data['date'] = date
        data['user_id'] = user_id
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved entry for {user_id} on {date}")
        return filepath
    
    def get_last_n_days(self, user_id: str, n: int = 7) -> pd.DataFrame:
        """
        Load last N days of data for a user.
        
        Args:
            user_id: User identifier
            n: Number of days to retrieve
            
        Returns:
            DataFrame with behavioral data, sorted by date (oldest to newest)
        """
        user_dir = self.base_dir / user_id
        
        if not user_dir.exists():
            print(f"⚠️  No history found for {user_id}")
            return pd.DataFrame()
        
        # Get all JSON files, sorted by date (newest first)
        files = sorted(user_dir.glob("*.json"), reverse=True)[:n]
        
        if len(files) < n:
            print(f"⚠️  Only {len(files)}/{n} days available for {user_id}")
        
        if not files:
            return pd.DataFrame()
        
        # Load into list
        data = []
        for file in files:
            with open(file, 'r') as f:
                entry = json.load(f)
                data.append(entry)
        
        # Convert to DataFrame and sort chronologically
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df = df.sort_values('date')
        elif 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    def get_date_range(self, user_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data for specific date range.
        
        Args:
            user_id: User identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with data in date range
        """
        user_dir = self.base_dir / user_id
        
        if not user_dir.exists():
            return pd.DataFrame()
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        data = []
        for file in user_dir.glob("*.json"):
            file_date = datetime.strptime(file.stem, "%Y-%m-%d")
            
            if start <= file_date <= end:
                with open(file, 'r') as f:
                    data.append(json.load(f))
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data).sort_values('date')
    
    def get_trend_data(self, user_id: str, days: int = 30) -> Dict[str, List]:
        """
        Get trend data for visualization (last N days).
        
        Args:
            user_id: User identifier
            days: Number of days to retrieve
            
        Returns:
            Dictionary with dates and values for plotting:
            {
                'dates': ['2024-12-01', '2024-12-02', ...],
                'stress': [4.2, 5.1, ...],
                'mood': [7.5, 6.8, ...],
                'sleep_hours': [7.5, 6.0, ...],
                ...
            }
        """
        user_dir = self.base_dir / user_id
        
        if not user_dir.exists():
            return {}
        
        # Get files from last N days
        cutoff = datetime.now() - timedelta(days=days)
        files = []
        
        for file in user_dir.glob("*.json"):
            try:
                file_date = datetime.strptime(file.stem, "%Y-%m-%d")
                if file_date >= cutoff:
                    files.append((file_date, file))
            except ValueError:
                continue
        
        # Sort by date
        files.sort(key=lambda x: x[0])
        
        # Build trend dictionary
        trends = {
            'dates': [],
            'stress_level': [],
            'mood_score': [],
            'energy_level': [],
            'anxiety_score': [],
            'depression_score': [],
            'job_satisfaction': [],
            'sleep_hours': [],
            'work_hours': [],
            'exercise_minutes': []
        }
        
        for date, file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                
                trends['dates'].append(date.strftime("%Y-%m-%d"))
                
                # Add predictions if available
                for key in ['stress_level', 'mood_score', 'energy_level', 
                           'anxiety_score', 'depression_score', 'job_satisfaction']:
                    trends[key].append(data.get(key, None))
                
                # Add behavioral features
                for key in ['sleep_hours', 'work_hours', 'exercise_minutes']:
                    trends[key].append(data.get(key, None))
        
        return trends
    
    def calculate_day_over_day_change(self, user_id: str) -> Dict[str, float]:
        """
        Calculate change from yesterday to today.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with changes for each metric:
            {
                'stress_level': +0.5,
                'mood_score': -1.2,
                ...
            }
        """
        # Get last 2 days
        df = self.get_last_n_days(user_id, n=2)
        
        if len(df) < 2:
            return {}
        
        yesterday = df.iloc[-2]
        today = df.iloc[-1]
        
        changes = {}
        metrics = ['stress_level', 'mood_score', 'energy_level', 'anxiety_score',
                  'depression_score', 'job_satisfaction', 'sleep_hours', 'work_hours']
        
        for metric in metrics:
            if metric in yesterday and metric in today:
                try:
                    changes[metric] = float(today[metric]) - float(yesterday[metric])
                except (ValueError, TypeError):
                    changes[metric] = None
        
        return changes
    
    def get_user_stats(self, user_id: str, days: int = 30) -> Dict:
        """
        Calculate summary statistics for user.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        df = self.get_last_n_days(user_id, n=days)
        
        if df.empty:
            return {}
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        stats = {}
        
        for col in numeric_cols:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return stats
    
    def list_users(self) -> List[str]:
        """
        Get list of all users with history.
        
        Returns:
            List of user IDs
        """
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def get_user_day_count(self, user_id: str) -> int:
        """
        Count number of days recorded for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of days with data
        """
        user_dir = self.base_dir / user_id
        
        if not user_dir.exists():
            return 0
        
        return len(list(user_dir.glob("*.json")))
    
    def delete_user_history(self, user_id: str):
        """
        Delete all history for a user (use with caution!).
        
        Args:
            user_id: User identifier
        """
        user_dir = self.base_dir / user_id
        
        if user_dir.exists():
            import shutil
            shutil.rmtree(user_dir)
            print(f"🗑️  Deleted all history for {user_id}")
        else:
            print(f"⚠️  No history found for {user_id}")


if __name__ == "__main__":
    # Example usage
    manager = UserHistoryManager()
    
    # Example data
    example_data = {
        'sleep_hours': 7.5,
        'sleep_quality': 4,
        'work_hours': 8.0,
        'meetings_count': 3,
        'tasks_completed': 12,
        'emails_received': 45,
        'commute_minutes': 30,
        'exercise_minutes': 30,
        'steps_count': 8000,
        'caffeine_mg': 200,
        'alcohol_units': 0,
        'screen_time_hours': 10,
        'social_interactions': 5,
        'outdoor_time_minutes': 20,
        'diet_quality': 4,
        'work_pressure': 3,
        'weather_mood_impact': 0,
        'stress_level': 4.2,
        'mood_score': 7.5,
        'energy_level': 6.8,
        'job_title': 'Software Engineer'
    }
    
    # Save example
    manager.save_daily_entry('demo@example.com', example_data)
    
    # Retrieve
    history = manager.get_last_n_days('demo@example.com', n=7)
    print(f"\n📊 Retrieved {len(history)} days of history")
    
    # List users
    users = manager.list_users()
    print(f"\n👥 Users with history: {users}")
