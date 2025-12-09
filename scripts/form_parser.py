"""
================================================================================
UNIFIED GOOGLE FORM PARSER
================================================================================
Consolidates all Google Form CSV parsing logic into a single, maintainable module.

This module provides:
- GoogleFormParser: Central class for all form parsing
- Unified column mapping (100+ Google Form variations)
- Standardized categorical mappings
- Numeric cleaning (handles European formats, ranges, units)
- Format auto-detection (weekly vs daily)

USAGE:
    from form_parser import GoogleFormParser
    
    parser = GoogleFormParser()
    df = parser.parse_google_form_csv("responses.csv")
    format_type = parser.detect_format(df)

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class GoogleFormParser:
    """Unified Google Form CSV parser with flexible column mapping."""
    
    def __init__(self):
        """Initialize parser with unified column mappings and categorical mappings."""
        
        # ====================================================================
        # UNIFIED COLUMN MAPPING
        # ====================================================================
        # Maps 100+ Google Form column variations to standardized feature names
        # IMPORTANT: More specific patterns MUST come before generic ones!
        # E.g., "recover after stress" must come before "stress"
        
        self.column_mapping = {
            # ===== IDENTITY & METADATA =====
            "informazioni cronologiche": "_timestamp",
            "name / email": "_name",
            "name/email": "_name",
            "your name": "_name",
            "what is your name": "_name",
            "email": "_email",
            "which day": "_day",
            "what day": "_day",
            
            # ===== RECOVERY (MUST come before "stress" to avoid conflicts) =====
            "how well do you recover": "recovery_ability",
            "recover after stress": "recovery_ability",
            "recover from a stressful": "recovery_ability",
            "recover from": "recovery_ability",
            "recovery ability": "recovery_ability",
            "recovery:": "recovery_ability",
            
            # ===== STRESS (after recovery) =====
            "stress level": "stress_level",
            "how stressed": "stress_level",
            "stress": "stress_level",
            "perceived work pressure": "work_pressure",
            "work pressure": "work_pressure",
            "pressure": "work_pressure",
            
            # ===== SLEEP =====
            "sleep hours (last night)": "sleep_hours",
            "sleep hours": "sleep_hours",
            "hours of sleep": "sleep_hours",
            "how many hours": "sleep_hours",
            "sleep quality": "sleep_quality",
            "quality of sleep": "sleep_quality",
            
            # ===== MOOD & ENERGY =====
            "how did you feel when you woke up": "morning_mood",
            "feel when you woke up": "morning_mood",
            "woke up this morning": "morning_mood",
            "morning mood": "morning_mood",
            "mood score": "mood_score",
            "mood": "mood_score",
            "how do you feel": "mood_score",
            "energy level": "energy_level",
            "energy": "energy_level",
            "focus / productivity": "focus_score",
            "focus": "focus_score",
            
            # ===== WORK =====
            "work hours (today)": "work_hours",
            "work hours": "work_hours",
            "hours worked": "work_hours",
            "number of meetings": "meetings_count",
            "meetings": "meetings_count",
            "emails received (approx)": "emails_received",
            "emails received": "emails_received",
            "emails": "emails_received",
            "tasks completed": "tasks_completed",
            "tasks": "tasks_completed",
            "how productive were you during": "productivity_today",
            "productive were you during": "productivity_today",
            "workday productivity": "productivity_today",
            "overall, how was your day": "day_overall_rating",
            "how was your day": "day_overall_rating",
            "overall how was your day": "day_overall_rating",
            
            # ===== PHYSICAL ACTIVITY =====
            "exercise / workout (minutes)": "exercise_minutes",
            "exercise": "exercise_minutes",
            "workout": "exercise_minutes",
            "steps count": "steps_count",
            "steps": "steps_count",
            "time spent outdoors today": "outdoor_time_minutes",
            "time spent outdoors": "outdoor_time_minutes",
            "time outdoors": "outdoor_time_minutes",
            "outdoor time": "outdoor_time_minutes",
            "outdoor": "outdoor_time_minutes",
            
            # ===== COMMUTE & SCREEN =====
            "commute time (minutes)": "commute_minutes",
            "commute time": "commute_minutes",
            "commute": "commute_minutes",
            "total screen time (hours)": "screen_time_hours",
            "screen time": "screen_time_hours",
            "screen hours": "screen_time_hours",
            "does your job require extended screen time?": "job_requires_screen",
            "job require extended screen": "job_requires_screen",
            "screen time required": "job_requires_screen",
            "extended screen time": "job_requires_screen",
            
            # ===== DIET & SUBSTANCES =====
            "diet quality": "diet_quality",
            "diet": "diet_quality",
            "nutrition": "diet_quality",
            "caffeine intake (mg)": "caffeine_mg",
            "caffeine": "caffeine_mg",
            "coffee": "caffeine_mg",
            "alcohol units": "alcohol_units",
            "alcohol": "alcohol_units",
            "drinks": "alcohol_units",
            
            # ===== JOB CONTEXT =====
            "what best describes your job type?": "job_type",
            "what best describes your job": "job_type",
            "job type": "job_type",
            "job title": "job_type",
            "primary work arrangement": "work_arrangement",
            "work arrangement": "work_arrangement",
            "workspace type": "workspace_type",
            
            # ===== SOCIAL & ISOLATION =====
            # IMPORTANT: More specific patterns first!
            "number of meaningful social interactions today": "social_interactions",
            "number of meaningful social": "social_interactions",
            "meaningful social interaction": "social_interactions",
            "quality of social interactions today": "social_quality",
            "quality of social interaction": "social_quality",
            "quality of social": "social_quality",
            "how often do you feel lonely or isolated?": "loneliness_level",
            "lonely or isolated": "loneliness_level",
            "lonely": "loneliness_level",
            "isolated": "loneliness_level",
            "do you have someone to talk to about personal problems?": "social_support",
            "someone to talk to": "social_support",
            "did you reach out to anyone for help": "reached_out_for_help",
            "reach out to anyone for help": "reached_out_for_help",
            "reached out to anyone": "reached_out_for_help",
            
            # ===== BOUNDARIES & RECOVERY =====
            "work-life boundary clarity": "work_life_boundary",
            "work-life boundary": "work_life_boundary",
            "boundary clarity": "work_life_boundary",
            "boundary": "work_life_boundary",
            "checking emails outside work hours?": "after_hours_checking",
            "checking emails outside": "after_hours_checking",
            "check work outside": "after_hours_checking",
            "outside hours": "after_hours_checking",
            "how many times have you checked your email after hours": "email_checks_after_hours",
            "checked your email after hours": "email_checks_after_hours",
            "email after hours": "email_checks_after_hours",
            "easy for you to completely disconnect": "work_disconnect_ease",
            "disconnect from work emails": "work_disconnect_ease",
            
            # ===== WORK ENVIRONMENT =====
            "how distracting was your work environment?": "environment_distractions",
            "distracting": "environment_distractions",
            "distraction": "environment_distractions",
            "take breaks": "break_flexibility",
            "break flexibility": "break_flexibility",
            
            # ===== CHRONOTYPE & PERSONALITY =====
            "natural sleep pattern (chronotype)": "chronotype",
            "natural sleep pattern": "chronotype",
            "chronotype": "chronotype",
            "sleep pattern": "chronotype",
            "morning person": "chronotype",
            "night owl": "chronotype",
            
            # ===== JOB SATISFACTION & MOOD =====
            "job satisfaction (weekly)": "job_satisfaction",
            "job satisfaction": "job_satisfaction",
            "satisfied with job": "job_satisfaction",
        }
        
        # ====================================================================
        # CATEGORICAL MAPPINGS
        # ====================================================================
        
        self.job_type_mapping = {
            # 💻 Software Engineer
            "software developer": "software_engineer",
            "engineer": "software_engineer",
            "programmer": "software_engineer",
            "it": "software_engineer",
            "tech": "software_engineer",
            "data scientist": "software_engineer",
            "developer": "software_engineer",
            
            # 🏫 Teacher
            "teacher": "teacher",
            "professor": "teacher",
            "instructor": "teacher",
            "tutor": "teacher",
            "educator": "teacher",
            
            # 🏥 Nurse
            "nurse": "nurse",
            "healthcare": "nurse",
            "physiotherapist": "nurse",
            "doctor": "nurse",
            "medical": "nurse",
            
            # 👔 Manager
            "manager": "manager",
            "executive": "manager",
            "director": "manager",
            "supervisor": "manager",
            "ceo": "manager",
            "lead": "manager",
            "leadership": "manager",
            
            # 🏢 Operations
            "operations": "operations",
            "administrative": "operations",
            "coordinator": "operations",
            "specialist": "operations",
            "admin": "operations",
            
            # 💪 Wellness Coach
            "wellness coach": "wellness_coach",
            "health coach": "wellness_coach",
            "fitness": "wellness_coach",
            "trainer": "wellness_coach",
            "psychologist": "wellness_coach",
            "coach": "wellness_coach",
            "counselor": "wellness_coach",
        }
        
        self.work_arrangement_mapping = {
            "remote": "remote",
            "work from home": "remote",
            "wfh": "remote",
            "fully remote": "remote",
            
            "hybrid": "hybrid",
            "flexible": "hybrid",
            "sometimes in office": "hybrid",
            
            "office": "office",
            "in-office": "office",
            "on-site": "office",
            "fully in office": "office",
        }
        
        self.chronotype_mapping = {
            "early riser": "early",
            "morning person": "early",
            "lark": "early",
            "early bird": "early",
            
            "intermediate": "intermediate",
            "neither": "intermediate",
            "flexible": "intermediate",
            
            "night owl": "late",
            "evening person": "late",
            "owl": "late",
            "late riser": "late",
        }
        
        self.loneliness_mapping = {
            "never": 0,
            "rarely": 1,
            "sometimes": 2,
            "often": 3,
            "always": 4,
        }
        
        self.work_pressure_mapping = {
            "low": 0,
            "low pressure": 0,
            "moderate": 1,
            "medium": 1,
            "high": 2,
            "high pressure": 2,
        }
        
        self.work_life_boundary_mapping = {
            "very clear": 4,
            "clear": 4,
            "somewhat clear": 3,
            "neutral": 2,
            "somewhat blurred": 2,
            "blurred": 1,
            "very blurred": 1,
        }
        
        self.morning_mood_mapping = {
            "very bad": 1,
            "bad": 2,
            "neutral": 3,
            "good": 4,
            "excellent": 5,
        }
        
        self.yes_no_mapping = {
            "yes": 1,
            "no": 0,
            "sometimes": 1,
        }
        
        # Default feature columns (from training)
        self.feature_cols = [
            "stress_level",
            "commute_minutes",
            "exercise_minutes",
            "work_hours",
            "mood_score",
            "sleep_quality",
            "emails_received",
            "caffeine_mg",
            "energy_level",
            "sleep_hours",
            "focus_score",
            "meetings_count",
            "tasks_completed",
            "steps_count",
            "alcohol_units",
            "screen_time_hours",
            "work_pressure",
            "morning_mood",
            "productivity_today",
            "day_overall_rating",
        ]
        
        # Default values for missing features
        self.defaults = {
            "stress_level": 5,
            "commute_minutes": 30,
            "exercise_minutes": 30,
            "work_hours": 8,
            "mood_score": 6,
            "sleep_quality": 6,
            "emails_received": 20,
            "caffeine_mg": 100,
            "energy_level": 6,
            "sleep_hours": 7,
            "focus_score": 6,
            "meetings_count": 3,
            "tasks_completed": 5,
            "steps_count": 5000,
            "alcohol_units": 0,
            "screen_time_hours": 4,
            "work_pressure": 1,
            "morning_mood": 3,
            "productivity_today": 3,
            "day_overall_rating": 6,
            "social_quality": 6,
            "social_interactions": 3,
            "outdoor_time_minutes": 30,
            "diet_quality": 5,
            "job_satisfaction": 6,
            "recovery_ability": 5,
            "job_type": "other",
            "work_arrangement": "hybrid",
            "environment_distractions": 5,
            "work_life_boundary": 2,
            "after_hours_checking": 2,
            "break_flexibility": 3,
            "chronotype": "intermediate",
        }
    
    def normalize_column_name(self, col: str) -> Optional[str]:
        """
        Normalize a Google Form column name to feature name.
        
        Args:
            col: Raw column name from Google Form
            
        Returns:
            Standardized feature name, or None if no mapping found
        """
        col_lower = col.lower().strip()
        
        # Try exact match first
        if col_lower in self.column_mapping:
            return self.column_mapping[col_lower]
        
        # Try substring matching (more flexible)
        for pattern, feature in self.column_mapping.items():
            if pattern in col_lower:
                return feature
        
        return None
    
    def clean_numeric(self, value: any, default: float = 5.0) -> float:
        """
        Extract numeric value from messy user input.
        
        Handles:
        - Missing values (returns default)
        - Direct numbers (returns as-is)
        - Italian text ("abbastanza", "molto", "poco")
        - Time durations ("2+ hours", "15-30 min")
        - European format ("15.000" = 15000)
        - Ranges ("10/15", "6-10")
        
        Args:
            value: Raw input value (can be any type)
            default: Value to return if cleaning fails
            
        Returns:
            Cleaned numeric value
        """
        if pd.isna(value):
            return default
        
        if isinstance(value, (int, float)):
            return float(value)
        
        s = str(value).strip().lower()
        
        # ===== TEXT → NUMBER MAPPINGS =====
        text_to_num = {
            "abbastanza": 7,      # "enough" in Italian
            "molto": 8,           # "a lot"
            "poco": 3,            # "little"
            "normale": 5,         # "normal"
            "high": 2,
            "medium": 1,
            "low": 0,
            "yes": 1,
            "no": 0,
        }
        
        for text, num in text_to_num.items():
            if text in s:
                return float(num)
        
        # ===== TIME DURATION PARSING =====
        # "2+ hours" → 120 minutes
        hour_match = re.search(r'(\d+)\+?\s*hour', s)
        if hour_match:
            hours = float(hour_match.group(1))
            return hours * 60
        
        # "15-30 min" → 22.5 (midpoint)
        min_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*min', s)
        if min_range_match:
            low = float(min_range_match.group(1))
            high = float(min_range_match.group(2))
            return (low + high) / 2
        
        # "< 15 min", "30 min" → extract value
        single_min_match = re.search(r'[<>]?\s*(\d+)\s*min', s)
        if single_min_match:
            return float(single_min_match.group(1))
        
        # ===== NUMERIC PARSING =====
        # Remove units and text
        s_cleaned = re.sub(r'[a-zA-Z%°]+', '', s).strip()
        
        # Handle European format "15.000" or "15,000" = 15000
        # Pattern: "X.000" or "X,000" (thousands with separator)
        european_match = re.match(r'^(\d{1,3})[.,](\d{3})$', s_cleaned)
        if european_match:
            thousands = float(european_match.group(1))
            return thousands * 1000
        
        # Handle decimal numbers (European: 3,14 or US: 3.14)
        try:
            # Try US format first (. = decimal)
            if '.' in s_cleaned and ',' not in s_cleaned:
                return float(s_cleaned)
            
            # Try European format (, = decimal)
            if ',' in s_cleaned:
                return float(s_cleaned.replace(',', '.'))
        except ValueError:
            pass
        
        # Extract first number found
        numbers = re.findall(r'\d+', s_cleaned)
        if numbers:
            return float(numbers[0])
        
        return default
    
    def map_categorical(self, feature: str, value: str) -> any:
        """
        Map categorical text value to standardized category.
        
        Args:
            feature: Feature name (e.g., "job_type")
            value: Raw categorical value
            
        Returns:
            Standardized category code or value
        """
        if pd.isna(value):
            return self.defaults.get(feature)
        
        value_str = str(value).lower().strip()
        
        if feature == "job_type":
            # Try exact match, then keyword match
            for key, mapped in self.job_type_mapping.items():
                if key in value_str:
                    return mapped
            return "other"
        
        elif feature == "work_arrangement":
            for key, mapped in self.work_arrangement_mapping.items():
                if key in value_str:
                    return mapped
            return "hybrid"
        
        elif feature == "chronotype":
            for key, mapped in self.chronotype_mapping.items():
                if key in value_str:
                    return mapped
            return "intermediate"
        
        elif feature == "loneliness_level":
            return self.loneliness_mapping.get(value_str, 2)
        
        elif feature == "work_pressure":
            return self.work_pressure_mapping.get(value_str, 1)
        
        elif feature == "work_life_boundary":
            return self.work_life_boundary_mapping.get(value_str, 2)
        
        elif feature == "morning_mood":
            return self.morning_mood_mapping.get(value_str, 3)
        
        elif feature in ["reached_out_for_help", "work_disconnect_ease"]:
            return self.yes_no_mapping.get(value_str, 0)
        
        return value
    
    def parse_google_form_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """
        Parse a Google Form CSV export and map to standardized format.
        
        This function:
        1. Maps verbose column headers to standardized names
        2. Cleans numeric values (ranges, units, European format)
        3. Maps categorical text to standardized values
        4. Fills missing features with defaults
        
        Args:
            csv_path: Path to the Google Form CSV export
            
        Returns:
            DataFrame with cleaned columns and values
        """
        df = pd.read_csv(csv_path)
        
        # Map columns
        mapped_df = pd.DataFrame()
        for col in df.columns:
            feature_name = self.normalize_column_name(col)
            if feature_name:
                mapped_df[feature_name] = df[col]
        
        # Clean numeric columns
        numeric_features = [
            "sleep_hours", "sleep_quality", "work_hours", "stress_level",
            "mood_score", "energy_level", "focus_score", "exercise_minutes",
            "caffeine_mg", "alcohol_units", "steps_count", "meetings_count",
            "emails_received", "commute_minutes", "screen_time_hours",
            "social_quality", "social_interactions", "outdoor_time_minutes",
            "diet_quality", "job_satisfaction", "recovery_ability",
            "environment_distractions", "productivity_today", "day_overall_rating",
            "email_checks_after_hours"
        ]
        
        for col in numeric_features:
            if col in mapped_df.columns:
                mapped_df[col] = mapped_df[col].apply(
                    lambda x: self.clean_numeric(x, self.defaults.get(col, 5))
                )
        
        # Map categorical columns
        categorical_features = [
            "job_type", "work_arrangement", "chronotype",
            "loneliness_level", "work_pressure", "work_life_boundary",
            "morning_mood", "reached_out_for_help", "work_disconnect_ease"
        ]
        
        for col in categorical_features:
            if col in mapped_df.columns:
                mapped_df[col] = mapped_df[col].apply(
                    lambda x: self.map_categorical(col, x) if pd.notna(x) else self.defaults.get(col)
                )
        
        # Fill missing features with defaults
        for feature, default in self.defaults.items():
            if feature not in mapped_df.columns:
                mapped_df[feature] = default
        
        return mapped_df
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """
        Detect whether data is in daily or weekly format.
        
        Args:
            df: Parsed DataFrame
            
        Returns:
            "daily" if daily format detected, else "weekly"
        """
        # Check for day indicators
        if "_day" in df.columns:
            return "daily"
        
        if "_day_number" in df.columns:
            return "daily"
        
        # Check for multiple rows per user (indicates daily)
        if "_name" in df.columns:
            name_counts = df["_name"].value_counts()
            if (name_counts > 1).any():
                return "daily"
        
        # Default to weekly
        return "weekly"
