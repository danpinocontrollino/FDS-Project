"""
================================================================================
PREDICTION EXPLANATION ENGINE
================================================================================
This module provides explainable AI capabilities for mental health predictions.

For each prediction, it calculates:
1. Feature contributions - How much each behavior impacts the predicted outcome
2. Personalized explanations - Why THIS user got THIS prediction
3. Actionable recommendations - What specific changes would help most
4. Impact estimates - Expected improvement from each recommendation

The explanation engine uses feature importance weights from comprehensive
analysis (Random Forest + Permutation + Correlation) combined with user-specific
deviations from population norms to generate SHAP-like explanations.

Author: University Project - Burnout Prediction
Date: 2025-12-05
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

WEIGHTS_PATH = Path("data/processed/feature_importance_weights.json")

# Feature controllability (can user actually change this?)
CONTROLLABLE_FEATURES = {
    "work_hours": True,
    "sleep_hours": True,
    "sleep_quality": True,
    "exercise_minutes": True,
    "caffeine_mg": True,
    "alcohol_units": True,
    "screen_time_hours": True,
    "social_interactions": True,
    "outdoor_time_minutes": True,
    "diet_quality": True,
    "work_pressure": True,  # Partially controllable
    "meetings_count": False,  # Usually set by others
    "tasks_completed": False,  # Outcome of work, not input
    "emails_received": False,  # Can't control incoming emails
    "commute_minutes": False,  # Usually fixed by location
    "steps_count": False,  # Tracked metric, use exercise_minutes instead
    "weather_mood_impact": False,  # External factor
}

# Inverted targets (higher = better)
INVERTED_TARGETS = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FeatureContribution:
    """Represents one feature's contribution to a prediction."""
    
    feature: str
    display_name: str
    user_value: float
    population_mean: float
    population_std: float
    z_score: float  # Standardized deviation from mean
    importance_weight: float  # From feature importance analysis
    contribution_points: float  # Actual impact on prediction
    direction: str  # "increases" or "decreases"
    is_controllable: bool
    optimal_range: Tuple[float, float]
    unit: str
    
    @property
    def deviation_percent(self) -> float:
        """How far from average (as percentage)."""
        if self.population_mean == 0:
            return 0.0
        return ((self.user_value - self.population_mean) / self.population_mean) * 100
    
    @property
    def is_suboptimal(self) -> bool:
        """Is user value outside optimal range?"""
        optimal_min, optimal_max = self.optimal_range
        return self.user_value < optimal_min or self.user_value > optimal_max
    
    @property
    def distance_to_optimal(self) -> float:
        """How far from optimal range."""
        optimal_min, optimal_max = self.optimal_range
        if self.user_value < optimal_min:
            return optimal_min - self.user_value
        elif self.user_value > optimal_max:
            return self.user_value - optimal_max
        return 0.0


@dataclass
class Recommendation:
    """An actionable recommendation for improving outcomes."""
    
    feature: str
    display_name: str
    current_value: float
    recommended_value: float
    expected_impact: float  # Expected change in target (points)
    priority: str  # "high", "medium", "low"
    action_description: str
    difficulty: str  # "easy", "moderate", "hard"
    unit: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature": self.feature,
            "display_name": self.display_name,
            "current_value": round(self.current_value, 1),
            "recommended_value": round(self.recommended_value, 1),
            "expected_impact": round(self.expected_impact, 2),
            "priority": self.priority,
            "action": self.action_description,
            "difficulty": self.difficulty,
            "unit": self.unit
        }


@dataclass
class PredictionExplanation:
    """Complete explanation for a prediction."""
    
    target: str
    target_display_name: str
    predicted_value: float
    population_mean: float
    baseline_prediction: float  # What average person would get
    contributions: List[FeatureContribution]
    recommendations: List[Recommendation]
    summary: str
    
    @property
    def top_increasing_factors(self) -> List[FeatureContribution]:
        """Top factors making prediction worse."""
        increasing = [c for c in self.contributions if c.contribution_points > 0]
        return sorted(increasing, key=lambda x: abs(x.contribution_points), reverse=True)[:5]
    
    @property
    def top_decreasing_factors(self) -> List[FeatureContribution]:
        """Top factors making prediction better (protective factors)."""
        decreasing = [c for c in self.contributions if c.contribution_points < 0]
        return sorted(decreasing, key=lambda x: abs(x.contribution_points), reverse=True)[:5]
    
    @property
    def top_recommendations(self) -> List[Recommendation]:
        """Top 3 highest priority recommendations."""
        return [r for r in self.recommendations if r.priority == "high"][:3]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target": self.target,
            "target_display_name": self.target_display_name,
            "predicted_value": round(self.predicted_value, 2),
            "population_mean": round(self.population_mean, 2),
            "baseline_prediction": round(self.baseline_prediction, 2),
            "summary": self.summary,
            "top_increasing_factors": [
                {
                    "feature": c.display_name,
                    "value": round(c.user_value, 1),
                    "contribution": round(c.contribution_points, 2),
                    "unit": c.unit
                }
                for c in self.top_increasing_factors[:3]
            ],
            "top_protecting_factors": [
                {
                    "feature": c.display_name,
                    "value": round(c.user_value, 1),
                    "contribution": round(c.contribution_points, 2),
                    "unit": c.unit
                }
                for c in self.top_decreasing_factors[:3]
            ],
            "recommendations": [r.to_dict() for r in self.top_recommendations]
        }


# ============================================================================
# EXPLANATION ENGINE
# ============================================================================

class ExplanationEngine:
    """
    Generates personalized explanations for mental health predictions.
    
    Uses feature importance weights + user deviations to calculate
    SHAP-like contributions for each feature.
    """
    
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        """
        Initialize explanation engine.
        
        Args:
            weights_path: Path to feature importance weights JSON
        """
        self.weights_path = weights_path
        self.weights_data = self._load_weights()
        
    def _load_weights(self) -> Dict:
        """Load feature importance weights and population statistics."""
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Feature importance weights not found: {self.weights_path}\n"
                f"Run feature importance analysis first to generate this file."
            )
        
        with open(self.weights_path, 'r') as f:
            return json.load(f)
    
    def calculate_contribution(
        self,
        user_value: float,
        feature: str,
        target: str
    ) -> float:
        """
        Calculate feature contribution to prediction.
        
        Formula: contribution = importance_weight × z_score × target_std × direction
        
        Direction is determined by correlation with target:
        - For stress/burnout: high work_hours = bad (positive), high sleep = good (negative)
        - For mood: high exercise = good (negative contribution to low mood)
        
        This gives us a value in the same units as the target variable,
        representing how much this feature pushes the prediction up or down.
        
        Args:
            user_value: User's value for this feature
            feature: Feature name
            target: Target variable name
        
        Returns:
            Contribution in target units (e.g., stress points)
        """
        # Get feature importance weight for this target
        importance = self.weights_data["targets"][target]["feature_weights"].get(feature, 0.0)
        
        # Get population statistics
        pop_stats = self.weights_data["feature_population_stats"][feature]
        pop_mean = pop_stats["mean"]
        pop_std = pop_stats["std"]
        optimal_min, optimal_max = pop_stats["optimal_range"]
        
        # Get target statistics
        target_stats = self.weights_data["targets"][target]["population_stats"]
        target_std = target_stats["std"]
        
        # Calculate z-score (how many standard deviations from mean)
        if pop_std == 0:
            z_score = 0.0
        else:
            z_score = (user_value - pop_mean) / pop_std
        
        # Determine direction of effect based on optimal range
        # If user is above optimal, that's typically bad (increases stress/burnout)
        # If user is below optimal, that's also typically bad
        # Exception: Features where MORE is always better (exercise, sleep)
        # or LESS is always better (caffeine, alcohol, work_hours)
        
        beneficial_features = ["sleep_hours", "sleep_quality", "exercise_minutes", 
                               "social_interactions", "outdoor_time_minutes", "diet_quality"]
        harmful_features = ["work_hours", "caffeine_mg", "alcohol_units", 
                           "screen_time_hours", "work_pressure", "commute_minutes"]
        
        # Determine if target is inverted (higher = better)
        is_inverted_target = target in INVERTED_TARGETS
        
        if feature in beneficial_features:
            # For beneficial features: being above mean helps
            # - For normal targets (stress/anxiety): helps by LOWERING them (negative contribution)
            # - For inverted targets (mood/energy): helps by RAISING them (positive contribution)
            if is_inverted_target:
                contribution = importance * z_score * target_std  # More sleep = higher mood
            else:
                contribution = -importance * z_score * target_std  # More sleep = lower stress
        elif feature in harmful_features:
            # For harmful features: being above mean hurts
            # - For normal targets (stress): hurts by RAISING them (positive contribution)
            # - For inverted targets (mood): hurts by LOWERING them (negative contribution)
            if is_inverted_target:
                contribution = -importance * z_score * target_std  # More work_hours = lower mood
            else:
                contribution = importance * z_score * target_std  # More work_hours = higher stress
        else:
            # For neutral features, use standard calculation
            contribution = importance * z_score * target_std
        
        return contribution
    
    def explain_prediction(
        self,
        user_data: Dict[str, float],
        prediction: float,
        target: str
    ) -> PredictionExplanation:
        """
        Generate complete explanation for a prediction.
        
        Args:
            user_data: Dictionary of user's feature values
            prediction: Predicted value for target
            target: Target variable name (e.g., "stress_level")
        
        Returns:
            PredictionExplanation object with contributions and recommendations
        """
        if target not in self.weights_data["targets"]:
            raise ValueError(f"Unknown target: {target}")
        
        target_info = self.weights_data["targets"][target]
        target_stats = target_info["population_stats"]
        feature_weights = target_info["feature_weights"]
        
        # Calculate contributions for all features
        contributions = []
        
        for feature, importance in feature_weights.items():
            if feature not in user_data:
                continue
            
            user_value = user_data[feature]
            pop_stats = self.weights_data["feature_population_stats"][feature]
            
            contrib_points = self.calculate_contribution(user_value, feature, target)
            
            z_score = (user_value - pop_stats["mean"]) / pop_stats["std"] if pop_stats["std"] > 0 else 0.0
            
            contribution = FeatureContribution(
                feature=feature,
                display_name=self.weights_data["feature_display_names"].get(feature, feature),
                user_value=user_value,
                population_mean=pop_stats["mean"],
                population_std=pop_stats["std"],
                z_score=z_score,
                importance_weight=importance,
                contribution_points=contrib_points,
                direction="increases" if contrib_points > 0 else "decreases",
                is_controllable=CONTROLLABLE_FEATURES.get(feature, False),
                optimal_range=tuple(pop_stats["optimal_range"]),
                unit=pop_stats["unit"]
            )
            
            contributions.append(contribution)
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution_points), reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(contributions, target)
        
        # Generate summary
        summary = self._generate_summary(prediction, contributions, target_stats)
        
        # Calculate baseline (what average person would get)
        baseline = target_stats["mean"]
        
        return PredictionExplanation(
            target=target,
            target_display_name=target.replace("_", " ").title(),
            predicted_value=prediction,
            population_mean=target_stats["mean"],
            baseline_prediction=baseline,
            contributions=contributions,
            recommendations=recommendations,
            summary=summary
        )
    
    def _generate_recommendations(
        self,
        contributions: List[FeatureContribution],
        target: str
    ) -> List[Recommendation]:
        """
        Generate actionable recommendations based on contributions.
        
        Prioritizes:
        1. High-impact, controllable features
        2. Features currently outside optimal range
        3. Easy changes first
        """
        recommendations = []
        
        for contrib in contributions:
            # Skip if not controllable
            if not contrib.is_controllable:
                continue
            
            # Skip if contribution is negligible  
            if abs(contrib.contribution_points) < 0.1:
                continue
            
            # Only recommend changes for features making things WORSE
            # (positive contribution to bad outcomes like stress/burnout)
            if contrib.contribution_points <= 0:
                continue
            
            # Determine recommended value (midpoint of optimal range)
            optimal_min, optimal_max = contrib.optimal_range
            recommended_value = (optimal_min + optimal_max) / 2
            
            # Calculate expected impact if user moves to optimal
            # Use the current contribution as estimate (conservative: 70%)
            expected_impact = -abs(contrib.contribution_points * 0.7)
            
            # Determine priority based on impact magnitude
            if abs(expected_impact) > 0.5:
                priority = "high"
            elif abs(expected_impact) > 0.25:
                priority = "medium"
            else:
                priority = "low"
            
            # Determine difficulty based on magnitude of change needed
            change_magnitude = abs(contrib.user_value - recommended_value) / contrib.population_std
            if change_magnitude < 0.5:
                difficulty = "easy"
            elif change_magnitude < 1.5:
                difficulty = "moderate"
            else:
                difficulty = "hard"
            
            # Generate action description
            action = self._generate_action_description(
                contrib.feature,
                contrib.display_name,
                contrib.user_value,
                recommended_value,
                contrib.unit
            )
            
            recommendation = Recommendation(
                feature=contrib.feature,
                display_name=contrib.display_name,
                current_value=contrib.user_value,
                recommended_value=recommended_value,
                expected_impact=expected_impact,
                priority=priority,
                action_description=action,
                difficulty=difficulty,
                unit=contrib.unit
            )
            
            recommendations.append(recommendation)
        
        # Sort by priority (high first), then expected impact
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(
            key=lambda r: (priority_order[r.priority], -abs(r.expected_impact))
        )
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _generate_action_description(
        self,
        feature: str,
        display_name: str,
        current: float,
        recommended: float,
        unit: str
    ) -> str:
        """Generate natural language action description."""
        diff = recommended - current
        
        # Feature-specific recommendations
        actions = {
            "work_hours": {
                "increase": f"Add {abs(diff):.0f} hours of productive work",
                "decrease": f"Reduce work hours by {abs(diff):.0f}h (set boundaries, delegate tasks)"
            },
            "sleep_hours": {
                "increase": f"Sleep {abs(diff):.1f}h more per night (aim for {recommended:.0f}h total)",
                "decrease": f"Reduce sleep to {recommended:.0f}h (may indicate oversleeping)"
            },
            "sleep_quality": {
                "increase": f"Improve sleep quality to {recommended:.0f}/10 (dark room, no screens before bed)",
                "decrease": ""  # Doesn't make sense to decrease quality
            },
            "exercise_minutes": {
                "increase": f"Add {abs(diff):.0f} minutes of exercise daily",
                "decrease": f"Reduce exercise intensity (possible overtraining)"
            },
            "caffeine_mg": {
                "increase": f"Increase caffeine to {recommended:.0f}mg (if needed for focus)",
                "decrease": f"Cut caffeine by {abs(diff):.0f}mg (switch to decaf after 2pm)"
            },
            "work_pressure": {
                "increase": "",  # Never recommend increasing pressure
                "decrease": f"Reduce work pressure (discuss workload with manager, prioritize tasks)"
            },
            "social_interactions": {
                "increase": f"Add {abs(diff):.0f} social interactions per day (call friends, join activities)",
                "decrease": f"Reduce social obligations (protect alone time)"
            },
            "outdoor_time_minutes": {
                "increase": f"Spend {abs(diff):.0f} more minutes outdoors (walk breaks, eat lunch outside)",
                "decrease": ""
            },
            "screen_time_hours": {
                "increase": "",
                "decrease": f"Reduce screen time by {abs(diff):.0f}h (digital detox, no phone in bedroom)"
            },
            "alcohol_units": {
                "increase": "",  # Never recommend more alcohol
                "decrease": f"Reduce alcohol to {recommended:.0f} units (alcohol-free days, smaller portions)"
            },
            "diet_quality": {
                "increase": f"Improve diet quality to {recommended:.0f}/10 (more vegetables, less processed food)",
                "decrease": ""
            }
        }
        
        direction = "increase" if diff > 0 else "decrease"
        
        if feature in actions and actions[feature][direction]:
            return actions[feature][direction]
        
        # Generic fallback
        if diff > 0:
            return f"Increase {display_name} from {current:.1f} to {recommended:.1f} {unit}"
        else:
            return f"Decrease {display_name} from {current:.1f} to {recommended:.1f} {unit}"
    
    def _generate_summary(
        self,
        prediction: float,
        contributions: List[FeatureContribution],
        target_stats: Dict
    ) -> str:
        """Generate natural language summary of prediction."""
        mean = target_stats["mean"]
        
        # Determine if prediction is high, average, or low
        if prediction > mean + 0.5:
            level = "high"
        elif prediction < mean - 0.5:
            level = "low"
        else:
            level = "average"
        
        # Get top contributor
        top_contrib = contributions[0] if contributions else None
        
        if not top_contrib:
            return f"Your score of {prediction:.1f} is {level} (population average: {mean:.1f})."
        
        contrib_text = f"{abs(top_contrib.contribution_points):.1f} points"
        
        if level == "high":
            summary = (
                f"Your score of {prediction:.1f} is above average (population mean: {mean:.1f}). "
                f"The main driver is **{top_contrib.display_name}** ({top_contrib.user_value:.1f} {top_contrib.unit}), "
                f"which is contributing +{contrib_text}."
            )
        elif level == "low":
            summary = (
                f"Your score of {prediction:.1f} is below average (population mean: {mean:.1f}). "
                f"The main protective factor is **{top_contrib.display_name}** ({top_contrib.user_value:.1f} {top_contrib.unit})."
            )
        else:
            summary = (
                f"Your score of {prediction:.1f} is close to average (population mean: {mean:.1f}). "
                f"Key factor: **{top_contrib.display_name}** ({top_contrib.user_value:.1f} {top_contrib.unit})."
            )
        
        return summary


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def explain_predictions(
    user_data: Dict[str, float],
    predictions: Dict[str, float],
    targets: Optional[List[str]] = None
) -> Dict[str, PredictionExplanation]:
    """
    Generate explanations for multiple predictions at once.
    
    Args:
        user_data: Dictionary of user's feature values
        predictions: Dictionary of predictions {target: value}
        targets: List of targets to explain (default: all in predictions)
    
    Returns:
        Dictionary mapping target names to PredictionExplanation objects
    """
    engine = ExplanationEngine()
    
    if targets is None:
        targets = list(predictions.keys())
    
    explanations = {}
    for target in targets:
        if target in predictions:
            explanation = engine.explain_prediction(
                user_data=user_data,
                prediction=predictions[target],
                target=target
            )
            explanations[target] = explanation
    
    return explanations


def explain_single(
    user_data: Dict[str, float],
    prediction: float,
    target: str
) -> PredictionExplanation:
    """
    Generate explanation for a single prediction.
    
    Convenience wrapper for single-prediction use cases.
    """
    engine = ExplanationEngine()
    return engine.explain_prediction(user_data, prediction, target)


# ============================================================================
# MAIN (TESTING/DEMO)
# ============================================================================

def main():
    """Demo the explanation engine with example data."""
    print("="*80)
    print("EXPLANATION ENGINE DEMO")
    print("="*80)
    
    # Example user data
    user_data = {
        "sleep_hours": 5.5,  # Low
        "sleep_quality": 4.0,  # Low
        "work_hours": 11.2,  # High
        "meetings_count": 6.0,
        "tasks_completed": 8.0,
        "emails_received": 85.0,
        "commute_minutes": 45.0,
        "exercise_minutes": 15.0,  # Low
        "steps_count": 4500.0,
        "caffeine_mg": 550.0,  # High
        "alcohol_units": 2.0,
        "screen_time_hours": 10.0,  # High
        "social_interactions": 2.0,  # Low
        "outdoor_time_minutes": 10.0,  # Low
        "diet_quality": 4.0,  # Low
        "work_pressure": 2.0,  # High
        "weather_mood_impact": -1.0
    }
    
    # Example prediction
    prediction = 7.8  # High stress
    target = "stress_level"
    
    # Generate explanation
    engine = ExplanationEngine()
    explanation = engine.explain_prediction(user_data, prediction, target)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"PREDICTION: {explanation.target_display_name} = {explanation.predicted_value:.1f}")
    print(f"POPULATION AVERAGE: {explanation.population_mean:.1f}")
    print(f"{'='*80}")
    
    print(f"\n📊 {explanation.summary}\n")
    
    print("🔴 TOP FACTORS INCREASING STRESS:")
    for i, contrib in enumerate(explanation.top_increasing_factors[:5], 1):
        print(f"  {i}. {contrib.display_name}: {contrib.user_value:.1f} {contrib.unit}")
        print(f"     → Contributing +{contrib.contribution_points:.2f} points")
        print(f"     → Population average: {contrib.population_mean:.1f} {contrib.unit}\n")
    
    if explanation.top_decreasing_factors:
        print("✅ TOP PROTECTIVE FACTORS:")
        for i, contrib in enumerate(explanation.top_decreasing_factors[:3], 1):
            print(f"  {i}. {contrib.display_name}: {contrib.user_value:.1f} {contrib.unit}")
            print(f"     → Contributing {contrib.contribution_points:.2f} points\n")
    
    print("💡 TOP RECOMMENDATIONS:")
    if not explanation.recommendations:
        print("  (Debugging: No recommendations generated)")
        print(f"  Total contributions: {len(explanation.contributions)}")
        controllable = [c for c in explanation.contributions if c.is_controllable]
        print(f"  Controllable features: {len(controllable)}")
        positive = [c for c in controllable if c.contribution_points > 0]
        print(f"  Positive contributions (making it worse): {len(positive)}")
        if positive:
            print(f"  Top positive: {positive[0].feature} = {positive[0].contribution_points:.2f}")
    
    for i, rec in enumerate(explanation.recommendations[:5], 1):
        print(f"  {i}. [{rec.priority.upper()}] {rec.display_name}")
        print(f"     Current: {rec.current_value:.1f} {rec.unit} → Recommended: {rec.recommended_value:.1f} {rec.unit}")
        print(f"     Action: {rec.action_description}")
        print(f"     Expected impact: {rec.expected_impact:+.1f} stress points")
        print(f"     Difficulty: {rec.difficulty}\n")
    
    print(f"{'='*80}")
    print("EXPLANATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
