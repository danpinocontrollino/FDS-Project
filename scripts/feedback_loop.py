"""
================================================================================
FEEDBACK LOOP & INTERVENTION TRACKING
================================================================================
Tracks the effectiveness of burnout recommendations and adapts suggestions.

Features:
  - Log all interventions with timestamps
  - Track user follow-up compliance
  - Measure outcome (did burnout decrease?)
  - A/B test different recommendation strategies
  - Learn what works for each job type
  - Update model weights based on feedback

Author: FDS Project Team - Feedback Analysis
================================================================================
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FEEDBACK_DIR = PROJECT_ROOT / "data/feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

INTERVENTION_LOG = FEEDBACK_DIR / "interventions.jsonl"  # One intervention per line
EFFECTIVENESS_REPORT = FEEDBACK_DIR / "effectiveness.json"
MODEL_WEIGHTS_UPDATE = FEEDBACK_DIR / "model_updates.json"


# ============================================================================
# INTERVENTION TRACKING
# ============================================================================

class InterventionTracker:
    """Track interventions and their outcomes."""
    
    def __init__(self, user_id: str, job_type: str):
        """
        Initialize tracker for a user.
        
        Args:
            user_id: Email or unique identifier
            job_type: One of the 6 job categories
        """
        self.user_id = user_id
        self.job_type = job_type
        self.interventions: List[Dict] = []
        self.load_history()
    
    def load_history(self) -> List[Dict]:
        """Load previous interventions for this user."""
        self.interventions = []
        if INTERVENTION_LOG.exists():
            with open(INTERVENTION_LOG, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("user_id") == self.user_id:
                            self.interventions.append(data)
        return self.interventions
    
    def log_intervention(
        self,
        burnout_score: float,
        recommendations: List[Dict],
        advice_category: str,
        success_metrics: Optional[Dict] = None,
    ) -> Dict:
        """
        Log an intervention with recommendations given.
        
        Args:
            burnout_score: Current burnout risk (0-100)
            recommendations: List of recommended changes
            advice_category: Job type specific advice given
            success_metrics: Optional baseline metrics to track
            
        Returns:
            Intervention record with ID
        """
        intervention = {
            "user_id": self.user_id,
            "job_type": self.job_type,
            "timestamp": datetime.now().isoformat(),
            "intervention_id": f"{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "burnout_score": float(burnout_score),
            "advice_category": advice_category,
            "recommendations": recommendations,
            "baseline_metrics": success_metrics or {},
            "follow_up_timestamps": [],
            "outcomes": [],
            "effectiveness": None,
        }
        
        # Append to log
        with open(INTERVENTION_LOG, 'a') as f:
            f.write(json.dumps(intervention) + '\n')
        
        self.interventions.append(intervention)
        return intervention
    
    def record_follow_up(
        self,
        intervention_id: str,
        follow_up_metrics: Dict,
        days_elapsed: int,
    ) -> Dict:
        """
        Record follow-up data after intervention.
        
        Args:
            intervention_id: ID of the original intervention
            follow_up_metrics: New metric values
            days_elapsed: Days since intervention
            
        Returns:
            Outcome assessment
        """
        follow_up = {
            "timestamp": datetime.now().isoformat(),
            "days_elapsed": days_elapsed,
            "metrics": follow_up_metrics,
        }
        
        # Find and update the intervention
        updated_intervention = None
        with open(INTERVENTION_LOG, 'r') as f:
            lines = f.readlines()
        
        with open(INTERVENTION_LOG, 'w') as f:
            for line in lines:
                if line.strip():
                    data = json.loads(line)
                    if data.get("intervention_id") == intervention_id:
                        data["follow_up_timestamps"].append(follow_up["timestamp"])
                        data["outcomes"].append(follow_up)
                        updated_intervention = data
                    f.write(json.dumps(data) + '\n')
        
        if updated_intervention:
            self._compute_effectiveness(updated_intervention)
        
        return follow_up
    
    def _compute_effectiveness(self, intervention: Dict) -> float:
        """
        Compute effectiveness of an intervention.
        
        Returns:
            Effectiveness score (0-1):
            - 1.0: Perfect improvement
            - 0.5: No change
            - 0.0: Got worse
        """
        if not intervention["outcomes"]:
            return None
        
        baseline = intervention["baseline_metrics"]
        latest_outcome = intervention["outcomes"][-1]
        current = latest_outcome["metrics"]
        
        # Metrics that DECREASE when improving burnout
        inverse_metrics = {"burnout_score", "stress_level", "anxiety", "depression"}
        
        # Metrics that INCREASE when improving burnout
        positive_metrics = {"sleep_hours", "mood_score", "energy_level", "life_satisfaction"}
        
        improvements = []
        
        for metric_name, baseline_val in baseline.items():
            if metric_name not in current:
                continue
            
            current_val = current[metric_name]
            
            if metric_name in inverse_metrics:
                # Lower is better
                improvement = (baseline_val - current_val) / max(baseline_val, 1)
            elif metric_name in positive_metrics:
                # Higher is better
                improvement = (current_val - baseline_val) / max(baseline_val, 1)
            else:
                # Unknown metric - skip
                continue
            
            # Clamp to -1 to 1
            improvement = max(-1, min(1, improvement))
            improvements.append(improvement)
        
        if not improvements:
            return 0.5  # No data - neutral
        
        # Average improvement, scaled to 0-1
        mean_improvement = statistics.mean(improvements)
        effectiveness = (mean_improvement + 1) / 2  # Maps [-1, 1] to [0, 1]
        
        intervention["effectiveness"] = effectiveness
        return effectiveness
    
    def get_effectiveness_summary(self) -> Dict:
        """Get effectiveness of all interventions for this user."""
        if not self.interventions:
            return {"total": 0, "average_effectiveness": 0, "by_category": {}}
        
        by_category = {}
        all_effectiveness = []
        
        for intervention in self.interventions:
            if intervention["effectiveness"] is not None:
                effectiveness = intervention["effectiveness"]
                all_effectiveness.append(effectiveness)
                
                category = intervention["advice_category"]
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(effectiveness)
        
        # Average by category
        category_avg = {
            cat: statistics.mean(scores)
            for cat, scores in by_category.items()
        }
        
        return {
            "total_interventions": len(self.interventions),
            "with_followup": sum(1 for i in self.interventions if i["outcomes"]),
            "average_effectiveness": statistics.mean(all_effectiveness) if all_effectiveness else 0,
            "by_category": category_avg,
        }


# ============================================================================
# COHORT ANALYSIS
# ============================================================================

class CohortAnalyzer:
    """Analyze interventions across users and job types."""
    
    @staticmethod
    def analyze_by_job_type() -> Dict[str, Dict]:
        """Analyze which recommendations work best for each job."""
        if not INTERVENTION_LOG.exists():
            return {}
        
        job_data = {}
        
        with open(INTERVENTION_LOG, 'r') as f:
            for line in f:
                if line.strip():
                    intervention = json.loads(line)
                    job = intervention["job_type"]
                    
                    if job not in job_data:
                        job_data[job] = {
                            "total": 0,
                            "effective": 0,
                            "recommendations": {},
                        }
                    
                    job_data[job]["total"] += 1
                    
                    if intervention.get("effectiveness", 0) > 0.6:
                        job_data[job]["effective"] += 1
                    
                    # Track which recommendations work
                    for rec in intervention.get("recommendations", []):
                        feature = rec.get("feature", "unknown")
                        if feature not in job_data[job]["recommendations"]:
                            job_data[job]["recommendations"][feature] = {
                                "count": 0,
                                "successful": 0,
                            }
                        
                        job_data[job]["recommendations"][feature]["count"] += 1
                        if intervention.get("effectiveness", 0) > 0.6:
                            job_data[job]["recommendations"][feature]["successful"] += 1
        
        # Calculate success rates
        for job, data in job_data.items():
            data["success_rate"] = (
                data["effective"] / data["total"] if data["total"] > 0 else 0
            )
            
            # Success rate per recommendation
            for feature, rec_data in data["recommendations"].items():
                rec_data["success_rate"] = (
                    rec_data["successful"] / rec_data["count"]
                    if rec_data["count"] > 0
                    else 0
                )
        
        return job_data
    
    @staticmethod
    def find_best_interventions() -> Dict[str, Any]:
        """Find the most effective interventions overall."""
        if not INTERVENTION_LOG.exists():
            return {"most_effective": [], "least_effective": []}
        
        interventions = []
        
        with open(INTERVENTION_LOG, 'r') as f:
            for line in f:
                if line.strip():
                    intervention = json.loads(line)
                    if intervention.get("effectiveness") is not None:
                        interventions.append(intervention)
        
        # Sort by effectiveness
        interventions.sort(key=lambda x: x["effectiveness"], reverse=True)
        
        return {
            "most_effective": interventions[:10],
            "least_effective": interventions[-10:] if len(interventions) > 10 else [],
            "total_tracked": len(interventions),
        }


# ============================================================================
# MODEL WEIGHT UPDATES
# ============================================================================

def generate_model_update_weights(job_type: str) -> Dict[str, float]:
    """
    Generate weight adjustments for model based on intervention feedback.
    
    These weights can be used to fine-tune the LSTM:
    - Features that lead to positive outcomes get higher weight
    - Features that don't help get lower weight
    
    Args:
        job_type: One of the 6 job categories
        
    Returns:
        Dictionary of feature name -> weight adjustment
    """
    analyzer = CohortAnalyzer()
    job_analysis = analyzer.analyze_by_job_type()
    
    if job_type not in job_analysis:
        return {}
    
    job_data = job_analysis[job_type]
    recommendations = job_data["recommendations"]
    
    weights = {}
    
    for feature, rec_data in recommendations.items():
        success_rate = rec_data["success_rate"]
        
        # Weight adjustment: successful features get +0.1 to +0.3
        # Unsuccessful features get -0.1 to -0.2
        if success_rate > 0.7:
            weights[feature] = 0.3
        elif success_rate > 0.5:
            weights[feature] = 0.1
        elif success_rate < 0.3:
            weights[feature] = -0.2
        else:
            weights[feature] = 0  # No change
    
    return weights


# ============================================================================
# REPORTING
# ============================================================================

def generate_effectiveness_report() -> Dict:
    """Generate comprehensive effectiveness report."""
    analyzer = CohortAnalyzer()
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "job_analysis": analyzer.analyze_by_job_type(),
        "best_interventions": analyzer.find_best_interventions(),
    }
    
    # Save report
    with open(EFFECTIVENESS_REPORT, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report


def print_effectiveness_summary():
    """Print summary to console."""
    if not INTERVENTION_LOG.exists():
        print("❌ No intervention data yet. Run predict_burnout.py first.")
        return
    
    analyzer = CohortAnalyzer()
    job_analysis = analyzer.analyze_by_job_type()
    
    print("\n" + "=" * 70)
    print("📊 INTERVENTION EFFECTIVENESS SUMMARY")
    print("=" * 70)
    
    for job_type, data in job_analysis.items():
        print(f"\n🎯 {job_type.upper()}")
        print(f"   Total interventions: {data['total']}")
        print(f"   Success rate: {data['success_rate']:.1%}")
        print(f"   Most effective recommendations:")
        
        recs = sorted(
            data["recommendations"].items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True,
        )[:3]
        
        for feature, rec_data in recs:
            print(f"      • {feature}: {rec_data['success_rate']:.1%} "
                  f"({rec_data['successful']}/{rec_data['count']})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    print_effectiveness_summary()
