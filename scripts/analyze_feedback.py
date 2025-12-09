"""
================================================================================
ANALYZE INTERVENTION FEEDBACK
================================================================================
Analyzes the effectiveness of burnout interventions and generates recommendations.

Run with:
    python scripts/analyze_feedback.py
    python scripts/analyze_feedback.py --user user@example.com
    python scripts/analyze_feedback.py --job teacher
    python scripts/analyze_feedback.py --report

Author: FDS Project Team
================================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add scripts to path
sys.path.append(str(Path(__file__).resolve().parent))

from feedback_loop import (
    InterventionTracker,
    CohortAnalyzer,
    generate_model_update_weights,
    generate_effectiveness_report,
    print_effectiveness_summary,
    FEEDBACK_DIR,
)


def analyze_user_effectiveness(user_id: str) -> None:
    """Analyze effectiveness for a specific user."""
    print(f"\n{'='*70}")
    print(f"📊 EFFECTIVENESS ANALYSIS FOR: {user_id}")
    print(f"{'='*70}")
    
    tracker = InterventionTracker(user_id, "unknown")
    summary = tracker.get_effectiveness_summary()
    
    if summary["total_interventions"] == 0:
        print(f"❌ No interventions found for {user_id}")
        return
    
    print(f"\n📈 Overview:")
    print(f"  • Total interventions: {summary['total_interventions']}")
    print(f"  • With follow-up data: {summary['with_followup']}")
    print(f"  • Average effectiveness: {summary['average_effectiveness']:.1%}")
    
    if summary["by_category"]:
        print(f"\n📋 Effectiveness by advice category:")
        for category, effectiveness in summary["by_category"].items():
            emoji = "✅" if effectiveness > 0.6 else "⚠️"
            print(f"  {emoji} {category}: {effectiveness:.1%}")
    
    # Show interventions
    print(f"\n🔍 Recent interventions:")
    for intervention in tracker.interventions[-5:]:
        ts = intervention["timestamp"][:10]  # Date only
        score = intervention["burnout_score"]
        effectiveness = intervention.get("effectiveness")
        if effectiveness is not None:
            eff_str = f"{effectiveness:.0%}"
        else:
            eff_str = "pending follow-up"
        print(f"  • [{ts}] Score: {score:.1f} → Effectiveness: {eff_str}")


def analyze_job_type(job_type: str) -> None:
    """Analyze what works best for a specific job type."""
    print(f"\n{'='*70}")
    print(f"🎯 EFFECTIVENESS ANALYSIS FOR: {job_type.upper()}")
    print(f"{'='*70}")
    
    analyzer = CohortAnalyzer()
    job_analysis = analyzer.analyze_by_job_type()
    
    if job_type not in job_analysis:
        print(f"❌ No data found for job type: {job_type}")
        return
    
    data = job_analysis[job_type]
    
    print(f"\n📊 Overall stats:")
    print(f"  • Total interventions: {data['total']}")
    print(f"  • Successful (>60% effective): {data['effective']}")
    print(f"  • Success rate: {data['success_rate']:.1%}")
    
    print(f"\n✅ Most effective recommendations:")
    recs = sorted(
        data["recommendations"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True,
    )
    
    for feature, rec_data in recs[:5]:
        print(f"  • {feature}")
        print(f"    Success rate: {rec_data['success_rate']:.1%} "
              f"({rec_data['successful']}/{rec_data['count']} worked)")
    
    # Generate model weights
    weights = generate_model_update_weights(job_type)
    if weights:
        print(f"\n🤖 Suggested model weight adjustments:")
        for feature, weight in sorted(weights.items(), 
                                     key=lambda x: abs(x[1]), 
                                     reverse=True)[:5]:
            direction = "↑" if weight > 0 else "↓"
            print(f"  {direction} {feature}: {weight:+.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze burnout intervention effectiveness"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Analyze specific user (e.g., user@example.com)"
    )
    parser.add_argument(
        "--job",
        type=str,
        choices=["software_engineer", "teacher", "nurse", "manager", 
                "operations", "wellness_coach"],
        help="Analyze job type"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full effectiveness report"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary (default action)"
    )
    
    args = parser.parse_args()
    
    # Check if feedback data exists
    if not FEEDBACK_DIR.exists() or not list(FEEDBACK_DIR.glob("*.jsonl")):
        print("❌ No intervention data found. Run predict_burnout.py first to generate data.")
        return
    
    if args.user:
        analyze_user_effectiveness(args.user)
    elif args.job:
        analyze_job_type(args.job)
    elif args.report:
        report = generate_effectiveness_report()
        print(f"\n✅ Report generated: {FEEDBACK_DIR / 'effectiveness.json'}")
        print(f"   {report['job_analysis']}")
    else:
        # Default: print summary
        print_effectiveness_summary()


if __name__ == "__main__":
    main()
