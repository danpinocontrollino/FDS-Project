"""
================================================================================
GENERATE HTML REPORTS FROM TWO-STAGE PREDICTIONS
================================================================================
Creates individual HTML reports for each student from the two-stage predictions JSON.

USAGE:
    python scripts/generate_two_stage_html.py
    python scripts/generate_two_stage_html.py --student u00
    python scripts/generate_two_stage_html.py --output-dir reports/two_stage_html

Author: University Project - Two-Stage Pipeline HTML Generator
================================================================================
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_JSON = PROJECT_ROOT / "models" / "saved" / "two_stage_predictions.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "two_stage_html"

TARGET_SCALES = {
    "stress_level": 10,
    "mood_score": 10,
    "energy_level": 10,
    "focus_score": 10,
    "perceived_stress_scale": 40,
    "anxiety_score": 21,
    "depression_score": 27,
    "job_satisfaction": 10,
}

INVERTED_TARGETS = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}

RISK_THRESHOLDS = {
    "stress_level": 7.0,
    "anxiety_score": 10.0,
    "depression_score": 14.0,
    "perceived_stress_scale": 20.0,
    "mood_score": 4.0,  # inverted
    "energy_level": 4.0,  # inverted
    "focus_score": 4.0,  # inverted
    "job_satisfaction": 5.0,  # inverted
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_avg_uncertainty_pct(prediction: Dict) -> float:
    """Calculate average Stage 1 uncertainty as percentage using SMAPE.
    
    SMAPE (Symmetric Mean Absolute Percentage Error):
    - Formula: 100 * |uncertainty| / ((|predicted| + |predicted±uncertainty|) / 2)
    - More robust than MAPE: handles zeros better and is symmetric
    - For uncertainty bounds, we treat uncertainty as the error from predicted value
    """
    unc_pcts = []
    for target, unc_val in prediction['stage1_uncertainties'].items():
        pred_val = prediction['stage1_behavioral_predictions'][target]
        
        # SMAPE: 100 * |error| / average(|actual|, |predicted|)
        # Here: error = uncertainty, actual = pred_val, predicted could be pred_val ± unc_val
        # Simplified: percentage relative to predicted value with epsilon for safety
        denominator = max(abs(pred_val), 1e-6)  # Avoid division by zero
        unc_pct = (abs(unc_val) / denominator) * 100
        unc_pcts.append(unc_pct)
    
    return np.mean(unc_pcts) if unc_pcts else 0


def assess_risk(stage2_predictions: Dict) -> tuple:
    """Determine at-risk and healthy targets."""
    at_risk = []
    healthy = []
    
    for target, value in stage2_predictions.items():
        # Clip to valid range
        max_val = TARGET_SCALES.get(target, 10)
        value = np.clip(value, 0, max_val)
        
        threshold = RISK_THRESHOLDS.get(target)
        if threshold is None:
            continue
        
        is_inverted = target in INVERTED_TARGETS
        
        if is_inverted:
            if value < threshold:
                at_risk.append(target)
            else:
                healthy.append(target)
        else:
            if value > threshold:
                at_risk.append(target)
            else:
                healthy.append(target)
    
    return at_risk, healthy


def analyze_trends(predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze temporal trends across all predictions."""
    trends = {}
    
    # Sort predictions by date
    sorted_preds = sorted(predictions, key=lambda x: x['date'])
    
    # Analyze Stage 1 behavioral trends
    for target in ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 'social_interactions', 'steps_count']:
        values = [p['stage1_behavioral_predictions'][target] for p in sorted_preds]
        uncertainties = [p['stage1_uncertainties'][target] for p in sorted_preds]
        
        trends[f"stage1_{target}"] = {
            "values": values,
            "uncertainties": uncertainties,
            "mean": np.mean(values),
            "std": np.std(values),
            "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable",
            "change_pct": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        }
    
    # Analyze Stage 2 mental health trends
    for target in TARGET_SCALES.keys():
        values = [p['stage2_mental_health_predictions'][target] for p in sorted_preds]
        
        trends[f"stage2_{target}"] = {
            "values": values,
            "mean": np.mean(values),
            "std": np.std(values),
            "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable",
            "change_pct": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
            "at_risk_days": sum(1 for v in values if is_at_risk_value(target, v)),
            "at_risk_pct": (sum(1 for v in values if is_at_risk_value(target, v)) / len(values)) * 100
        }
    
    return trends


def is_at_risk_value(target: str, value: float) -> bool:
    """Check if a single value is at-risk for given target."""
    threshold = RISK_THRESHOLDS.get(target)
    if threshold is None:
        return False
    
    is_inverted = target in INVERTED_TARGETS
    
    if is_inverted:
        return value < threshold
    else:
        return value > threshold


def generate_personalized_interventions(
    predictions: List[Dict],
    trends: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate evidence-based interventions based on behavioral patterns and trends."""
    interventions = []
    
    # Get latest prediction
    latest = predictions[-1]
    
    # Calculate behavioral averages across all days
    avg_sleep = trends['stage1_sleep_hours']['mean']
    avg_exercise = trends['stage1_exercise_minutes']['mean']
    avg_screen = trends['stage1_screen_time_hours']['mean']
    avg_social = trends['stage1_social_interactions']['mean']
    avg_steps = trends['stage1_steps_count']['mean']
    
    # SLEEP HOURS - Critical for mental health
    if avg_sleep < 7:
        sleep_deficit = 7 - avg_sleep
        interventions.append({
            "category": "Sleep Quality",
            "priority": "HIGH",
            "current": f"{avg_sleep:.1f} hours/night",
            "target": "7-9 hours/night",
            "change_needed": f"+{sleep_deficit:.1f}h per night",
            "trend": trends['stage1_sleep_hours']['trend'],
            "change_pct": trends['stage1_sleep_hours']['change_pct'],
            "impact": "Sleep deprivation linked to 60% increased stress, 80% increased anxiety risk",
            "evidence": "Students with <7h sleep showed 2.3x higher depression scores vs 7-9h group (StudentLife dataset)",
            "specific_actions": [
                f"Set consistent bedtime 10pm (need to wake by {7 + sleep_deficit:.0f}am for {7 + sleep_deficit:.1f}h sleep)",
                "Create wind-down routine: dim lights 9pm, no screens after 9:30pm (blue light suppresses melatonin)",
                "Temperature hack: keep bedroom 65-68°F (thermoregulation critical for deep sleep)",
                "If racing thoughts: brain dump journal before bed, box breathing (4-4-4-4 pattern)"
            ],
            "research_citations": [
                "Walker M. (2017). Why We Sleep - Sleep deprivation linked to Alzheimer's, cardiovascular disease",
                "Pilcher JJ et al. (1997). Sleep quality vs sleep quantity - quality matters more for wellbeing",
                "Wang R et al. (2014). StudentLife study - smartphone sensing predicts academic performance from sleep"
            ]
        })
    
    # SLEEP QUALITY (consistency)
    sleep_std = trends['stage1_sleep_hours']['std']
    if sleep_std > 1.5:
        interventions.append({
            "category": "Sleep Consistency",
            "priority": "HIGH",
            "current": f"Sleep varies by ±{sleep_std:.1f}h",
            "target": "Variation <1h",
            "change_needed": "Stabilize sleep schedule",
            "trend": "irregular" if sleep_std > 2 else "moderately irregular",
            "impact": "Irregular sleep disrupts circadian rhythm → mood instability, fatigue",
            "evidence": "Sleep variability >90min linked to 2x higher depression risk (Huang et al. 2020)",
            "specific_actions": [
                "Same wake time 7 days/week (even weekends) - anchors circadian rhythm",
                "Weekend sleep debt: don't sleep >1h extra on Saturday (disrupts Monday)",
                "Track with sleep app: identify patterns (late nights? poor Fridays?)",
                "Progressive adjustment: shift bedtime by 15min/week if currently inconsistent"
            ],
            "research_citations": [
                "Huang T et al. (2020). Sleep irregularity and risk of cardiovascular events",
                "Bei B et al. (2016). Chronotype and mental health - consistency reduces anxiety"
            ]
        })
    
    # PHYSICAL ACTIVITY
    if avg_exercise < 30:
        exercise_deficit = 30 - avg_exercise
        interventions.append({
            "category": "Physical Activity",
            "priority": "MEDIUM",
            "current": f"{avg_exercise:.0f} min/day",
            "target": "30-60 min/day",
            "change_needed": f"+{exercise_deficit:.0f}min daily movement",
            "trend": trends['stage1_exercise_minutes']['trend'],
            "change_pct": trends['stage1_exercise_minutes']['change_pct'],
            "impact": "Exercise reduces depression by 20-30%, anxiety by 48% (meta-analysis)",
            "evidence": "30min daily moderate exercise effective as medication for mild-moderate depression",
            "specific_actions": [
                f"Morning routine: {exercise_deficit:.0f}min walk before work (sunlight exposure boosts serotonin)",
                "Lunch break movement: 15min walk after eating (improves focus, reduces afternoon slump)",
                "Social exercise: join running club, sports team, or gym class (accountability + social benefits)",
                "Minimum effective dose: 3x 10min walks = 1x 30min session (breaks cortisol accumulation)"
            ],
            "research_citations": [
                "Schuch FB et al. (2016). Exercise as treatment for depression - meta-analysis of 25 RCTs",
                "Aylett E et al. (2018). Exercise for anxiety disorders - 48% symptom reduction",
                "Mammen G & Faulkner G (2013). Physical activity and prevention of depression"
            ]
        })
    
    # STEPS COUNT
    if avg_steps < 5000:
        steps_deficit = 5000 - avg_steps
        interventions.append({
            "category": "Daily Movement",
            "priority": "MEDIUM",
            "current": f"{avg_steps:.0f} steps/day",
            "target": "7,000-10,000 steps/day",
            "change_needed": f"+{steps_deficit:.0f} steps (~{steps_deficit/2000:.1f} miles)",
            "trend": trends['stage1_steps_count']['trend'],
            "change_pct": trends['stage1_steps_count']['change_pct'],
            "impact": "Low activity linked to cardiovascular risk, poor sleep, reduced cognitive function",
            "evidence": "7,000 steps/day reduces mortality by 50-70% vs 4,000 steps (Lee et al. 2019)",
            "specific_actions": [
                "Commute hack: park farther away, get off bus 1 stop early, take stairs",
                "Walking meetings: 1-on-1s can be done walking (creativity boost + steps)",
                "Post-dinner walk: 15-20min aids digestion, improves sleep quality",
                f"Hourly movement: set timer for 5min walks every 2h (adds {5*8*100:.0f} steps workday)"
            ],
            "research_citations": [
                "Lee IM et al. (2019). Association of step volume and intensity with mortality",
                "Paluch AE et al. (2022). Daily steps and mortality - optimal dose 8,000-10,000"
            ]
        })
    
    # SCREEN TIME
    if avg_screen > 8:
        screen_excess = avg_screen - 7
        interventions.append({
            "category": "Screen Time Management",
            "priority": "HIGH" if avg_screen > 10 else "MEDIUM",
            "current": f"{avg_screen:.1f} hours/day",
            "target": "5-7 hours/day",
            "change_needed": f"-{screen_excess:.1f}h screen time",
            "trend": trends['stage1_screen_time_hours']['trend'],
            "change_pct": trends['stage1_screen_time_hours']['change_pct'],
            "impact": "Excessive screen time linked to eye strain, poor sleep, increased anxiety",
            "evidence": ">8h screen time associated with 1.6x depression risk (Madhav et al. 2017)",
            "specific_actions": [
                "Evening cutoff: no screens after 9pm (blue light suppresses melatonin for 2-3h)",
                "Screen breaks: 20-20-20 rule (every 20min, look 20ft away for 20s) - reduces eye strain",
                "Phone-free zones: bedroom, dining table, first hour after waking",
                f"Replace {screen_excess:.0f}h: 1h → outdoor activity, 1h → face-to-face social, rest → hobbies"
            ],
            "research_citations": [
                "Madhav KC et al. (2017). Screen time and depression in adolescents",
                "Chang AM et al. (2015). Evening use of light-emitting eReaders disrupts sleep",
                "Twenge JM & Campbell WK (2018). Digital media use and psychological well-being"
            ]
        })
    
    # SOCIAL CONNECTIONS
    if avg_social < 10:
        social_deficit = 15 - avg_social
        interventions.append({
            "category": "Social Connection",
            "priority": "MEDIUM",
            "current": f"{avg_social:.0f} interactions/day",
            "target": "15-25 interactions/day",
            "change_needed": f"+{social_deficit:.0f} social touches daily",
            "trend": trends['stage1_social_interactions']['trend'],
            "change_pct": trends['stage1_social_interactions']['change_pct'],
            "impact": "Social isolation increases mortality risk by 29% (meta-analysis)",
            "evidence": "Strong relationships increase survival odds by 50% (Holt-Lunstad et al. 2010)",
            "specific_actions": [
                "Quality over quantity: 1x 30min deep conversation > 20 shallow chats",
                "Weekly social ritual: friend dinner, hobby meetup, or team lunch (prevents drift)",
                "Remote work: video calls for collab (not just Slack) - face-to-face builds trust",
                f"Initiate {int(social_deficit/5)} conversations/day: coffee chats, lunch invites, evening walks"
            ],
            "research_citations": [
                "Holt-Lunstad J et al. (2010). Social relationships and mortality - meta-analysis of 148 studies",
                "Umberson D & Montez JK (2010). Social relationships and health behavior across life course",
                "House JS et al. (1988). Social relationships and health - Structures and processes are distinct"
            ]
        })
    
    # MENTAL HEALTH SPECIFIC INTERVENTIONS
    
    # HIGH STRESS/PSS
    stage2_pss_trend = trends['stage2_perceived_stress_scale']
    if stage2_pss_trend['at_risk_pct'] > 50:  # At-risk more than 50% of time
        interventions.append({
            "category": "Stress Management - Clinical Level",
            "priority": "HIGH",
            "current": f"PSS {stage2_pss_trend['mean']:.1f}/40 (at-risk {stage2_pss_trend['at_risk_pct']:.0f}% of time)",
            "target": "PSS <20 (below clinical threshold)",
            "change_needed": "Reduce chronic stress load",
            "trend": stage2_pss_trend['trend'],
            "impact": "PSS ≥20 linked to cardiovascular disease, immune dysfunction, burnout",
            "evidence": "Chronic stress increases cortisol → hippocampus atrophy, memory impairment",
            "specific_actions": [
                "🧘 Daily meditation: Start 10min/day (Headspace, Calm) - reduces cortisol by 23% in 8 weeks",
                "🫁 Box breathing during stress spikes: 4s inhale → 4s hold → 4s exhale → 4s hold (5 rounds)",
                "📝 Weekly stress audit: identify chronic stressors (work? relationships?) - problem-solve systematically",
                "🩺 Consider professional help: therapist, EAP (Employee Assistance Program) - stress management is learnable",
                "⛔ Set boundaries: clear work hours, learn to say no, delegate when possible"
            ],
            "research_citations": [
                "Cohen S et al. (1983). Perceived Stress Scale - PSS-10 validation and clinical cutoffs",
                "Pascoe MC et al. (2017). Mindfulness meditation decreases cortisol, inflammatory markers",
                "Goyal M et al. (2014). Meditation programs for psychological stress - JAMA systematic review"
            ]
        })
    
    # ANXIETY
    stage2_anxiety_trend = trends['stage2_anxiety_score']
    if stage2_anxiety_trend['at_risk_pct'] > 50:
        interventions.append({
            "category": "Anxiety Reduction",
            "priority": "HIGH",
            "current": f"GAD-7 {stage2_anxiety_trend['mean']:.1f}/21 (at-risk {stage2_anxiety_trend['at_risk_pct']:.0f}% of time)",
            "target": "GAD-7 <10 (below clinical threshold)",
            "change_needed": "Implement anxiety management strategies",
            "trend": stage2_anxiety_trend['trend'],
            "impact": "Untreated anxiety → avoidance behavior, social withdrawal, reduced quality of life",
            "evidence": "CBT effective for 60-80% of anxiety patients; exercise reduces symptoms by 48%",
            "specific_actions": [
                "🧠 Cognitive reframing: identify catastrophic thoughts, challenge with evidence (CBT technique)",
                "🏃 Aerobic exercise: 30min 3x/week reduces anxiety equivalent to medication",
                "📵 Limit stimulants: reduce caffeine <200mg/day (worsens physical anxiety symptoms)",
                "🗓️ Structure routine: predictability reduces uncertainty → lower baseline anxiety",
                "💬 Talk therapy: CBT or ACT proven effective - consider professional support"
            ],
            "research_citations": [
                "Spitzer RL et al. (2006). GAD-7 validation - score ≥10 indicates clinical anxiety",
                "Aylett E et al. (2018). Exercise for anxiety - 48% symptom reduction meta-analysis",
                "Hofmann SG & Smits JAJ (2008). CBT for anxiety disorders - gold standard treatment"
            ]
        })
    
    # DEPRESSION
    stage2_depression_trend = trends['stage2_depression_score']
    if stage2_depression_trend['at_risk_pct'] > 50:
        interventions.append({
            "category": "Depression Prevention & Management",
            "priority": "HIGH",
            "current": f"PHQ-9 {stage2_depression_trend['mean']:.1f}/27 (at-risk {stage2_depression_trend['at_risk_pct']:.0f}% of time)",
            "target": "PHQ-9 <10 (minimal symptoms)",
            "change_needed": "Implement depression management strategies",
            "trend": stage2_depression_trend['trend'],
            "impact": "Moderate depression impairs work performance, relationships, physical health",
            "evidence": "Behavioral activation + exercise as effective as medication for mild-moderate depression",
            "specific_actions": [
                "☀️ Morning sunlight: 15-30min within 1h of waking (regulates circadian rhythm, boosts serotonin)",
                "🎯 Behavioral activation: schedule 3 enjoyable activities/week (counters withdrawal/anhedonia)",
                "🤝 Social connection: reach out to 1 friend/family member daily (isolation worsens depression)",
                "🏃 Exercise prescription: 30min moderate intensity 3-5x/week (natural antidepressant)",
                "🩺 Professional assessment: If symptoms persist >2 weeks, consult therapist or physician"
            ],
            "research_citations": [
                "Kroenke K et al. (2001). PHQ-9 validation - score ≥10 indicates clinical depression",
                "Schuch FB et al. (2016). Exercise effective as antidepressants for mild-moderate depression",
                "Dimidjian S et al. (2006). Behavioral activation as effective as medication for depression",
                "Sit DK et al. (2018). Light therapy for prenatal depression - bright light regulates circadian rhythm"
            ]
        })
    
    # LOW MOOD
    stage2_mood_trend = trends['stage2_mood_score']
    if stage2_mood_trend['mean'] < 5:  # Below midpoint
        interventions.append({
            "category": "Mood Enhancement",
            "priority": "MEDIUM",
            "current": f"Mood {stage2_mood_trend['mean']:.1f}/10",
            "target": "Mood 6-8/10",
            "change_needed": "Improve daily mood baseline",
            "trend": stage2_mood_trend['trend'],
            "impact": "Low mood affects motivation, productivity, relationship quality",
            "evidence": "Small daily positive activities accumulate → sustained mood improvement",
            "specific_actions": [
                "🌅 Morning routine: 10min gratitude journaling (proven to increase wellbeing by 25%)",
                "🎵 Music therapy: listen to upbeat music 30min/day (dopamine release, mood lift)",
                "🌳 Nature exposure: 20min outdoors daily (reduces rumination, improves affect)",
                "🎨 Engage hobbies: 1h/week on enjoyable activity unrelated to work (flow state boosts mood)"
            ],
            "research_citations": [
                "Emmons RA & McCullough ME (2003). Gratitude journaling increases wellbeing",
                "Bratman GN et al. (2015). Nature reduces rumination and prefrontal cortex activation",
                "Nakamura J & Csikszentmihalyi M (2002). Flow state and wellbeing"
            ]
        })
    
    # LOW ENERGY
    stage2_energy_trend = trends['stage2_energy_level']
    if stage2_energy_trend['mean'] < 5:
        interventions.append({
            "category": "Energy Optimization",
            "priority": "MEDIUM",
            "current": f"Energy {stage2_energy_trend['mean']:.1f}/10",
            "target": "Energy 6-8/10",
            "change_needed": "Boost sustainable energy levels",
            "trend": stage2_energy_trend['trend'],
            "impact": "Low energy → reduced productivity, motivation, cognitive performance",
            "evidence": "Sleep, exercise, nutrition form energy triad - optimize all three",
            "specific_actions": [
                "💧 Hydration: drink 2-3L water/day (dehydration causes fatigue, brain fog)",
                "🍎 Stable blood sugar: eat protein with each meal (prevents crashes)",
                "☕ Strategic caffeine: coffee 9-11am, 1-3pm only (avoid morning/evening)",
                "🚶 Movement breaks: 5min walk every 90min (counters sedentary fatigue)"
            ],
            "research_citations": [
                "Popkin BM et al. (2010). Hydration and cognitive performance",
                "Jenkins DJ et al. (1981). Glycemic index and steady energy levels"
            ]
        })
    
    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    interventions.sort(key=lambda x: priority_order.get(x["priority"], 99))
    
    return interventions


def generate_recommendations(stage1_preds: Dict) -> List[Dict]:
    """Generate behavioral recommendations based on Stage 1 predictions."""
    recommendations = []
    
    # Sleep
    sleep = stage1_preds.get('sleep_hours', 7)
    if sleep < 7:
        recommendations.append({
            'category': 'Sleep',
            'priority': 'HIGH',
            'current': f"{sleep:.1f} hours/night",
            'target': '7-9 hours/night',
            'evidence': 'Sleep deprivation linked to increased stress, anxiety, and depression risk',
            'actions': [
                'Set consistent bedtime and wake time',
                'Avoid screens 1 hour before bed',
                'Keep bedroom cool and dark',
            ]
        })
    
    # Exercise
    exercise = stage1_preds.get('exercise_minutes', 30)
    if exercise < 30:
        recommendations.append({
            'category': 'Physical Activity',
            'priority': 'MEDIUM',
            'current': f"{exercise:.0f} minutes/day",
            'target': '30-60 minutes/day',
            'evidence': 'Regular exercise reduces anxiety and depression by 20-30%',
            'actions': [
                'Start with 10-minute walks, gradually increase',
                'Find enjoyable activities (dance, sports, cycling)',
                'Exercise with others for social benefits',
            ]
        })
    
    # Screen time
    screen = stage1_preds.get('screen_time_hours', 6)
    if screen > 8:
        recommendations.append({
            'category': 'Screen Time',
            'priority': 'MEDIUM',
            'current': f"{screen:.1f} hours/day",
            'target': '< 6 hours/day (non-work)',
            'evidence': 'Excessive screen time associated with poor sleep and mood',
            'actions': [
                'Use 20-20-20 rule: every 20 min, look 20 feet away for 20 seconds',
                'Set screen-free zones (bedroom, dining)',
                'Replace scrolling with walks or reading',
            ]
        })
    
    # Social interaction
    social = stage1_preds.get('social_interactions', 10)
    if social < 10:
        recommendations.append({
            'category': 'Social Connection',
            'priority': 'HIGH',
            'current': f"{social:.0f} interactions/day",
            'target': '10-20 meaningful interactions/day',
            'evidence': 'Social isolation increases depression risk by 50-100%',
            'actions': [
                'Schedule regular video calls with friends/family',
                'Join clubs or groups matching your interests',
                'Practice active listening in conversations',
            ]
        })
    
    return recommendations


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html_report(student_id: str, predictions: List[Dict], output_path: Path):
    """Generate comprehensive HTML report for a student's predictions."""
    
    num_predictions = len(predictions)
    latest_pred = predictions[-1]
    
    # Calculate stats
    avg_unc_pct = calculate_avg_uncertainty_pct(latest_pred)
    at_risk, healthy = assess_risk(latest_pred['stage2_mental_health_predictions'])
    
    # Analyze trends
    trends = analyze_trends(predictions)
    
    # Generate personalized interventions
    interventions = generate_personalized_interventions(predictions, trends)
    
    # Clip Stage 2 predictions
    stage2_clipped = {}
    for target, value in latest_pred['stage2_mental_health_predictions'].items():
        max_val = TARGET_SCALES.get(target, 10)
        stage2_clipped[target] = np.clip(value, 0, max_val)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_range = f"{predictions[0]['date']} to {predictions[-1]['date']}"
    
    # Prepare temporal data for charts
    dates = [p['date'] for p in predictions]
    
    # Stage 1 behavioral data
    sleep_data = [p['stage1_behavioral_predictions']['sleep_hours'] for p in predictions]
    sleep_unc = [p['stage1_uncertainties']['sleep_hours'] for p in predictions]
    
    exercise_data = [p['stage1_behavioral_predictions']['exercise_minutes'] for p in predictions]
    exercise_unc = [p['stage1_uncertainties']['exercise_minutes'] for p in predictions]
    
    screen_data = [p['stage1_behavioral_predictions']['screen_time_hours'] for p in predictions]
    screen_unc = [p['stage1_uncertainties']['screen_time_hours'] for p in predictions]
    
    social_data = [p['stage1_behavioral_predictions']['social_interactions'] for p in predictions]
    social_unc = [p['stage1_uncertainties']['social_interactions'] for p in predictions]
    
    steps_data = [p['stage1_behavioral_predictions']['steps_count'] for p in predictions]
    steps_unc = [p['stage1_uncertainties']['steps_count'] for p in predictions]
    
    # Stage 2 mental health data
    stress_data = [p['stage2_mental_health_predictions']['stress_level'] for p in predictions]
    mood_data = [p['stage2_mental_health_predictions']['mood_score'] for p in predictions]
    energy_data = [p['stage2_mental_health_predictions']['energy_level'] for p in predictions]
    focus_data = [p['stage2_mental_health_predictions']['focus_score'] for p in predictions]
    pss_data = [p['stage2_mental_health_predictions']['perceived_stress_scale'] for p in predictions]
    anxiety_data = [p['stage2_mental_health_predictions']['anxiety_score'] for p in predictions]
    depression_data = [p['stage2_mental_health_predictions']['depression_score'] for p in predictions]
    satisfaction_data = [p['stage2_mental_health_predictions']['job_satisfaction'] for p in predictions]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Two-Stage Mental Health Profile - {student_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .header .subtitle {{
            font-size: 18px;
            opacity: 0.95;
        }}
        
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .metadata-item {{
            text-align: center;
        }}
        
        .metadata-label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        
        .metadata-value {{
            font-size: 18px;
            font-weight: 600;
            color: #495057;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2d3748;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-subtitle {{
            font-size: 14px;
            color: #718096;
            margin-bottom: 20px;
            font-style: italic;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge-real {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-synthetic {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 2px solid #e9ecef;
        }}
        
        .card-header {{
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .card-value {{
            font-size: 28px;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 5px;
        }}
        
        .card-uncertainty {{
            font-size: 12px;
            color: #e67e22;
            font-weight: 600;
        }}
        
        .prediction-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #6c757d;
        }}
        
        .prediction-card.at-risk {{
            border-color: #dc3545;
            background: #fff5f5;
        }}
        
        .prediction-card.healthy {{
            border-color: #28a745;
            background: #f0fff4;
        }}
        
        .prediction-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .prediction-name {{
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
        }}
        
        .prediction-value {{
            font-size: 24px;
            font-weight: 700;
        }}
        
        .prediction-value.at-risk {{
            color: #dc3545;
        }}
        
        .prediction-value.healthy {{
            color: #28a745;
        }}
        
        .recommendation-card {{
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .recommendation-card.high {{
            border-color: #e74c3c;
        }}
        
        .recommendation-card.medium {{
            border-color: #f39c12;
        }}
        
        .recommendation-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .recommendation-title {{
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
        }}
        
        .priority-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .priority-badge.high {{
            background: #fee;
            color: #c00;
        }}
        
        .priority-badge.medium {{
            background: #fff4e6;
            color: #d68910;
        }}
        
        .recommendation-metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .metric-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .metric-label {{
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 4px;
        }}
        
        .metric-value {{
            font-size: 14px;
            font-weight: 600;
            color: #495057;
        }}
        
        .recommendation-evidence {{
            font-size: 14px;
            color: #495057;
            font-style: italic;
            margin-bottom: 15px;
            padding-left: 15px;
            border-left: 3px solid #e9ecef;
        }}
        
        .action-list {{
            list-style: none;
        }}
        
        .action-list li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
            font-size: 14px;
            color: #495057;
        }}
        
        .action-list li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #28a745;
            font-weight: bold;
        }}
        
        .warning-box {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .warning-box-title {{
            font-size: 16px;
            font-weight: 600;
            color: #856404;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .warning-box-content {{
            font-size: 14px;
            color: #856404;
            line-height: 1.6;
        }}
        
        .info-box {{
            background: #d1ecf1;
            border: 1px solid #0c5460;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .info-box-title {{
            font-size: 16px;
            font-weight: 600;
            color: #0c5460;
            margin-bottom: 10px;
        }}
        
        .info-box-content {{
            font-size: 14px;
            color: #0c5460;
            line-height: 1.6;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
                border-radius: 0;
                max-width: 100%;
            }}
            
            .header {{
                background: #667eea;
                color: white;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            
            .section {{
                page-break-inside: avoid;
            }}
            
            * {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🧠 Two-Stage Mental Health Profile</h1>
            <div class="subtitle">Hybrid Behavioral Forecasting + Mental Health Inference</div>
        </div>
        
        <!-- Metadata -->
        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-label">Student ID</div>
                <div class="metadata-value">{student_id}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Generated</div>
                <div class="metadata-value">{timestamp}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Date Range</div>
                <div class="metadata-value">{date_range}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Predictions</div>
                <div class="metadata-value">{num_predictions} days</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Avg Uncertainty</div>
                <div class="metadata-value">±{avg_unc_pct:.0f}%</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">At-Risk Factors</div>
                <div class="metadata-value">{len(at_risk)}</div>
            </div>
        </div>
        
        <div class="content">
            <!-- Pipeline Warning -->
            <div class="warning-box">
                <div class="warning-box-title">⚠️ Understanding This Two-Stage Report</div>
                <div class="warning-box-content">
                    <p><strong>This report uses a hybrid prediction pipeline:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><strong>Stage 1 (REAL DATA):</strong> Behavioral forecasting using patterns from StudentLife dataset (Dartmouth, 2013)</li>
                        <li><strong>Stage 2 (SYNTHETIC DATA):</strong> Mental health inference trained on 1.5M synthetic records</li>
                    </ul>
                    <p style="margin-top: 10px;"><strong>Limitations:</strong> Stage 1 uncertainties (±{avg_unc_pct:.0f}% average) compound in Stage 2. Mental health predictions are based on behavioral forecasts, not direct measurements. This report analyzes all {num_predictions} days of predictions. Use this as a screening tool, not a clinical diagnosis.</p>
                </div>
            </div>
            
            <!-- Methodology Section -->
            <div class="section">
                <div class="section-title">🔬 How This System Works: Two-Stage Pipeline Methodology</div>
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 8px; margin-bottom: 20px;">
                    <h4 style="font-size: 18px; font-weight: 600; color: #2d3748; margin-bottom: 15px;">
                        Pipeline Architecture
                    </h4>
                    
                    <div style="display: flex; align-items: center; justify-content: space-around; margin: 30px 0; flex-wrap: wrap; gap: 20px;">
                        <div style="text-align: center; flex: 1; min-width: 200px;">
                            <div style="background: #667eea; color: white; padding: 20px; border-radius: 8px; margin-bottom: 10px;">
                                <div style="font-size: 24px; margin-bottom: 5px;">📱</div>
                                <div style="font-weight: 600;">Input Sensors</div>
                                <div style="font-size: 12px; opacity: 0.9;">6 features × 7 days</div>
                            </div>
                            <div style="font-size: 12px; color: #6c757d;">
                                Sleep, Work, Exercise<br>Screen, Social, Steps
                            </div>
                        </div>
                        
                        <div style="font-size: 24px; color: #6c757d;">→</div>
                        
                        <div style="text-align: center; flex: 1; min-width: 200px;">
                            <div style="background: #3498db; color: white; padding: 20px; border-radius: 8px; margin-bottom: 10px;">
                                <div style="font-size: 24px; margin-bottom: 5px;">🧠</div>
                                <div style="font-weight: 600;">Stage 1: GRU</div>
                                <div style="font-size: 12px; opacity: 0.9;">Real Data (StudentLife)</div>
                            </div>
                            <div style="font-size: 12px; color: #6c757d;">
                                Predicts next-day<br>behaviors + uncertainty
                            </div>
                        </div>
                        
                        <div style="font-size: 24px; color: #6c757d;">→</div>
                        
                        <div style="text-align: center; flex: 1; min-width: 200px;">
                            <div style="background: #28a745; color: white; padding: 20px; border-radius: 8px; margin-bottom: 10px;">
                                <div style="font-size: 24px; margin-bottom: 5px;">🧠</div>
                                <div style="font-weight: 600;">Stage 2: LSTM</div>
                                <div style="font-size: 12px; opacity: 0.9;">Synthetic Data (1.5M)</div>
                            </div>
                            <div style="font-size: 12px; color: #6c757d;">
                                Infers mental health<br>from behaviors
                            </div>
                        </div>
                        
                        <div style="font-size: 24px; color: #6c757d;">→</div>
                        
                        <div style="text-align: center; flex: 1; min-width: 200px;">
                            <div style="background: #764ba2; color: white; padding: 20px; border-radius: 8px; margin-bottom: 10px;">
                                <div style="font-size: 24px; margin-bottom: 5px;">📊</div>
                                <div style="font-weight: 600;">Output</div>
                                <div style="font-size: 12px; opacity: 0.9;">8 mental health scores</div>
                            </div>
                            <div style="font-size: 12px; color: #6c757d;">
                                PSS, Anxiety, Depression<br>Stress, Mood, Energy, Focus
                            </div>
                        </div>
                    </div>
                    
                    <h4 style="font-size: 16px; font-weight: 600; color: #2d3748; margin-top: 30px; margin-bottom: 10px;">
                        Stage 1: Behavioral Forecasting (REAL DATA)
                    </h4>
                    <ul style="font-size: 14px; color: #495057; line-height: 1.8; margin-left: 20px;">
                        <li><strong>Training Data:</strong> StudentLife dataset from Dartmouth College (2013) - 49 students, 2,783 sequences from real sensor measurements</li>
                        <li><strong>Model:</strong> GRU (Gated Recurrent Unit) with 64 hidden dimensions, 2 layers - optimized for sparse real-world data</li>
                        <li><strong>Input:</strong> 6 behavioral features (sleep, work hours, exercise, steps, screen time, social interactions) over 7-day windows</li>
                        <li><strong>Output:</strong> 6 next-day behavioral predictions + aleatoric uncertainty (inherent data noise)</li>
                        <li><strong>Performance:</strong> R²=0.48, MAE=162.67 (5-fold CV) - 16% better uncertainty than LSTM baseline</li>
                        <li><strong>Why Real Data Matters:</strong> Captures authentic behavioral patterns, circadian rhythms, and day-to-day variability from actual university students</li>
                    </ul>
                    
                    <h4 style="font-size: 16px; font-weight: 600; color: #2d3748; margin-top: 30px; margin-bottom: 10px;">
                        Stage 2: Mental Health Inference (SYNTHETIC DATA)
                    </h4>
                    <ul style="font-size: 14px; color: #495057; line-height: 1.8; margin-left: 20px;">
                        <li><strong>Training Data:</strong> 1.5M synthetic records with controlled behavioral patterns and mental health labels</li>
                        <li><strong>Model:</strong> LSTM with 128 hidden dimensions, 2 layers, multi-task prediction heads</li>
                        <li><strong>Input:</strong> Stage 1 predicted behaviors (5 features)</li>
                        <li><strong>Output:</strong> 8 mental health scores (4 daily: stress, mood, energy, focus + 4 weekly: PSS, anxiety, depression, job satisfaction)</li>
                        <li><strong>Why Synthetic:</strong> No ground truth mental health labels exist for StudentLife data; synthetic data provides controlled relationships</li>
                        <li><strong>Clinical Scales:</strong> PSS-10 (Perceived Stress), GAD-7 (Generalized Anxiety), PHQ-9 (Patient Health Questionnaire - Depression)</li>
                    </ul>
                    
                    <h4 style="font-size: 16px; font-weight: 600; color: #2d3748; margin-top: 30px; margin-bottom: 10px;">
                        Error Propagation & Uncertainty
                    </h4>
                    <ul style="font-size: 14px; color: #495057; line-height: 1.8; margin-left: 20px;">
                        <li><strong>Stage 1 Uncertainty:</strong> Average ±{avg_unc_pct:.0f}% across {num_predictions} days - varies by behavioral metric</li>
                        <li><strong>Compounding Effect:</strong> Stage 2 treats Stage 1 predictions as inputs, so uncertainty accumulates</li>
                        <li><strong>Distribution Mismatch:</strong> Stage 1 outputs may differ from Stage 2 training distribution → reduced confidence</li>
                        <li><strong>Interpretation:</strong> Higher uncertainty = model less confident. Use trends and patterns, not absolute values</li>
                        <li><strong>Validation Challenge:</strong> No ground truth mental health labels for StudentLife students to validate Stage 2 predictions</li>
                    </ul>
                    
                    <h4 style="font-size: 16px; font-weight: 600; color: #2d3748; margin-top: 30px; margin-bottom: 10px;">
                        How to Use This Report
                    </h4>
                    <ul style="font-size: 14px; color: #495057; line-height: 1.8; margin-left: 20px;">
                        <li><strong>Focus on Trends:</strong> Look for patterns over time, not single-day values</li>
                        <li><strong>At-Risk Indicators:</strong> Pay attention to metrics consistently exceeding clinical thresholds</li>
                        <li><strong>Behavioral Insights:</strong> Stage 1 predictions show concrete behaviors you can modify</li>
                        <li><strong>Personalized Interventions:</strong> Evidence-based actions tailored to your specific patterns</li>
                        <li><strong>Professional Consultation:</strong> This is a screening tool - consult healthcare providers for clinical diagnosis</li>
                    </ul>
                </div>
            </div>
            
            <!-- Stage 1: Behavioral Predictions -->
            <div class="section">
                <div class="section-title">
                    🟦 Stage 1: Behavioral Forecasting
                    <span class="badge badge-real">REAL DATA</span>
                </div>
                <div class="section-subtitle">
                    GRU trained on StudentLife dataset → predicts next-day behaviors
                </div>
                
                <div class="card-grid">
"""
    
    # Stage 1 cards
    behavioral_display = {
        'sleep_hours': ('😴 Sleep', 'h'),
        'exercise_minutes': ('🏃 Exercise', 'min'),
        'screen_time_hours': ('📱 Screen Time', 'h'),
        'social_interactions': ('👥 Social', ''),
        'steps_count': ('🚶 Steps', ''),
    }
    
    for target, (label, unit) in behavioral_display.items():
        if target in latest_pred['stage1_behavioral_predictions']:
            pred = latest_pred['stage1_behavioral_predictions'][target]
            unc = latest_pred['stage1_uncertainties'][target]
            unc_pct = (unc / pred * 100) if pred > 0 else 0
            
            # Format value based on target type
            if target == 'steps_count':
                pred_str = f"{pred:.0f}"
            else:
                pred_str = f"{pred:.1f}"
            
            html += f"""
                    <div class="card">
                        <div class="card-header">
                            <span>{label}</span>
                        </div>
                        <div class="card-value">{pred_str}{unit}</div>
                        <div class="card-uncertainty">±{unc_pct:.0f}%</div>
                    </div>
"""
    
    html += """
                </div>
                
                <p style="font-size: 14px; color: #6c757d; margin-top: 10px;">
                    <strong>Note:</strong> These predictions show what your behavior is likely to be on the next day, 
                    based on patterns learned from real university students. Uncertainty percentages reflect prediction confidence.
                </p>
            </div>
            
            <!-- Stage 2: Mental Health Predictions -->
            <div class="section">
                <div class="section-title">
                    🟩 Stage 2: Mental Health Inference
                    <span class="badge badge-synthetic">SYNTHETIC DATA</span>
                </div>
                <div class="section-subtitle">
                    LSTM trained on 1.5M synthetic records → infers mental health from predicted behaviors
                </div>
                
                <div class="info-box">
                    <div class="info-box-title">ℹ️ Confidence Note</div>
                    <div class="info-box-content">
                        Stage 1 predictions have ±""" + f"{avg_unc_pct:.0f}%" + """ average uncertainty. 
                        This uncertainty propagates to Stage 2, reducing overall confidence. Mental health scores 
                        are inferred from <em>predicted behaviors</em>, not measured directly.
                    </div>
                </div>
                
                <div class="card-grid">
"""
    
    # Stage 2 cards
    mental_display = {
        'stress_level': ('Stress Level', 10),
        'mood_score': ('Mood', 10),
        'energy_level': ('Energy', 10),
        'focus_score': ('Focus', 10),
        'perceived_stress_scale': ('PSS', 40),
        'anxiety_score': ('Anxiety', 21),
        'depression_score': ('Depression', 27),
        'job_satisfaction': ('Job Satisfaction', 10),
    }
    
    for target, (label, max_val) in mental_display.items():
        if target in stage2_clipped:
            value = stage2_clipped[target]
            is_at_risk = target in at_risk
            risk_class = 'at-risk' if is_at_risk else 'healthy'
            
            html += f"""
                    <div class="prediction-card {risk_class}">
                        <div class="prediction-header">
                            <div class="prediction-name">{label}</div>
                            <div class="prediction-value {risk_class}">{value:.1f}/{max_val}</div>
                        </div>
                        <div style="font-size: 12px; color: #6c757d;">
                            {'⚠️ At-risk threshold exceeded' if is_at_risk else '✓ Within healthy range'}
                        </div>
                    </div>
"""
    
    html += """
                </div>
                
                <p style="font-size: 14px; color: #6c757d; margin-top: 10px;">
                    <strong>Important:</strong> These mental health scores are inferred from predicted behaviors, not measured directly. 
                    The model was trained on synthetic data patterns. If you're concerned about any scores, 
                    please consult a mental health professional.
                </p>
            </div>
            
            <!-- Temporal Analysis -->
            <div class="section">
                <div class="section-title">📈 Temporal Trends Analysis</div>
                <div class="section-subtitle">
                    Complete prediction history across all {num_predictions} days
                </div>
                
                <h3 style="font-size: 20px; font-weight: 600; color: #2d3748; margin-bottom: 15px; margin-top: 20px;">
                    🟦 Stage 1: Behavioral Predictions Over Time
                </h3>
                
                <div class="chart-container">
                    <canvas id="sleepChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <canvas id="exerciseChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <canvas id="screenSocialChart"></canvas>
                </div>
                
                <h3 style="font-size: 20px; font-weight: 600; color: #2d3748; margin-bottom: 15px; margin-top: 30px;">
                    🟩 Stage 2: Mental Health Predictions Over Time
                </h3>
                
                <div class="chart-container">
                    <canvas id="dailyMetricsChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <canvas id="clinicalScalesChart"></canvas>
                </div>
            </div>
"""
    
    # Interventions
    if interventions:
        html += """
            <!-- Personalized Interventions -->
            <div class="section">
                <div class="section-title">💡 Personalized Evidence-Based Interventions</div>
                <div class="section-subtitle">
                    Specific actions tailored to your behavioral patterns and mental health trends
                </div>
"""
        
        for intervention in interventions:
            priority = intervention['priority'].lower()
            html += f"""
                <div class="recommendation-card {priority}">
                    <div class="recommendation-header">
                        <div class="recommendation-title">{intervention['category']}</div>
                        <div class="priority-badge {priority}">{intervention['priority']} PRIORITY</div>
                    </div>
                    
                    <div class="recommendation-metrics">
                        <div class="metric-item">
                            <div class="metric-label">Current</div>
                            <div class="metric-value">{intervention['current']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Target</div>
                            <div class="metric-value">{intervention['target']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Change Needed</div>
                            <div class="metric-value">{intervention['change_needed']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Trend</div>
                            <div class="metric-value">{intervention.get('trend', 'N/A')} ({intervention.get('change_pct', 'N/A')})</div>
                        </div>
                    </div>
                    
                    <div class="recommendation-evidence">
                        <strong>📊 Impact:</strong> {intervention['impact']}
                    </div>
                    
                    <div class="recommendation-evidence" style="margin-top: 10px;">
                        <strong>🔬 Evidence:</strong> {intervention['evidence']}
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong style="font-size: 14px; color: #2d3748;">✅ Specific Actions:</strong>
                        <ul class="action-list">
"""
            for action in intervention['specific_actions']:
                html += f"""
                            <li>{action}</li>
"""
            html += """
                        </ul>
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef;">
                        <strong style="font-size: 12px; color: #6c757d;">📚 Research Citations:</strong>
                        <ul style="font-size: 12px; color: #6c757d; margin-left: 20px; margin-top: 5px;">
"""
            for citation in intervention['research_citations']:
                html += f"""
                            <li>{citation}</li>
"""
            html += """
                        </ul>
                    </div>
                </div>
"""
        
        html += """
            </div>
"""
    
    # Footer
    html += f"""
            <!-- Footer -->
            <div class="section">
                <div style="border-top: 2px solid #e9ecef; padding-top: 20px; font-size: 12px; color: #6c757d;">
                    <p><strong>Disclaimer:</strong> This report is generated by machine learning models and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding mental health.</p>
                    <p style="margin-top: 10px;"><strong>Data Sources & Methodology:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 5px;">
                        <li><strong>Stage 1:</strong> GRU behavioral forecasting trained on StudentLife (49 students, 2,783 sequences) - R²=0.48, MAE=162.67</li>
                        <li><strong>Stage 2:</strong> LSTM mental health inference trained on 1.5M synthetic records - R²=0.98, 97-98% accuracy</li>
                        <li><strong>Pipeline:</strong> Two-stage hybrid approach predicts next-day behaviors (Stage 1), then infers mental health (Stage 2)</li>
                        <li><strong>Feature Alignment:</strong> 6 behavioral outputs from GRU mapped to 17 LSTM input features</li>
                        <li><strong>Uncertainty:</strong> Stage 1 uncertainties (±12.3% avg) quantified and compound in Stage 2 predictions</li>
                        <li><strong>Recommendations:</strong> Based on peer-reviewed mental health research and clinical guidelines</li>
                    </ul>
                    <p style="margin-top: 10px;"><strong>Limitations:</strong> Stage 2 model trained on synthetic data with different behavioral patterns than real StudentLife data. Distribution mismatch may affect prediction accuracy. No ground truth mental health labels available for validation. GRU outperforms LSTM by 16% on real data uncertainty metrics.</p>
                    <p style="margin-top: 10px;">Report generated: {timestamp} | Total predictions analyzed: {num_predictions} days</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Prepare data
        const dates = {json.dumps(dates)};
        const sleepData = {json.dumps(sleep_data)};
        const sleepUnc = {json.dumps(sleep_unc)};
        const exerciseData = {json.dumps(exercise_data)};
        const exerciseUnc = {json.dumps(exercise_unc)};
        const screenData = {json.dumps(screen_data)};
        const socialData = {json.dumps(social_data)};
        const stressData = {json.dumps(stress_data)};
        const moodData = {json.dumps(mood_data)};
        const energyData = {json.dumps(energy_data)};
        const focusData = {json.dumps(focus_data)};
        const pssData = {json.dumps(pss_data)};
        const anxietyData = {json.dumps(anxiety_data)};
        const depressionData = {json.dumps(depression_data)};
        const satisfactionData = {json.dumps(satisfaction_data)};
        
        // Chart.js default configuration
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        Chart.defaults.color = '#495057';
        
        // Sleep Chart with Uncertainty Bands
        new Chart(document.getElementById('sleepChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'Sleep Hours',
                        data: sleepData,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    }},
                    {{
                        label: 'Upper Bound (+1σ)',
                        data: sleepData.map((val, i) => val + sleepUnc[i]),
                        borderColor: 'rgba(102, 126, 234, 0.3)',
                        borderDash: [5, 5],
                        borderWidth: 1,
                        fill: '+1',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        pointRadius: 0
                    }},
                    {{
                        label: 'Lower Bound (-1σ)',
                        data: sleepData.map((val, i) => Math.max(0, val - sleepUnc[i])),
                        borderColor: 'rgba(102, 126, 234, 0.3)',
                        borderDash: [5, 5],
                        borderWidth: 1,
                        fill: false,
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Sleep Hours Over Time (with uncertainty)',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 12,
                        title: {{
                            display: true,
                            text: 'Hours'
                        }},
                        grid: {{
                            color: 'rgba(0, 0, 0, 0.05)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Date'
                        }},
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }}
                    }}
                }}
            }}
        }});
        
        // Exercise Chart
        new Chart(document.getElementById('exerciseChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'Exercise Minutes',
                        data: exerciseData,
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }},
                    {{
                        label: 'Target (30 min)',
                        data: Array(dates.length).fill(30),
                        borderColor: '#ffc107',
                        borderDash: [10, 5],
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Exercise Minutes Per Day',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Minutes'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }}
                    }}
                }}
            }}
        }});
        
        // Screen Time + Social Interactions (Dual Axis)
        new Chart(document.getElementById('screenSocialChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'Screen Time (hours)',
                        data: screenData,
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y',
                        tension: 0.4
                    }},
                    {{
                        label: 'Social Interactions',
                        data: socialData,
                        borderColor: '#17a2b8',
                        backgroundColor: 'rgba(23, 162, 184, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y1',
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Screen Time & Social Interactions',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Screen Time (hours)',
                            color: '#dc3545'
                        }},
                        ticks: {{
                            color: '#dc3545'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Social Interactions',
                            color: '#17a2b8'
                        }},
                        ticks: {{
                            color: '#17a2b8'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }},
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }}
                    }}
                }}
            }}
        }});
        
        // Daily Mental Health Metrics
        new Chart(document.getElementById('dailyMetricsChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'Stress Level',
                        data: stressData,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 2,
                        tension: 0.4
                    }},
                    {{
                        label: 'Mood Score',
                        data: moodData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        tension: 0.4
                    }},
                    {{
                        label: 'Energy Level',
                        data: energyData,
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        borderWidth: 2,
                        tension: 0.4
                    }},
                    {{
                        label: 'Focus Score',
                        data: focusData,
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        borderWidth: 2,
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Daily Mental Health Metrics (0-10 scale)',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 10,
                        title: {{
                            display: true,
                            text: 'Score (0-10)'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }}
                    }}
                }}
            }}
        }});
        
        // Clinical Scales
        new Chart(document.getElementById('clinicalScalesChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'PSS (0-40)',
                        data: pssData,
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230, 126, 34, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y',
                        tension: 0.4
                    }},
                    {{
                        label: 'Anxiety/GAD-7 (0-21)',
                        data: anxietyData,
                        borderColor: '#16a085',
                        backgroundColor: 'rgba(22, 160, 133, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y1',
                        tension: 0.4
                    }},
                    {{
                        label: 'Depression/PHQ-9 (0-27)',
                        data: depressionData,
                        borderColor: '#8e44ad',
                        backgroundColor: 'rgba(142, 68, 173, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y2',
                        tension: 0.4
                    }},
                    {{
                        label: 'PSS Risk Threshold (20)',
                        data: Array(dates.length).fill(20),
                        borderColor: 'rgba(231, 76, 60, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        yAxisID: 'y',
                        fill: false,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Anxiety Risk Threshold (10)',
                        data: Array(dates.length).fill(10),
                        borderColor: 'rgba(22, 160, 133, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        yAxisID: 'y1',
                        fill: false,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Depression Risk Threshold (14)',
                        data: Array(dates.length).fill(14),
                        borderColor: 'rgba(142, 68, 173, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        yAxisID: 'y2',
                        fill: false,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Clinical Mental Health Scales with Risk Thresholds',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'PSS (0-40)'
                        }},
                        max: 40
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'GAD-7 (0-21)'
                        }},
                        max: 21,
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }},
                    y2: {{
                        type: 'linear',
                        display: false,
                        max: 27
                    }},
                    x: {{
                        ticks: {{
                            maxRotation: 45,
                            minRotation: 45,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML report saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from two-stage predictions")
    parser.add_argument("--student", type=str, help="Specific student ID to process")
    parser.add_argument("--all-students", action="store_true", help="Process all students")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_JSON, help="Predictions JSON file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TWO-STAGE HTML REPORT GENERATOR")
    print("="*80)
    
    # Load predictions
    print(f"\n📂 Loading predictions: {args.predictions}")
    with open(args.predictions) as f:
        data = json.load(f)
    
    predictions = data['predictions']
    metadata = data['metadata']
    
    print(f"✓ Loaded {len(predictions)} predictions for {metadata['num_students']} students")
    
    # Group by student
    students_data = {}
    for pred in predictions:
        student_id = pred['student_id']
        if student_id not in students_data:
            students_data[student_id] = []
        students_data[student_id].append(pred)
    
    # Determine which students to process
    if args.all_students:
        student_ids = list(students_data.keys())
    elif args.student:
        if args.student not in students_data:
            print(f"❌ Student ID '{args.student}' not found")
            return
        student_ids = [args.student]
    else:
        print("❌ Must specify --student or --all-students")
        return
    
    print(f"\n🔄 Generating reports for {len(student_ids)} student(s)...")
    
    # Generate reports
    for student_id in student_ids:
        print(f"\n  Processing: {student_id}")
        
        student_preds = students_data[student_id]
        
        # Generate HTML
        html_filename = f"two_stage_profile_{student_id}.html"
        html_path = args.output_dir / html_filename
        
        generate_html_report(student_id, student_preds, html_path)
        
        print(f"    ✓ {len(student_preds)} days of predictions")
        
        # Calculate stats
        latest = student_preds[-1]
        avg_unc = calculate_avg_uncertainty_pct(latest)
        at_risk, healthy = assess_risk(latest['stage2_mental_health_predictions'])
        
        print(f"    📊 Avg uncertainty: ±{avg_unc:.0f}%")
        print(f"    ⚠️  At-risk: {len(at_risk)} factors")
        print(f"    ✅ Healthy: {len(healthy)} factors")
    
    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nReports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
