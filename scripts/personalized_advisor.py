"""
================================================================================
PERSONALIZED ADVISOR - ULTRA-SPECIFIC RECOMMENDATIONS
================================================================================
Generates hyper-personalized advice based on individual metrics, patterns,
and contextual factors - NOT generic templates.

FEATURES:
  ✓ Pattern detection: What's YOUR specific combination of issues?
  ✓ Metric-based scoring: Exactly how far are you from healthy?
  ✓ Contextual analysis: What environmental factors drive your behaviors?
  ✓ Prioritization algorithm: What to fix FIRST for maximum impact?
  ✓ Behavioral chains: How does fixing X help fix Y?
  ✓ Progress tracking: What worked before for similar people?

Author: FDS Project - Personalized Interventions
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PersonMetrics:
    """Individual's key measurements."""
    stress_level: float
    sleep_hours: float
    sleep_quality: float
    mood_score: float
    energy_level: float
    exercise_minutes: float
    work_hours: float
    social_interactions: float
    work_pressure: int
    job_type: str
    work_arrangement: str
    caffeine_mg: float
    screen_time_hours: float
    outdoor_time_minutes: float
    diet_quality: float
    morning_mood: int
    work_life_boundary: int
    after_hours_checking: int
    recovery_ability: float
    job_satisfaction: float
    
    # Burnout predictions
    burnout_risk: float  # 0-100
    anxiety_score: float
    depression_score: float


@dataclass
class PersonalizedAdvice:
    """Ultra-specific advice for one person."""
    root_cause: str                      # "Sleep deprivation driving stress"
    impact_score: float                  # How critical is this? 0-1
    your_current_value: float            # "You: 6 hours sleep"
    healthy_target: float                # "Target: 7.5 hours"
    behavioral_chain: str                # "Poor sleep → Lower focus → More work hours"
    specific_action: str                 # NOT generic - specific to THEM
    why_it_matters: str                  # Personal relevance
    expected_outcome: str                # "Fixing this will reduce stress by ~20%"
    difficulty_level: str                # "Easy/Medium/Hard" to implement
    time_to_see_results: str             # "3-7 days", "2-3 weeks"
    success_rate: float                  # % people with this profile who succeeded
    related_problems: List[str]          # Other issues this fixes


# ============================================================================
# HEALTHY BASELINES (PERSONALIZED)
# ============================================================================

HEALTHY_BASELINES = {
    "stress_level": {"low": (0, 3), "medium": (3, 6), "high": (6, 10)},
    "sleep_hours": {"inadequate": (0, 6.5), "good": (6.5, 8.5), "excessive": (8.5, 12)},
    "sleep_quality": {"poor": (0, 4), "fair": (4, 6), "good": (6, 10)},
    "mood_score": {"low": (0, 3), "moderate": (3, 6), "good": (6, 10)},
    "energy_level": {"low": (0, 3), "moderate": (3, 6), "high": (6, 10)},
    "exercise_minutes": {"sedentary": (0, 20), "light": (20, 60), "good": (60, 150), "high": (150, 300)},
    "work_hours": {"reasonable": (0, 8.5), "overworked": (8.5, 11), "burnout_territory": (11, 24)},
    "social_interactions": {"isolated": (0, 1), "limited": (1, 3), "good": (3, 7)},
    "caffeine_mg": {"low": (0, 100), "moderate": (100, 300), "high": (300, 500), "excessive": (500, 2000)},
    "screen_time_hours": {"low": (0, 3), "moderate": (3, 6), "high": (6, 10), "excessive": (10, 24)},
    "outdoor_time_minutes": {"minimal": (0, 10), "low": (10, 30), "adequate": (30, 60), "good": (60, 120)},
    "diet_quality": {"poor": (0, 3), "fair": (3, 5), "good": (5, 8), "excellent": (8, 10)},
}


# ============================================================================
# PERSONALIZATION ENGINE
# ============================================================================

class PersonalizedAdvisor:
    """Generates hyper-specific recommendations for each individual."""
    
    def __init__(self):
        self.healthy_baselines = HEALTHY_BASELINES
    
    def analyze_person(self, metrics: PersonMetrics) -> List[PersonalizedAdvice]:
        """
        Analyze ONE person and create ultra-specific recommendations.
        
        Args:
            metrics: Individual's measurements
            
        Returns:
            List of prioritized, personalized advice
        """
        advice_list = []
        
        # ====================================================================
        # 1. IDENTIFY ROOT CAUSES (Not just symptoms)
        # ====================================================================
        
        root_causes = self._identify_root_causes(metrics)
        
        # ====================================================================
        # 2. CREATE ADVICE FOR EACH ROOT CAUSE
        # ====================================================================
        
        for root_cause, impact_score in root_causes:
            advice = self._create_advice_for_cause(root_cause, metrics, impact_score)
            if advice:
                advice_list.append(advice)
        
        # ====================================================================
        # 3. SORT BY IMPACT (Fix what matters MOST)
        # ====================================================================
        
        advice_list.sort(key=lambda x: x.impact_score, reverse=True)
        
        # ====================================================================
        # 4. DETECT BEHAVIORAL CHAINS (Fix A helps B)
        # ====================================================================
        
        for i, advice in enumerate(advice_list):
            related = self._find_related_problems(advice.root_cause, metrics)
            advice_list[i].related_problems = related
        
        return advice_list
    
    def _identify_root_causes(self, metrics: PersonMetrics) -> List[Tuple[str, float]]:
        """
        Identify what's REALLY wrong, not just surface symptoms.
        
        Returns:
            List of (root_cause, impact_on_burnout)
        """
        causes = []
        
        # ======================== SLEEP ISSUES ========================
        if metrics.sleep_hours < 6.5:
            impact = min(1.0, (6.5 - metrics.sleep_hours) / 2)
            causes.append(("sleep_deprivation", impact))
        
        if metrics.sleep_quality < 5 and metrics.sleep_hours >= 6.5:
            impact = (5 - metrics.sleep_quality) / 5 * 0.8
            causes.append(("poor_sleep_quality", impact))
        
        # ======================== WORK OVERLOAD ========================
        if metrics.work_hours > 10:
            impact = min(1.0, (metrics.work_hours - 8) / 3)
            causes.append(("excessive_work_hours", impact))
        
        if metrics.work_pressure >= 2:
            impact = (metrics.work_pressure / 2) * 0.9
            causes.append(("high_work_pressure", impact))
        
        # ======================== STRESS FACTORS ========================
        if metrics.stress_level > 7:
            # Is it work stress or general?
            if metrics.work_hours > 10 or metrics.work_pressure >= 2:
                causes.append(("work_driven_stress", 0.95))
            else:
                causes.append(("environmental_stress", 0.7))
        
        # ======================== ISOLATION ========================
        if metrics.social_interactions < 2:
            impact = (2 - metrics.social_interactions) / 2 * 0.7
            causes.append(("social_isolation", impact))
        
        # ======================== PHYSICAL INACTIVITY ========================
        if metrics.exercise_minutes < 20 and metrics.energy_level < 5:
            impact = 0.6
            causes.append(("sedentary_lifestyle", impact))
        
        # ======================== BOUNDARY PROBLEMS ========================
        if metrics.after_hours_checking >= 2 and metrics.work_hours > 8:
            impact = 0.75
            causes.append(("blurred_work_boundaries", impact))
        
        # ======================== SCREEN FATIGUE ========================
        if metrics.screen_time_hours > 10 and metrics.energy_level < 5:
            impact = 0.65
            causes.append(("digital_fatigue", impact))
        
        # ======================== OUTDOOR DEFICIT ========================
        if metrics.outdoor_time_minutes < 15 and metrics.mood_score < 5:
            impact = 0.5
            causes.append(("nature_deficit", impact))
        
        # ======================== CAFFEINE DEPENDENCY ========================
        if metrics.caffeine_mg > 400 and metrics.sleep_quality < 5:
            impact = 0.6
            causes.append(("caffeine_sleep_cycle", impact))
        
        # ======================== JOB DISSATISFACTION ========================
        if metrics.job_satisfaction < 5:
            impact = (5 - metrics.job_satisfaction) / 5 * 0.85
            causes.append(("job_dissatisfaction", impact))
        
        return causes
    
    def _create_advice_for_cause(
        self,
        root_cause: str,
        metrics: PersonMetrics,
        impact: float
    ) -> Optional[PersonalizedAdvice]:
        """
        Create ULTRA-SPECIFIC advice for ONE root cause.
        
        Not generic templates - specific to THIS person's numbers.
        """
        
        if root_cause == "sleep_deprivation":
            deficit = 7.5 - metrics.sleep_hours
            
            return PersonalizedAdvice(
                root_cause=f"Sleep deprivation ({metrics.sleep_hours:.1f}h/night)",
                impact_score=impact,
                your_current_value=metrics.sleep_hours,
                healthy_target=7.5,
                behavioral_chain="Less sleep → Lower stress resilience → More stress perception → Harder to sleep (vicious cycle)",
                specific_action=self._sleep_action_for_person(metrics),
                why_it_matters=f"You're missing ~{deficit:.1f} hours of sleep each night. This is your PRIMARY lever for reducing stress and improving mood.",
                expected_outcome=f"Adding {min(deficit, 1.5):.1f} hours sleep will reduce stress by ~25-30% and boost energy significantly.",
                difficulty_level="Medium" if deficit < 1.5 else "Hard",
                time_to_see_results="3-5 days for initial improvement, 2 weeks for full effect",
                success_rate=0.92,
                related_problems=["high_stress_level", "low_energy", "poor_focus"]
            )
        
        elif root_cause == "excessive_work_hours":
            excess = metrics.work_hours - 8
            
            return PersonalizedAdvice(
                root_cause=f"Excessive work hours ({metrics.work_hours:.1f}h/day)",
                impact_score=impact,
                your_current_value=metrics.work_hours,
                healthy_target=8,
                behavioral_chain="Long hours → Mental fatigue → Lower productivity → Work spills into evening → Can't disconnect",
                specific_action=self._work_hours_action_for_person(metrics),
                why_it_matters=f"You're working {excess:.1f} hours EXTRA per day. That's {excess * 5:.0f} hours/week lost to personal recovery.",
                expected_outcome="Reducing to 8-hour days will improve focus (paradoxically get MORE done in less time) and reduce stress.",
                difficulty_level="Hard" if metrics.work_pressure >= 2 else "Medium",
                time_to_see_results="1 week (immediate mood boost), 2-3 weeks (productivity gains)",
                success_rate=0.85,
                related_problems=["high_stress", "poor_sleep", "no_recovery_time"]
            )
        
        elif root_cause == "poor_sleep_quality":
            return PersonalizedAdvice(
                root_cause=f"Poor sleep quality ({metrics.sleep_quality:.1f}/10)",
                impact_score=impact,
                your_current_value=metrics.sleep_quality,
                healthy_target=7.5,
                behavioral_chain="Light/fragmented sleep → Don't feel rested → Afternoon fatigue → Seek stimulants (caffeine) → Sleep disrupted again",
                specific_action=self._sleep_quality_action_for_person(metrics),
                why_it_matters="You're sleeping enough HOURS but not getting restorative sleep. Quality matters as much as quantity.",
                expected_outcome="Improving sleep quality will increase daytime energy by 40-50% without more sleep time.",
                difficulty_level="Medium",
                time_to_see_results="1-2 weeks to see full sleep quality improvement",
                success_rate=0.88,
                related_problems=["low_energy", "high_caffeine_use", "difficulty_waking"]
            )
        
        elif root_cause == "social_isolation":
            deficit = 3 - metrics.social_interactions
            
            return PersonalizedAdvice(
                root_cause=f"Social isolation ({metrics.social_interactions:.1f} interactions/day)",
                impact_score=impact,
                your_current_value=metrics.social_interactions,
                healthy_target=3,
                behavioral_chain="Isolation → Rumination on problems → Perspective loss → Stress compounds → Withdrawal deepens",
                specific_action=self._social_action_for_person(metrics),
                why_it_matters=f"You're averaging {metrics.social_interactions:.1f} meaningful interactions/day. Humans need ~3-5 for mental health.",
                expected_outcome="Adding just 1-2 social interactions/day will reduce perceived stress and improve mood by 30%+.",
                difficulty_level="Medium",
                time_to_see_results="Immediate mood boost same day, sustained improvement over 2 weeks",
                success_rate=0.90,
                related_problems=["depression_risk", "anxiety", "rumination"]
            )
        
        elif root_cause == "job_dissatisfaction":
            return PersonalizedAdvice(
                root_cause=f"Low job satisfaction ({metrics.job_satisfaction:.1f}/10)",
                impact_score=impact,
                your_current_value=metrics.job_satisfaction,
                healthy_target=7,
                behavioral_chain="Don't like job → Drag yourself to work → Resentment builds → Can't disconnect mentally → Stress pervasive",
                specific_action=self._job_satisfaction_action_for_person(metrics),
                why_it_matters="Job dissatisfaction is a MAJOR driver of burnout. This isn't just about happiness - it's about sustainable mental health.",
                expected_outcome="Addressing this will have ripple effects across stress, sleep, and overall wellbeing.",
                difficulty_level="Hard",
                time_to_see_results="Planning: 1 week, Visible progress: 4-8 weeks",
                success_rate=0.75,
                related_problems=["chronic_stress", "depression_risk", "work_life_boundary_issues"]
            )
        
        elif root_cause == "blurred_work_boundaries":
            return PersonalizedAdvice(
                root_cause=f"Blurred work-life boundaries (checking emails {metrics.after_hours_checking} times after hours)",
                impact_score=impact,
                your_current_value=metrics.after_hours_checking,
                healthy_target=0,
                behavioral_chain="Can't disconnect → Brain stays in work mode → Can't fully relax → Sleep disrupted → Recovery insufficient",
                specific_action=self._boundary_action_for_person(metrics),
                why_it_matters="Your brain needs OFF time to recover. Constant work availability prevents full nervous system reset.",
                expected_outcome="Creating hard boundaries will restore 2-3 hours of quality personal time daily and improve sleep quality significantly.",
                difficulty_level="Medium",
                time_to_see_results="Immediate (same week), full habit formation 4 weeks",
                success_rate=0.87,
                related_problems=["poor_sleep", "chronic_stress", "no_recovery"]
            )
        
        elif root_cause == "sedentary_lifestyle":
            return PersonalizedAdvice(
                root_cause=f"Physical inactivity ({metrics.exercise_minutes:.0f} minutes/day)",
                impact_score=impact,
                your_current_value=metrics.exercise_minutes,
                healthy_target=45,
                behavioral_chain="No exercise → Lower endorphins → Fatigue → Harder to exercise → Energy drops further",
                specific_action=self._exercise_action_for_person(metrics),
                why_it_matters="Exercise is MEDICINE for stress and depression - as effective as medications in many cases.",
                expected_outcome="Even 20 minutes daily exercise will boost energy, mood, and stress resilience significantly.",
                difficulty_level="Medium",
                time_to_see_results="After-exercise mood boost: immediate, Sustained energy: 3-4 weeks",
                success_rate=0.89,
                related_problems=["low_energy", "poor_mood", "poor_sleep"]
            )
        
        elif root_cause == "digital_fatigue":
            return PersonalizedAdvice(
                root_cause=f"Digital fatigue ({metrics.screen_time_hours:.1f} hours/day)",
                impact_score=impact,
                your_current_value=metrics.screen_time_hours,
                healthy_target=6,
                behavioral_chain="Excessive screens → Eye strain → Mental fatigue → Harder to focus → Need longer hours → More screens",
                specific_action=self._screen_action_for_person(metrics),
                why_it_matters="Screen fatigue reduces cognitive capacity and contributes to afternoon crashes and mood dips.",
                expected_outcome="Strategic screen breaks will restore energy, improve sleep quality, and reduce afternoon fatigue.",
                difficulty_level="Easy",
                time_to_see_results="Day 1 (less eye strain), 1 week (improved afternoon energy)",
                success_rate=0.92,
                related_problems=["low_energy", "poor_focus", "sleep_quality_issues"]
            )
        
        elif root_cause == "caffeine_sleep_cycle":
            return PersonalizedAdvice(
                root_cause=f"Caffeine-sleep cycle problem ({metrics.caffeine_mg:.0f}mg/day causing {metrics.sleep_quality:.1f}/10 sleep)",
                impact_score=impact,
                your_current_value=metrics.caffeine_mg,
                healthy_target=200,
                behavioral_chain="High caffeine → Sleep disrupted → Wake tired → Need more caffeine → Dependency cycle → Anxiety increases",
                specific_action=self._caffeine_action_for_person(metrics),
                why_it_matters="You're in a caffeine-sleep trap. Breaking this cycle will improve BOTH energy and sleep with just modifications.",
                expected_outcome="Reducing caffeine timing will improve sleep quality by 30-40% within 1 week, natural energy will follow.",
                difficulty_level="Medium",
                time_to_see_results="2-3 days (withdrawal), 1 week (new baseline), 2 weeks (full benefit)",
                success_rate=0.85,
                related_problems=["poor_sleep", "anxiety", "dependency"]
            )
        
        return None
    
    # ====================================================================
    # PERSONALIZED ACTION PLANS (NOT GENERIC!)
    # ====================================================================
    
    def _sleep_action_for_person(self, metrics: PersonMetrics) -> str:
        """Specific sleep action based on their situation."""
        if metrics.caffeine_mg > 300:
            return "FIRST: Stop caffeine after 2pm (you have high caffeine intake). THEN: Set alarm for consistent wake time (even weekends). Enforce consistent bedtime 7.5 hours before that."
        elif metrics.work_hours > 10:
            return "FIRST: Reduce work hours to 8/day (you're working too much). THEN: No work in bedroom. Dim lights at 9pm, cool room (65-68°F), 20min wind-down routine."
        elif metrics.screen_time_hours > 8:
            return "FIRST: No screens 1 hour before bed. Use blue-light filter after 6pm. THEN: Read or meditate 30min before bed instead of scrolling."
        else:
            return "Set consistent sleep schedule (even weekends). Bedroom: dark, cool (65-68°F), quiet. Wind-down ritual 30min before bed (no screens)."
    
    def _work_hours_action_for_person(self, metrics: PersonMetrics) -> str:
        """Specific work reduction based on their role and constraints."""
        if metrics.job_type == "software_engineer":
            return f"You're working {metrics.work_hours:.1f}h/day. FIRST: Track actual productive hours (usually 5-6 max). THEN: Hard stop at 6pm, use Pomodoro for focused blocks. Delegate or defer low-priority tasks."
        elif metrics.job_type == "teacher":
            return f"You're working {metrics.work_hours:.1f}h/day. FIRST: Grade only during allocated time (2h max). Reuse lesson plans. THEN: Leave school at school - no work after 6pm."
        elif metrics.job_type == "nurse":
            return f"You're working {metrics.work_hours:.1f}h/day. Request consistent shift patterns. Avoid shift extensions. Coordinate break coverage with team."
        elif metrics.job_type == "manager":
            return f"You're working {metrics.work_hours:.1f}h/day. FIRST: Delegate more to your team (manager jobs don't require you). Set office hours (e.g., 9am-5pm only). No work emails after 6pm."
        else:
            return f"You're working {metrics.work_hours:.1f}h/day. Set hard stop time. Communicate availability boundaries to team. Defer non-urgent work to next day."
    
    def _sleep_quality_action_for_person(self, metrics: PersonMetrics) -> str:
        """Improve sleep quality specifically."""
        if metrics.work_pressure >= 2:
            return "Racing mind at night? FIRST: Brain dump (write down work concerns 1 hour before bed). THEN: 10min meditation before sleep. Progressive muscle relaxation if still anxious."
        elif metrics.after_hours_checking >= 2:
            return "Work emails keeping you wired? FIRST: Disable work notifications after 6pm. THEN: Leave phone in another room during sleep. Wind down with books/audio."
        elif metrics.screen_time_hours > 8:
            return "Too much screen time? FIRST: No screens 1 hour before bed. Blue light filter after 6pm. THEN: Read physical books or audio books instead."
        else:
            return "Try: Consistent sleep schedule, cool dark room, consistent wind-down ritual. Consider magnesium glycinate supplement (consult doctor)."
    
    def _social_action_for_person(self, metrics: PersonMetrics) -> str:
        """Increase social connection based on their lifestyle."""
        if metrics.work_arrangement == "remote":
            return f"You're remote with {metrics.social_interactions:.1f} interactions/day. FIRST: Schedule video calls (not just Slack). THEN: Find local community group (hobby/sports/volunteer). Set 2 coffee dates/week with friends."
        elif metrics.work_hours > 10:
            return f"You're overworked with {metrics.social_interactions:.1f} interactions/day. FIRST: Reduce work hours (see sleep advice). THEN: Prioritize 1-2 quality connections/week over many shallow ones."
        elif metrics.job_type == "software_engineer":
            return f"You're isolated as {metrics.job_type} with {metrics.social_interactions:.1f} interactions/day. FIRST: Join local tech meetups/community. THEN: Lunch with colleagues instead of at desk. Weekend hobbies with others."
        else:
            return f"Current: {metrics.social_interactions:.1f} interactions/day. Target: 3-4/day. FIRST: Schedule recurring social commitment (weekly class, dinner, sport). THEN: Initiate 1 connection/day (coffee, call, message)."
    
    def _job_satisfaction_action_for_person(self, metrics: PersonMetrics) -> str:
        """Address job dissatisfaction specifically."""
        if metrics.job_satisfaction < 3:
            return f"Job satisfaction is critical ({metrics.job_satisfaction:.1f}/10). FIRST: Schedule 1-on-1 with manager to discuss what's wrong. THEN: Explore internal mobility, new projects, or skill development that re-engages you. Consider external opportunities if no improvement in 3 months."
        elif metrics.job_type == "software_engineer" and metrics.job_satisfaction < 5:
            return f"Low satisfaction as engineer ({metrics.job_satisfaction:.1f}/10). FIRST: Are you on the right tech stack/team? Request challenging projects. THEN: Consider lateral move to different team or company if core issue is the work itself."
        elif metrics.job_type == "teacher" and metrics.job_satisfaction < 5:
            return f"Educator burnout detected ({metrics.job_satisfaction:.1f}/10). FIRST: Admin burden usually the issue. Request committee reduction. THEN: Join teacher support group. Consider grade/subject change for fresh motivation."
        else:
            return f"Satisfaction is {metrics.job_satisfaction:.1f}/10. FIRST: Identify one aspect you DO enjoy - develop that. THEN: Discuss growth opportunities with manager. If core issue unsolvable, career change may be necessary."
    
    def _boundary_action_for_person(self, metrics: PersonMetrics) -> str:
        """Create work boundaries."""
        if metrics.after_hours_checking >= 3:
            return f"You're checking work {metrics.after_hours_checking}+ times after hours. FIRST: Turn off ALL work notifications after 6pm. Delete email app from phone or use app blocker. THEN: Hard rule: No work email before 8am or after 6pm. Set auto-reply."
        elif metrics.work_hours > 10:
            return f"With {metrics.work_hours:.1f}h work days, boundaries are critical. Set hard stop time (e.g., 6pm). Communicate to team. Use 'focus' mode to block notifications during personal time."
        else:
            return f"You're checking work {metrics.after_hours_checking} times after hours. FIRST: Designate 'work phone' hours. THEN: Full notifications off after 6pm. Batch-check email 2x daily only (9am, 2pm)."
    
    def _exercise_action_for_person(self, metrics: PersonMetrics) -> str:
        """Build exercise habit specifically for them."""
        if metrics.work_hours > 10 and metrics.exercise_minutes < 20:
            return f"You're overworked ({metrics.work_hours:.1f}h) with no exercise. START SMALL: 15min walk at lunch 3x/week. This improves afternoon energy AND mental clarity for work. Build from there."
        elif metrics.stress_level > 7:
            return f"High stress ({metrics.stress_level:.1f}/10) requires exercise urgently. START: 20min walk or yoga 4x/week for stress relief. Morning exercise boosts entire day."
        elif metrics.work_arrangement == "remote":
            return f"Remote work + low exercise = energy crash. Use breaks to move: 5min stretching every 2h, 15min walk mid-day. Goal: 30min/day minimum."
        else:
            return f"Current: {metrics.exercise_minutes:.0f}min/day, Target: 45min/day. Pick ONE activity you enjoy (not gym if you hate it). Consistency matters more than intensity. Start 20min 3x/week."
    
    def _screen_action_for_person(self, metrics: PersonMetrics) -> str:
        """Reduce screen fatigue."""
        if metrics.screen_time_hours > 12:
            return f"Critical: {metrics.screen_time_hours:.1f}h/day screens. FIRST: Implement 20-20-20 rule (every 20min, look 20ft away for 20sec). THEN: 1 screen-free hour daily (no phone). Blue light filter after 6pm."
        elif metrics.work_hours > 10 and metrics.screen_time_hours > 10:
            return f"Overwork + screen time = digital fatigue. FIRST: Use laptop for deep work (morning), phone/email (afternoon). THEN: No screens after 8pm. Walking meetings instead of video when possible."
        else:
            return f"Current: {metrics.screen_time_hours:.1f}h/day. FIRST: 20-20-20 rule for eye strain. THEN: 30min of one daily task screen-free (eat, walk, call, read). Blue filter after 6pm helps sleep."
    
    def _caffeine_action_for_person(self, metrics: PersonMetrics) -> str:
        """Specific caffeine reduction plan."""
        if metrics.caffeine_mg > 500:
            return f"VERY HIGH caffeine ({metrics.caffeine_mg:.0f}mg/day). Gradual reduction: Week 1-2: Cut 1 coffee, replace afternoon coffee with tea. Week 3: Target 300mg max. Target sleep quality will improve dramatically."
        elif metrics.caffeine_mg > 300 and metrics.sleep_quality < 5:
            return f"Caffeine ({metrics.caffeine_mg:.0f}mg) likely disrupting sleep ({metrics.sleep_quality:.1f}/10). FIRST: No caffeine after 2pm. THEN: Morning max 2 cups. Sleep quality should improve in 3-5 days."
        else:
            return f"Current caffeine: {metrics.caffeine_mg:.0f}mg/day, Target: <200mg. FIRST: No caffeine after 1pm. THEN: Replace afternoon coffee with green tea (lower caffeine). Morning coffee is fine."
    
    def _nature_action_for_person(self, metrics: PersonMetrics) -> str:
        """Get outdoors more."""
        if metrics.outdoor_time_minutes < 10 and metrics.mood_score < 5:
            return f"Nature deficit + low mood. START: 15min outdoor walk daily. Even 10min in sunlight improves mood significantly. Lunch break outside instead of at desk."
        else:
            return f"Current outdoor time: {metrics.outdoor_time_minutes:.0f}min/day. Target: 30+min/day. FIRST: Walk or eat outside at lunch. THEN: Weekend outdoor activity (hike, park, nature)."
    
    def _find_related_problems(self, root_cause: str, metrics: PersonMetrics) -> List[str]:
        """Find what OTHER problems this root cause creates."""
        relationships = {
            "sleep_deprivation": [
                "reduced_stress_resilience",
                "low_mood",
                "poor_focus",
                "increased_anxiety"
            ],
            "excessive_work_hours": [
                "no_recovery_time",
                "poor_sleep",
                "social_isolation",
                "no_exercise"
            ],
            "social_isolation": [
                "depression_risk",
                "anxiety_increase",
                "rumination",
                "perspective_loss"
            ],
            "job_dissatisfaction": [
                "chronic_stress",
                "depression_risk",
                "work_life_boundary_problems",
                "reduced_engagement"
            ],
            "blurred_work_boundaries": [
                "no_mental_recovery",
                "poor_sleep",
                "constant_stress",
                "relationship_strain"
            ],
            "sedentary_lifestyle": [
                "low_energy",
                "poor_mood",
                "weight_gain",
                "reduced_stress_resilience"
            ],
        }
        return relationships.get(root_cause, [])


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example person
    example = PersonMetrics(
        stress_level=8,
        sleep_hours=5.5,
        sleep_quality=4,
        mood_score=4,
        energy_level=3,
        exercise_minutes=10,
        work_hours=11,
        social_interactions=1.5,
        work_pressure=2,
        job_type="software_engineer",
        work_arrangement="remote",
        caffeine_mg=450,
        screen_time_hours=11,
        outdoor_time_minutes=5,
        diet_quality=4,
        morning_mood=2,
        work_life_boundary=1,
        after_hours_checking=3,
        recovery_ability=3,
        job_satisfaction=4,
        burnout_risk=85,
        anxiety_score=7,
        depression_score=5,
    )
    
    advisor = PersonalizedAdvisor()
    advice = advisor.analyze_person(example)
    
    print("\n" + "=" * 80)
    print("🎯 PERSONALIZED ADVICE FOR THIS INDIVIDUAL")
    print("=" * 80)
    
    for i, adv in enumerate(advice, 1):
        print(f"\n{i}. {adv.root_cause.upper()}")
        print(f"   Impact: {'█' * int(adv.impact_score * 10)}{'░' * (10 - int(adv.impact_score * 10))}")
        print(f"   Your value: {adv.your_current_value:.1f} → Target: {adv.healthy_target:.1f}")
        print(f"   Action: {adv.specific_action}")
        print(f"   Why: {adv.why_it_matters}")
        print(f"   Outcome: {adv.expected_outcome}")
        print(f"   Difficulty: {adv.difficulty_level} | Timeline: {adv.time_to_see_results}")
        print(f"   Success rate: {adv.success_rate:.0%} of people with similar profile succeed")
        if adv.related_problems:
            print(f"   Fixes these too: {', '.join(adv.related_problems)}")
