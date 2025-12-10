"""
================================================================================
GENERATE BURNOUT REPORT - Beautiful HTML Output
================================================================================
STANDALONE script - generates HTML report from predictions.
Works with models downloaded from Kaggle!

Usage:
    python generate_report.py --csv responses.csv --output report.html
    python generate_report.py --csv responses.csv --model-path ./lstm_sequence.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import from predict_burnout
from predict_burnout import (
    load_model,
    parse_google_form_csv,
    create_weekly_sequence,
    predict,
    RISK_LEVELS,
    DEFAULTS,
    FEATURE_COLS,
    PROJECT_ROOT,
    FEATURE_LABELS,
    get_specific_advice,
    detect_contradictions,
    get_cvae_advisor,
)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Burnout Risk Assessment Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .result-card {
            text-align: center;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .risk-low {
            background: #d4edda;
            color: #155724;
        }
        
        .risk-medium {
            background: #fff3cd;
            color: #856404;
        }
        
        .risk-high {
            background: #f8d7da;
            color: #721c24;
        }
        
        .confidence {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 20px;
        }
        
        .emoji {
            font-size: 4rem;
            margin-bottom: 10px;
        }
        
        .description {
            font-size: 1.1rem;
            color: #555;
            max-width: 500px;
            margin: 0 auto;
        }
        
        .meter-container {
            margin: 30px 0;
        }
        
        .meter-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .meter {
            height: 25px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        
        .meter-fill {
            height: 100%;
            border-radius: 15px;
            transition: width 0.5s ease;
        }
        
        .meter-low { background: linear-gradient(90deg, #28a745, #5cb85c); }
        .meter-medium { background: linear-gradient(90deg, #ffc107, #ffdb58); }
        .meter-high { background: linear-gradient(90deg, #dc3545, #ff6b6b); }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }
        
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        
        .metric-status {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        
        .recommendations {
            list-style: none;
        }
        
        .recommendations li {
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .recommendation-item {
            background: #f8f9fa !important;
        }
        
        .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .impact-badge {
            font-size: 0.85rem;
            padding: 4px 10px;
            border-radius: 15px;
            font-weight: 500;
        }
        
        .impact-good {
            background: #d4edda;
            color: #155724;
        }
        
        .impact-neutral {
            background: #e9ecef;
            color: #6c757d;
        }
        
        .rec-values {
            font-size: 0.95rem;
            color: #555;
            margin-bottom: 8px;
        }
        
        .value-bad {
            color: #dc3545;
            font-weight: 600;
        }
        
        .value-good {
            color: #28a745;
            font-weight: 600;
        }
        
        .rec-advice {
            font-size: 0.95rem;
            color: #333;
            background: #fff;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .recommendation-summary {
            background: linear-gradient(135deg, #667eea15, #764ba215) !important;
            border-left-color: #28a745 !important;
            text-align: center;
        }
        
        .person-name {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 20px;
            padding: 10px;
            background: linear-gradient(135deg, #667eea15, #764ba215);
            border-radius: 10px;
        }
        
        .person-name strong {
            color: #333;
            font-size: 1.4rem;
        }
        
        .contradictions-section {
            background: #fff8e6;
            border: 2px solid #ffc107;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .contradictions-section h3 {
            color: #856404;
            margin-bottom: 10px;
        }
        
        .contradictions-intro {
            color: #666;
            margin-bottom: 15px;
        }
        
        .warning-item {
            padding: 12px 15px;
            margin: 10px 0;
            border-radius: 8px;
            line-height: 1.5;
        }
        
        .warning-normal {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        
        .warning-critical {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        .footer {
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 20px;
        }
        
        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 0.9rem;
            color: #856404;
        }
        
        /* CVAE AI Suggestions */
        .cvae-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            color: white;
        }
        
        .cvae-section h3 {
            margin-bottom: 10px;
            color: white;
        }
        
        .cvae-intro {
            opacity: 0.9;
            margin-bottom: 15px;
            font-size: 0.95rem;
        }
        
        .cvae-suggestions {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
        }
        
        .cvae-item {
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 0.95rem;
        }
        
        .cvae-item:last-child {
            border-bottom: none;
        }
        
        .cvae-change {
            margin-bottom: 6px;
        }
        
        .cvae-tip {
            background: rgba(255, 255, 255, 0.15);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9rem;
            margin-top: 4px;
        }
        
        .cvae-arrow {
            margin-right: 5px;
        }
        
        .cvae-label {
            font-weight: 500;
        }
        
        .cvae-current {
            opacity: 0.7;
        }
        
        .cvae-arrow-text {
            margin: 0 5px;
        }
        
        .cvae-suggested {
            font-weight: 600;
        }
        
        .cvae-diff {
            opacity: 0.8;
            font-size: 0.85rem;
        }
        
        .cvae-note {
            margin-top: 15px;
            font-size: 0.9rem;
            opacity: 0.85;
        }
        
        .cvae-error {
            background: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        /* 7-Day Action Plan */
        .action-plan {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .action-plan h2 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .plan-intro {
            color: #666;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }
        
        .priority-focus {
            background: linear-gradient(135deg, #667eea15, #764ba215);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
        }
        
        .priority-focus h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .priority-items {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .priority-item {
            background: white;
            border: 2px solid #667eea;
            border-radius: 25px;
            padding: 8px 16px;
            font-size: 0.9rem;
            font-weight: 500;
            color: #667eea;
        }
        
        .priority-item.critical {
            background: #dc3545;
            border-color: #dc3545;
            color: white;
        }
        
        .weekly-schedule {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }
        
        .day-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }
        
        .day-card.rest-day {
            border-left-color: #28a745;
            background: #f0fff4;
        }
        
        .day-header {
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 1rem;
        }
        
        .day-tasks {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .day-tasks li {
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
            font-size: 0.9rem;
            color: #555;
        }
        
        .day-tasks li:last-child {
            border-bottom: none;
        }
        
        .task-time {
            font-weight: 500;
            color: #667eea;
            margin-right: 8px;
        }
        
        .task-category {
            display: inline-block;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 5px;
        }
        
        .cat-sleep { background: #e3f2fd; color: #1565c0; }
        .cat-exercise { background: #e8f5e9; color: #2e7d32; }
        .cat-stress { background: #fce4ec; color: #c2185b; }
        .cat-work { background: #fff3e0; color: #e65100; }
        .cat-social { background: #f3e5f5; color: #7b1fa2; }
        
        .weekly-goals {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .weekly-goals h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .goal-item {
            display: flex;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .goal-item:last-child {
            border-bottom: none;
        }
        
        .goal-checkbox {
            width: 24px;
            height: 24px;
            border: 2px solid #667eea;
            border-radius: 6px;
            margin-right: 12px;
            flex-shrink: 0;
        }
        
        .goal-text {
            flex: 1;
            font-size: 0.95rem;
        }
        
        .goal-metric {
            font-size: 0.85rem;
            color: #666;
            background: #e9ecef;
            padding: 4px 10px;
            border-radius: 15px;
        }
        
        .intervention-banner {
            background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
            color: white;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
        }
        
        .intervention-banner h3 {
            margin-bottom: 10px;
            color: white;
        }
        
        .intervention-banner p {
            opacity: 0.95;
            margin-bottom: 15px;
        }
        
        .intervention-options {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
        }
        
        .intervention-option {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        /* 7-Day Behavioral Heatmap */
        .heatmap-section {
            margin-top: 20px;
        }
        
        .heatmap-section h2 {
            margin-bottom: 15px;
        }
        
        .heatmap-intro {
            color: #666;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }
        
        .heatmap-container {
            overflow-x: auto;
        }
        
        .heatmap-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        
        .heatmap-table th {
            background: #f8f9fa;
            padding: 10px 8px;
            text-align: center;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
        }
        
        .heatmap-table td {
            padding: 8px;
            text-align: center;
            border-bottom: 1px solid #e9ecef;
            font-weight: 500;
        }
        
        .heatmap-table tr:hover {
            background: #f8f9fa;
        }
        
        .heatmap-feature {
            text-align: left !important;
            font-weight: 600;
            color: #333;
            padding-left: 15px !important;
            white-space: nowrap;
        }
        
        /* Risk-based cell colors */
        .risk-healthy { background: #d4edda; color: #155724; }
        .risk-mild { background: #fff3cd; color: #856404; }
        .risk-warning { background: #ffe5cc; color: #a35200; }
        .risk-high { background: #f8d7da; color: #721c24; }
        
        .heatmap-legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
        }
        
        .legend-color {
            width: 18px;
            height: 18px;
            border-radius: 4px;
        }
        
        @media (max-width: 600px) {
            .header h1 { font-size: 1.8rem; }
            .risk-badge { font-size: 1.2rem; padding: 10px 25px; }
            .emoji { font-size: 3rem; }
            .weekly-schedule { grid-template-columns: 1fr; }
            .heatmap-table { font-size: 0.75rem; }
            .heatmap-table th, .heatmap-table td { padding: 5px 3px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Burnout Risk Assessment</h1>
            <p>Generated on {date}</p>
        </div>
        
        {results}
        
        <div class="footer">
            <p>Powered by ML - University Project</p>
        </div>
    </div>
</body>
</html>
"""

RESULT_CARD_TEMPLATE = """
<div class="card result-card">
    <div class="emoji">{emoji}</div>
    <h2>Your Burnout Risk Level</h2>
    <div class="risk-badge risk-{risk_class}">{risk_name} RISK</div>
    <div class="confidence">Confidence: {confidence:.0f}%</div>
    <p class="description">{description}</p>
    
    <div class="meter-container">
        <div class="meter-label">
            <span>🟢 Low Risk</span>
            <span>{low_prob:.0f}%</span>
        </div>
        <div class="meter">
            <div class="meter-fill meter-low" style="width: {low_prob}%"></div>
        </div>
        
        <div class="meter-label">
            <span>🟡 Medium Risk</span>
            <span>{med_prob:.0f}%</span>
        </div>
        <div class="meter">
            <div class="meter-fill meter-medium" style="width: {med_prob}%"></div>
        </div>
        
        <div class="meter-label">
            <span>🔴 High Risk</span>
            <span>{high_prob:.0f}%</span>
        </div>
        <div class="meter">
            <div class="meter-fill meter-high" style="width: {high_prob}%"></div>
        </div>
    </div>
</div>

<div class="card">
    <h2>📊 Your Key Metrics</h2>
    <div class="metrics-grid">
        {metrics}
    </div>
</div>

<div class="card">
    <h2>💡 Personalized Recommendations</h2>
    <ul class="recommendations">
        {recommendations}
    </ul>
    
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This is a demo prediction from a machine learning model. 
        It is not medical advice. For real mental health concerns, please consult a healthcare professional.
    </div>
</div>
"""


def generate_recommendations_html(data: dict, pred_class: int, model, feature_cols: list) -> str:
    """
    Generate HTML for DATA-DRIVEN recommendations with what-if analysis.
    Same logic as predict_burnout.py but formatted as HTML.
    """
    import numpy as np
    
    # Healthy baselines (from low-burnout population)
    healthy_baselines = {
        "stress_level": 4.0,
        "sleep_hours": 7.5,
        "sleep_quality": 7.0,
        "work_hours": 7.5,
        "exercise_minutes": 45,
        "mood_score": 7.0,
        "energy_level": 7.0,
        "focus_score": 7.0,
        "caffeine_mg": 80,
        "screen_time_hours": 3.5,
        "meetings_count": 2,
        "alcohol_units": 0.5,
        "steps_count": 8000,
        "tasks_completed": 6,
        "work_pressure": 0.5,
        "commute_minutes": 20,
        "emails_received": 15,
    }
    
    # Calculate deviations from healthy baseline
    deviations = []
    for feature in feature_cols:
        current = data.get(feature, DEFAULTS.get(feature, 5))
        healthy = healthy_baselines.get(feature, current)
        
        lower_is_better = feature in [
            "stress_level", "work_hours", "caffeine_mg", "screen_time_hours",
            "meetings_count", "alcohol_units", "work_pressure", "commute_minutes",
            "emails_received"
        ]
        
        if lower_is_better:
            deviation = current - healthy
            direction = "reduce"
        else:
            deviation = healthy - current
            direction = "increase"
        
        ranges = {
            "stress_level": 10, "sleep_hours": 4, "sleep_quality": 10,
            "work_hours": 6, "exercise_minutes": 60, "mood_score": 10,
            "energy_level": 10, "focus_score": 10, "caffeine_mg": 300,
            "screen_time_hours": 8, "meetings_count": 8, "alcohol_units": 5,
            "steps_count": 10000, "tasks_completed": 10, "work_pressure": 2,
            "commute_minutes": 60, "emails_received": 50,
        }
        
        normalized_deviation = deviation / ranges.get(feature, 10)
        
        if normalized_deviation > 0.1:
            deviations.append({
                "feature": feature,
                "current": current,
                "healthy": healthy,
                "deviation": normalized_deviation,
                "direction": direction,
            })
    
    deviations.sort(key=lambda x: x["deviation"], reverse=True)
    
    if not deviations:
        return "<li>✨ Your metrics are close to healthy baselines! Keep up the great work.</li>"
    
    # Generate HTML for top 5 recommendations with what-if analysis
    html_parts = []
    total_reduction = 0
    
    for dev in deviations[:5]:
        feature = dev["feature"]
        current = dev["current"]
        healthy = dev["healthy"]
        direction = dev["direction"]
        
        # Run what-if simulation
        modified_data = data.copy()
        modified_data[feature] = healthy
        
        modified_sequence = create_weekly_sequence(modified_data, feature_cols)
        _, new_probs = predict(model, modified_sequence, feature_cols)
        
        current_sequence = create_weekly_sequence(data, feature_cols)
        _, current_probs = predict(model, current_sequence, feature_cols)
        
        current_risk = 1 - current_probs[0]
        new_risk = 1 - new_probs[0]
        risk_reduction = (current_risk - new_risk) * 100
        
        if risk_reduction > 0:
            total_reduction += risk_reduction
        
        # Format
        feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        diff = abs(current - healthy)
        specific_advice = get_specific_advice(feature, current, healthy, diff)
        
        if risk_reduction > 1:
            impact_badge = f'<span class="impact-badge impact-good">↓{risk_reduction:.0f}% risk</span>'
        else:
            impact_badge = '<span class="impact-badge impact-neutral">minimal impact</span>'
        
        html_parts.append(f"""
        <li class="recommendation-item">
            <div class="rec-header">
                <strong>{feature_label}</strong>
                {impact_badge}
            </div>
            <div class="rec-values">
                Current: <span class="value-bad">{current:.0f}</span> → 
                Target: <span class="value-good">{healthy:.0f}</span>
            </div>
            <div class="rec-advice">💡 {specific_advice}</div>
        </li>
        """)
    
    # Add summary if significant reduction possible
    if total_reduction > 10:
        html_parts.append(f"""
        <li class="recommendation-summary">
            📈 <strong>Combined potential risk reduction: ~{total_reduction:.0f}%</strong>
            <br><small>(if you address all factors above)</small>
        </li>
        """)
    
    return "\n".join(html_parts)


def generate_cvae_suggestions_html(data: dict, pred_class: int, probs: list = None) -> str:
    """
    Generate HTML for AI-powered CVAE suggestions.
    
    IMPORTANT: We filter suggestions to only show BENEFICIAL changes that align
    with health science. The CVAE learns correlations, not causation.
    
    Shows suggestions for:
    - Medium/High risk users (always)
    - Borderline low risk users (< 75% confidence) 
    """
    import torch
    
    # Skip if clearly low risk (>= 75% confidence)
    if pred_class == 0 and probs is not None and probs[0] >= 0.75:
        return ""  # Strong low risk, no suggestions needed
    
    cvae, stats = get_cvae_advisor()
    if cvae is None:
        return ""
    
    # Define what direction is ACTUALLY healthy for each feature
    # True = higher is better, False = lower is better
    HEALTHY_DIRECTION = {
        "stress_level": False,      # Lower stress is better
        "sleep_hours": True,        # More sleep is better
        "sleep_quality": True,      # Better sleep quality is better
        "work_hours": False,        # Fewer work hours is better
        "exercise_minutes": True,   # More exercise is better
        "mood_score": True,         # Better mood is better
        "energy_level": True,       # More energy is better
        "focus_score": True,        # Better focus is better
        "caffeine_mg": False,       # Less caffeine is better
        "screen_time_hours": False, # Less screen time is better
        "meetings_count": False,    # Fewer meetings is better
        "alcohol_units": False,     # Less alcohol is better
        "steps_count": True,        # More steps is better
        "tasks_completed": True,    # More tasks completed is better
        "work_pressure": False,     # Lower pressure is better
        "commute_minutes": False,   # Shorter commute is better
        "emails_received": False,   # Fewer emails is better
    }
    
    try:
        # Use CVAE's feature columns
        cvae_feature_cols = stats.get("feature_cols", FEATURE_COLS)
        
        # Create sequence
        sequence = create_weekly_sequence(data, cvae_feature_cols)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        current_label = torch.tensor([pred_class])
        
        # Normalize if stats available
        mean_val = stats.get("mean")
        std_val = stats.get("std")
        if mean_val is not None and std_val is not None:
            mean = torch.tensor(mean_val, dtype=torch.float32)
            std = torch.tensor(std_val, dtype=torch.float32)
            x_norm = (x - mean) / (std + 1e-8)
        else:
            x_norm = x
        
        # Generate counterfactual
        suggested = cvae.suggest_changes(x_norm, current_label, target_label=0)
        
        # Denormalize
        if mean_val is not None and std_val is not None:
            suggested = suggested * (std + 1e-8) + mean
        
        # Average over week
        suggested_avg = suggested.squeeze(0).mean(dim=0).numpy()
        
        # Build suggestions list - ONLY beneficial changes
        changes = []
        for i, feature in enumerate(cvae_feature_cols):
            current = data.get(feature, DEFAULTS.get(feature, 5))
            suggested_val = suggested_avg[i] if i < len(suggested_avg) else current
            diff = suggested_val - current
            
            # Skip tiny changes
            if abs(diff) < 0.5 and (abs(diff) / max(abs(current), 1)) < 0.1:
                continue
            
            # CRITICAL: Only show suggestions that align with health science!
            higher_is_better = HEALTHY_DIRECTION.get(feature, True)
            is_beneficial = (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better)
            
            if is_beneficial:
                changes.append({
                    "feature": feature,
                    "current": current,
                    "suggested": suggested_val,
                    "diff": diff,
                })
        
        changes.sort(key=lambda x: abs(x["diff"]), reverse=True)
        
        if not changes:
            return ""  # No beneficial suggestions found
        
        # Build HTML with actionable advice
        items = []
        for change in changes[:5]:
            feature = change["feature"]
            current = change["current"]
            suggested = change["suggested"]
            diff = change["diff"]
            
            feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            arrow = "↗️" if diff > 0 else "↘️"
            
            # Get actionable advice for HOW to make this change
            actionable_tip = get_specific_advice(feature, current, suggested, abs(diff))
            
            items.append(f"""
            <div class="cvae-item">
                <div class="cvae-change">
                    <span class="cvae-arrow">{arrow}</span>
                    <span class="cvae-label">{feature_label}:</span>
                    <span class="cvae-current">{current:.1f}</span>
                    <span class="cvae-arrow-text">→</span>
                    <span class="cvae-suggested">{suggested:.1f}</span>
                    <span class="cvae-diff">({diff:+.1f})</span>
                </div>
                <div class="cvae-tip">💡 {actionable_tip}</div>
            </div>
            """)
        
        return f"""
        <div class="cvae-section">
            <h3>🤖 AI-Powered Suggestions</h3>
            <p class="cvae-intro">Personalized changes with actionable tips:</p>
            <div class="cvae-suggestions">
                {"".join(items)}
            </div>
            <p class="cvae-note">✨ Tips based on behavioral science research.</p>
        </div>
        """
        
    except Exception as e:
        return f'<div class="cvae-error">⚠️ Could not generate AI suggestions: {e}</div>'


def generate_action_plan_html(data: dict, pred_class: int) -> str:
    """
    Generate a comprehensive 7-day action plan based on the user's data.
    
    This creates a structured weekly plan with:
    - Priority focus areas (what to fix first)
    - Daily action items with specific times
    - Weekly goals with measurable targets
    - Professional intervention suggestions if needed
    
    Based on intervention data: meditation, therapy, exercise_plan, 
    diet_coaching, workload_cap, vacation patterns from our dataset.
    """
    
    if pred_class == 0:
        # Low risk - maintenance plan
        return generate_maintenance_plan_html(data)
    
    # Identify priority areas (biggest gaps from healthy baseline)
    healthy_baselines = {
        "stress_level": 4.0,
        "sleep_hours": 7.5,
        "sleep_quality": 7.0,
        "work_hours": 7.5,
        "exercise_minutes": 45,
        "mood_score": 7.0,
        "energy_level": 7.0,
        "focus_score": 7.0,
        "caffeine_mg": 100,
        "screen_time_hours": 3.0,
        "meetings_count": 3,
        "alcohol_units": 0.5,
        "steps_count": 8000,
        "commute_minutes": 30,
        "emails_received": 20,
    }
    
    lower_is_better = {
        "stress_level", "work_hours", "caffeine_mg", "screen_time_hours",
        "meetings_count", "alcohol_units", "commute_minutes", "emails_received"
    }
    
    # Calculate deviations and identify priorities
    priorities = []
    for feature, healthy in healthy_baselines.items():
        current = data.get(feature, healthy)
        if feature in lower_is_better:
            deviation = (current - healthy) / max(healthy, 1)
        else:
            deviation = (healthy - current) / max(healthy, 1)
        
        if deviation > 0.2:  # More than 20% off target
            priorities.append({
                "feature": feature,
                "current": current,
                "target": healthy,
                "deviation": deviation,
                "critical": deviation > 0.5
            })
    
    priorities.sort(key=lambda x: x["deviation"], reverse=True)
    top_priorities = priorities[:4]
    
    # Determine which interventions are recommended
    interventions_needed = []
    
    stress = data.get("stress_level", 5)
    sleep_quality = data.get("sleep_quality", 5)
    exercise = data.get("exercise_minutes", 30)
    work_hours = data.get("work_hours", 8)
    mood = data.get("mood_score", 5)
    
    if stress >= 7 or mood <= 4:
        interventions_needed.append(("meditation", "Daily 10-min meditation practice"))
    if mood <= 3 or stress >= 8:
        interventions_needed.append(("therapy", "Consider professional counseling"))
    if exercise < 20:
        interventions_needed.append(("exercise_plan", "Structured 4-week fitness program"))
    if work_hours > 10:
        interventions_needed.append(("workload_cap", "Discuss workload limits with manager"))
    if stress >= 8 and work_hours > 9:
        interventions_needed.append(("vacation", "Plan recovery time off"))
    
    # Generate the action plan HTML
    html_parts = []
    
    # Priority Focus Section
    priority_items = []
    for p in top_priorities:
        label = FEATURE_LABELS.get(p["feature"], p["feature"].replace("_", " ").title())
        css_class = "priority-item critical" if p["critical"] else "priority-item"
        priority_items.append(f'<span class="{css_class}">{label}</span>')
    
    if priority_items:
        html_parts.append(f"""
        <div class="priority-focus">
            <h3>🎯 Your Priority Focus Areas</h3>
            <div class="priority-items">
                {"".join(priority_items)}
            </div>
        </div>
        """)
    
    # Weekly Schedule
    days = generate_weekly_schedule(data, top_priorities)
    
    day_cards = []
    for day in days:
        is_rest = day["name"] in ["Saturday", "Sunday"]
        css_class = "day-card rest-day" if is_rest else "day-card"
        
        tasks_html = ""
        for task in day["tasks"]:
            cat_class = f"task-category cat-{task['category']}"
            tasks_html += f"""
            <li>
                <span class="task-time">{task['time']}</span>
                {task['action']}
                <span class="{cat_class}">{task['category']}</span>
            </li>
            """
        
        day_cards.append(f"""
        <div class="{css_class}">
            <div class="day-header">📅 {day['name']}</div>
            <ul class="day-tasks">
                {tasks_html}
            </ul>
        </div>
        """)
    
    html_parts.append(f"""
    <div class="weekly-schedule">
        {"".join(day_cards)}
    </div>
    """)
    
    # Weekly Goals
    goals = generate_weekly_goals(data, top_priorities)
    goals_html = ""
    for goal in goals:
        goals_html += f"""
        <div class="goal-item">
            <div class="goal-checkbox"></div>
            <div class="goal-text">{goal['text']}</div>
            <div class="goal-metric">{goal['metric']}</div>
        </div>
        """
    
    html_parts.append(f"""
    <div class="weekly-goals">
        <h3>✅ Weekly Goals (Track Your Progress)</h3>
        {goals_html}
    </div>
    """)
    
    # Professional Intervention Banner (if needed)
    if interventions_needed and pred_class == 2:  # High risk
        options_html = ""
        for int_type, desc in interventions_needed[:3]:
            options_html += f'<span class="intervention-option">📊 {desc}</span>'
        
        html_parts.append(f"""
        <div class="intervention-banner">
            <h3>🏥 Consider Professional Support</h3>
            <p>Based on your assessment, these evidence-based interventions could help:</p>
            <div class="intervention-options">
                {options_html}
            </div>
        </div>
        """)
    
    return f"""
    <div class="card action-plan">
        <h2>📋 Your 7-Day Action Plan</h2>
        <p class="plan-intro">A structured plan to improve your work-life balance. Focus on small, consistent changes.</p>
        {"".join(html_parts)}
    </div>
    """


def generate_maintenance_plan_html(data: dict) -> str:
    """Generate a lighter maintenance plan for low-risk users."""
    return """
    <div class="card action-plan">
        <h2>📋 Your Wellness Maintenance Plan</h2>
        <p class="plan-intro">You're doing great! Here's how to maintain your healthy balance.</p>
        
        <div class="priority-focus">
            <h3>✨ Keep Up These Healthy Habits</h3>
            <div class="priority-items">
                <span class="priority-item">Consistent Sleep</span>
                <span class="priority-item">Regular Exercise</span>
                <span class="priority-item">Work-Life Balance</span>
                <span class="priority-item">Stress Management</span>
            </div>
        </div>
        
        <div class="weekly-goals">
            <h3>✅ Weekly Wellness Check</h3>
            <div class="goal-item">
                <div class="goal-checkbox"></div>
                <div class="goal-text">Maintained 7+ hours of sleep most nights</div>
                <div class="goal-metric">5/7 nights</div>
            </div>
            <div class="goal-item">
                <div class="goal-checkbox"></div>
                <div class="goal-text">Got at least 30 min of movement daily</div>
                <div class="goal-metric">150 min/week</div>
            </div>
            <div class="goal-item">
                <div class="goal-checkbox"></div>
                <div class="goal-text">Took real breaks during work</div>
                <div class="goal-metric">Daily</div>
            </div>
            <div class="goal-item">
                <div class="goal-checkbox"></div>
                <div class="goal-text">Connected with friends or family</div>
                <div class="goal-metric">2+ times</div>
            </div>
        </div>
    </div>
    """


def generate_weekly_schedule(data: dict, priorities: list) -> list:
    """Generate a 7-day schedule based on priorities."""
    
    stress = data.get("stress_level", 5)
    sleep_hours = data.get("sleep_hours", 7)
    sleep_quality = data.get("sleep_quality", 5)
    exercise = data.get("exercise_minutes", 30)
    work_hours = data.get("work_hours", 8)
    screen_time = data.get("screen_time_hours", 4)
    caffeine = data.get("caffeine_mg", 200)
    steps = data.get("steps_count", 5000)
    
    # Build personalized daily tasks based on priorities
    priority_features = {p["feature"] for p in priorities}
    
    # Morning routine (same every day)
    morning_routine = []
    if sleep_quality < 6 or sleep_hours < 7:
        morning_routine.append({"time": "7:00", "action": "Wake at consistent time (no snooze)", "category": "sleep"})
        morning_routine.append({"time": "7:15", "action": "10 min sunlight exposure", "category": "sleep"})
    
    if stress >= 6:
        morning_routine.append({"time": "7:30", "action": "5-min breathing exercise or meditation", "category": "stress"})
    
    # Work day structure
    work_tasks = []
    if work_hours > 9:
        work_tasks.append({"time": "12:00", "action": "Take a real lunch break (away from desk)", "category": "work"})
        work_tasks.append({"time": "17:00", "action": "Hard stop - close laptop, shutdown ritual", "category": "work"})
    
    if steps < 6000 or exercise < 30:
        work_tasks.append({"time": "12:30", "action": "15-min walk after lunch", "category": "exercise"})
    
    if screen_time > 5:
        work_tasks.append({"time": "19:00", "action": "Screen-free time begins", "category": "work"})
    
    # Evening routine
    evening_routine = []
    if sleep_quality < 6:
        evening_routine.append({"time": "21:00", "action": "No screens, dim lights", "category": "sleep"})
        evening_routine.append({"time": "21:30", "action": "Wind-down routine (reading, stretching)", "category": "sleep"})
    
    if caffeine > 200:
        evening_routine.append({"time": "14:00", "action": "No more caffeine after this time", "category": "sleep"})
    
    if stress >= 7:
        evening_routine.append({"time": "20:00", "action": "10-min evening meditation or journaling", "category": "stress"})
    
    evening_routine.append({"time": "22:30", "action": f"Bedtime (aim for {max(7, 8-int(work_hours-8))}h sleep)", "category": "sleep"})
    
    # Build week
    days = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for i, day_name in enumerate(day_names):
        is_weekend = i >= 5
        
        if is_weekend:
            # Weekend schedule - focus on recovery
            tasks = [
                {"time": "8:00", "action": "Sleep in (max 1h later than weekdays)", "category": "sleep"},
                {"time": "10:00", "action": "30-min exercise or outdoor activity", "category": "exercise"},
                {"time": "12:00", "action": "Social activity (friends, family, hobby)", "category": "social"},
            ]
            if stress >= 7:
                tasks.append({"time": "16:00", "action": "Nature time - park, hike, or garden", "category": "stress"})
            tasks.append({"time": "21:00", "action": "Prepare for week ahead (light planning)", "category": "work"})
        else:
            # Weekday schedule
            tasks = morning_routine.copy()
            tasks.extend(work_tasks)
            tasks.extend(evening_routine)
            
            # Add variety
            if i == 0:  # Monday
                tasks.insert(2, {"time": "8:00", "action": "Set 3 priority tasks for the week", "category": "work"})
            elif i == 2:  # Wednesday
                tasks.insert(2, {"time": "18:00", "action": "Mid-week social connection (call a friend)", "category": "social"})
            elif i == 4:  # Friday
                tasks.insert(2, {"time": "17:00", "action": "Weekly review - celebrate wins", "category": "work"})
        
        # Sort by time
        tasks.sort(key=lambda x: x["time"])
        
        days.append({"name": day_name, "tasks": tasks[:6]})  # Max 6 tasks per day
    
    return days


def generate_weekly_goals(data: dict, priorities: list) -> list:
    """Generate measurable weekly goals based on priorities."""
    
    goals = []
    
    stress = data.get("stress_level", 5)
    sleep_hours = data.get("sleep_hours", 7)
    sleep_quality = data.get("sleep_quality", 5)
    exercise = data.get("exercise_minutes", 30)
    work_hours = data.get("work_hours", 8)
    steps = data.get("steps_count", 5000)
    caffeine = data.get("caffeine_mg", 200)
    screen_time = data.get("screen_time_hours", 4)
    
    # Sleep goals
    if sleep_hours < 7:
        target = min(sleep_hours + 1, 8)
        goals.append({
            "text": f"Increase sleep from {sleep_hours:.0f}h to {target:.0f}h per night",
            "metric": f"{target:.0f}h × 7 nights"
        })
    
    if sleep_quality < 6:
        goals.append({
            "text": "No screens 1 hour before bed",
            "metric": "5/7 nights"
        })
    
    # Exercise goals
    if exercise < 30:
        target = min(exercise + 15, 45)
        goals.append({
            "text": f"Exercise {target:.0f} minutes daily (up from {exercise:.0f})",
            "metric": f"{target * 7:.0f} min/week"
        })
    
    if steps < 7000:
        target = min(steps + 2000, 8000)
        goals.append({
            "text": f"Reach {target:.0f} steps daily",
            "metric": f"{target:.0f} steps × 7"
        })
    
    # Stress goals
    if stress >= 7:
        goals.append({
            "text": "Practice stress management (breathing, meditation)",
            "metric": "10 min × 5 days"
        })
    
    # Work goals
    if work_hours > 9:
        target = max(work_hours - 1, 8)
        goals.append({
            "text": f"Reduce work hours from {work_hours:.0f}h to {target:.0f}h daily",
            "metric": f"Leave by {18 - int(work_hours - target)}:00"
        })
    
    # Caffeine
    if caffeine > 300:
        target = caffeine - 100
        goals.append({
            "text": f"Reduce caffeine from {caffeine:.0f}mg to {target:.0f}mg",
            "metric": "Skip 1 coffee"
        })
    
    # Screen time
    if screen_time > 5:
        target = screen_time - 1
        goals.append({
            "text": f"Reduce leisure screen time to {target:.0f}h",
            "metric": f"{target:.0f}h/day max"
        })
    
    # Always include social
    goals.append({
        "text": "Connect with a friend or loved one",
        "metric": "2+ times/week"
    })
    
    return goals[:6]  # Max 6 goals


def generate_contradictions_html(data: dict) -> str:
    """Generate HTML for contradiction warnings."""
    warnings = detect_contradictions(data)
    
    if not warnings:
        return ""
    
    warning_items = []
    for warning in warnings:
        # Determine if it's a critical warning (🚨) or regular (⚠️)
        is_critical = "🚨" in warning
        warning_class = "warning-critical" if is_critical else "warning-normal"
        
        warning_items.append(f'<div class="warning-item {warning_class}">{warning}</div>')
    
    return f"""
    <div class="contradictions-section">
        <h3>🔍 Data Consistency Check</h3>
        <p class="contradictions-intro">We noticed some patterns in your responses that may need attention:</p>
        {"".join(warning_items)}
    </div>
    """


def generate_heatmap_html(weekly_data: list) -> str:
    """
    Generate HTML heatmap showing 7 days of behavioral data.
    
    Args:
        weekly_data: List of 7 dicts, each containing feature values for one day
    
    Returns:
        HTML string with heatmap table
    """
    # Features to show in heatmap with healthy ranges (min_healthy, max_healthy)
    features_config = {
        'sleep_hours': ('😴 Sleep', 7, 9),
        'sleep_quality': ('💤 Sleep Quality', 7, 10),
        'work_hours': ('💼 Work Hours', 6, 9),
        'exercise_minutes': ('🏃 Exercise', 30, 120),
        'stress_level': ('😰 Stress', 0, 4),  # Lower is better
        'mood_score': ('😊 Mood', 7, 10),
        'energy_level': ('⚡ Energy', 7, 10),
        'caffeine_mg': ('☕ Caffeine', 0, 300),
        'screen_time_hours': ('📱 Screen Time', 0, 4),  # Lower is better
        'steps_count': ('👟 Steps', 7000, 15000),
        'meetings_count': ('📅 Meetings', 0, 5),  # Lower is better
    }
    
    # Features where lower is better
    lower_is_better = {'stress_level', 'caffeine_mg', 'screen_time_hours', 'meetings_count', 'work_hours'}
    
    def get_risk_class(feature, value):
        """Determine risk level CSS class based on value."""
        if feature not in features_config:
            return 'risk-healthy'
        
        _, min_h, max_h = features_config[feature]
        
        if feature in lower_is_better:
            # For features where lower is better
            if value <= max_h:
                return 'risk-healthy'
            elif value <= max_h * 1.3:
                return 'risk-mild'
            elif value <= max_h * 1.6:
                return 'risk-warning'
            else:
                return 'risk-high'
        else:
            # For features where higher is better
            if value >= min_h:
                return 'risk-healthy'
            elif value >= min_h * 0.7:
                return 'risk-mild'
            elif value >= min_h * 0.5:
                return 'risk-warning'
            else:
                return 'risk-high'
    
    # Day labels
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    
    # Build table header
    header_cells = '<th>Behavior</th>' + ''.join(f'<th>{day}</th>' for day in days)
    
    # Build table rows
    rows = []
    for feature, (label, _, _) in features_config.items():
        cells = [f'<td class="heatmap-feature">{label}</td>']
        
        for day_idx, day_data in enumerate(weekly_data):
            value = day_data.get(feature, 0)
            risk_class = get_risk_class(feature, value)
            
            # Format value based on feature
            if feature in ['steps_count']:
                display_val = f'{value/1000:.1f}k'
            elif feature in ['caffeine_mg']:
                display_val = f'{value:.0f}mg'
            elif feature in ['sleep_hours', 'work_hours', 'screen_time_hours']:
                display_val = f'{value:.1f}h'
            elif feature in ['exercise_minutes']:
                display_val = f'{value:.0f}m'
            else:
                display_val = f'{value:.1f}'
            
            cells.append(f'<td class="{risk_class}">{display_val}</td>')
        
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    
    return f"""
    <div class="card heatmap-section">
        <h2>🔥 Your 7-Day Behavioral Pattern</h2>
        <p class="heatmap-intro">See how your daily habits varied throughout the week. Colors indicate how each metric compares to healthy targets.</p>
        
        <div class="heatmap-container">
            <table class="heatmap-table">
                <thead>
                    <tr>{header_cells}</tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        
        <div class="heatmap-legend">
            <div class="legend-item">
                <div class="legend-color risk-healthy"></div>
                <span>✅ Healthy</span>
            </div>
            <div class="legend-item">
                <div class="legend-color risk-mild"></div>
                <span>🟡 Mild Concern</span>
            </div>
            <div class="legend-item">
                <div class="legend-color risk-warning"></div>
                <span>🟠 Warning</span>
            </div>
            <div class="legend-item">
                <div class="legend-color risk-high"></div>
                <span>🔴 High Risk</span>
            </div>
        </div>
    </div>
    """


def generate_metrics_html(data: dict) -> str:
    """Generate HTML for metrics grid."""
    metrics = [
        ("Stress", data.get("stress_level", 5), "/10", 5, "lower"),
        ("Sleep", data.get("sleep_hours", 7), "hrs", 7, "higher"),
        ("Work", data.get("work_hours", 8), "hrs", 8, "lower"),
        ("Exercise", data.get("exercise_minutes", 30), "min", 30, "higher"),
        ("Mood", data.get("mood_score", 6), "/10", 6, "higher"),
        ("Energy", data.get("energy_level", 6), "/10", 6, "higher"),
        # === NEW V2 METRICS ===
        ("Morning Mood", data.get("morning_mood", 3), "/5", 3, "higher"),
        ("Productivity", data.get("productivity_today", 3), "/5", 3, "higher"),
        ("Day Rating", data.get("day_overall_rating", 6), "/10", 6, "higher"),
    ]
    
    html_parts = []
    for name, value, unit, target, direction in metrics:
        if direction == "lower":
            is_good = value <= target
        else:
            is_good = value >= target
        
        status = "✅" if is_good else "⚠️"
        
        html_parts.append(f"""
        <div class="metric">
            <div class="metric-status">{status}</div>
            <div class="metric-value">{value:.0f}{unit}</div>
            <div class="metric-label">{name}</div>
        </div>
        """)
    
    return "\n".join(html_parts)


def generate_single_report(data: dict, name: str, model, feature_cols: list, output_path: str) -> None:
    """Generate a single HTML report for one person."""
    # Create sequence and predict
    sequence = create_weekly_sequence(data, feature_cols)
    pred_class, probs = predict(model, sequence, feature_cols)
    
    # Handle binary (2-class) vs 3-class models
    is_binary = len(probs) == 2
    
    if is_binary:
        # Binary model: 0=Low Risk, 1=High Risk
        # Map to display: show Low and High, set Medium to 0
        risk_mapping = {0: 0, 1: 2}  # Map class 1 to "high" display
        display_class = risk_mapping.get(pred_class, pred_class)
        low_prob = probs[0] * 100
        med_prob = 0  # No medium class in binary
        high_prob = probs[1] * 100
    else:
        # 3-class model: 0=Low, 1=Medium, 2=High
        display_class = pred_class
        low_prob = probs[0] * 100
        med_prob = probs[1] * 100
        high_prob = probs[2] * 100
    
    risk = RISK_LEVELS[display_class]
    risk_classes = {0: "low", 1: "medium", 2: "high"}
    
    # Generate result card
    result_html = RESULT_CARD_TEMPLATE.replace("{emoji}", risk["emoji"])
    result_html = result_html.replace("{risk_name}", risk["name"])
    result_html = result_html.replace("{risk_class}", risk_classes[display_class])
    result_html = result_html.replace("{confidence:.0f}", f"{probs[pred_class] * 100:.0f}")
    result_html = result_html.replace("{description}", risk["desc"])
    result_html = result_html.replace("{low_prob:.0f}", f"{low_prob:.0f}")
    result_html = result_html.replace("{low_prob}", f"{low_prob:.0f}")
    result_html = result_html.replace("{med_prob:.0f}", f"{med_prob:.0f}")
    result_html = result_html.replace("{med_prob}", f"{med_prob:.0f}")
    result_html = result_html.replace("{high_prob:.0f}", f"{high_prob:.0f}")
    result_html = result_html.replace("{high_prob}", f"{high_prob:.0f}")
    result_html = result_html.replace("{metrics}", generate_metrics_html(data))
    result_html = result_html.replace("{recommendations}", generate_recommendations_html(data, pred_class, model, feature_cols))
    
    # Add name to the report
    name_html = f'<div class="person-name">Report for: <strong>{name}</strong></div>'
    
    # Add contradictions check (if any)
    contradictions_html = generate_contradictions_html(data)
    
    # Add CVAE AI suggestions (if available and not clearly low risk)
    cvae_html = generate_cvae_suggestions_html(data, pred_class, probs)
    
    # Add 7-Day Behavioral Heatmap
    # Convert sequence array back to list of dicts for heatmap
    weekly_data = []
    for day_idx in range(len(sequence)):
        day_dict = {}
        for feat_idx, feature in enumerate(feature_cols):
            day_dict[feature] = sequence[day_idx, feat_idx]
        weekly_data.append(day_dict)
    heatmap_html = generate_heatmap_html(weekly_data)
    
    # Add 7-Day Action Plan
    action_plan_html = generate_action_plan_html(data, pred_class)
    
    result_html = name_html + result_html + heatmap_html + contradictions_html + cvae_html + action_plan_html
    
    # Generate full HTML
    html = HTML_TEMPLATE.replace("{date}", datetime.now().strftime("%B %d, %Y at %H:%M"))
    html = html.replace("{results}", result_html)
    
    # Save
    Path(output_path).write_text(html)


def generate_report(csv_path: str, output_path: str, model_path: str = None) -> None:
    """Generate HTML reports from CSV data - one per person."""
    import re
    
    # Load model
    if model_path is None:
        model_path = PROJECT_ROOT / "models/saved/lstm_sequence.pt"
    model, _, feature_cols, _ = load_model(model_path)
    
    # Parse CSV
    df, _ = parse_google_form_csv(csv_path)
    
    # Check if there's a name column
    name_col = None
    if "_name" in df.columns:
        name_col = "_name"
    elif "_email" in df.columns:
        name_col = "_email"
    
    output_path = Path(output_path)
    output_dir = output_path.parent
    base_name = output_path.stem
    
    # Generate report for each person
    generated_files = []
    
    for idx, row in df.iterrows():
        data = {f: row[f] for f in feature_cols if f in row}
        
        # Get person's name
        if name_col and pd.notna(row.get(name_col)):
            person_name = str(row[name_col]).strip()
        else:
            person_name = f"Person {idx + 1}"
        
        # Create safe filename from name
        safe_name = re.sub(r'[^\w\s-]', '', person_name).strip().lower()
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        
        if len(df) == 1:
            # Single person - use provided output path
            file_path = output_path
        else:
            # Multiple people - create individual files
            file_path = output_dir / f"{safe_name}_report.html"
        
        generate_single_report(data, person_name, model, feature_cols, str(file_path))
        generated_files.append((person_name, file_path))
    
    # Print summary
    print(f"\n✅ Generated {len(generated_files)} report(s):")
    for name, path in generated_files:
        print(f"   📄 {name}: {path}")
    
    if len(generated_files) > 1:
        print(f"\n💡 All reports saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML burnout report")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", default="burnout_report.html", help="Output HTML file (or directory base name for multiple)")
    parser.add_argument("--model-path", default=None, help="Path to .pt model file")
    
    args = parser.parse_args()
    generate_report(args.csv, args.output, args.model_path)


if __name__ == "__main__":
    main()
