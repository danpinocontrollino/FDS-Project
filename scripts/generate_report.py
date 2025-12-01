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
        
        @media (max-width: 600px) {
            .header h1 { font-size: 1.8rem; }
            .risk-badge { font-size: 1.2rem; padding: 10px 25px; }
            .emoji { font-size: 3rem; }
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


def generate_metrics_html(data: dict) -> str:
    """Generate HTML for metrics grid."""
    metrics = [
        ("Stress", data.get("stress_level", 5), "/10", 5, "lower"),
        ("Sleep", data.get("sleep_hours", 7), "hrs", 7, "higher"),
        ("Work", data.get("work_hours", 8), "hrs", 8, "lower"),
        ("Exercise", data.get("exercise_minutes", 30), "min", 30, "higher"),
        ("Mood", data.get("mood_score", 6), "/10", 6, "higher"),
        ("Energy", data.get("energy_level", 6), "/10", 6, "higher"),
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
    
    risk = RISK_LEVELS[pred_class]
    risk_classes = {0: "low", 1: "medium", 2: "high"}
    
    # Generate result card
    result_html = RESULT_CARD_TEMPLATE.replace("{emoji}", risk["emoji"])
    result_html = result_html.replace("{risk_name}", risk["name"])
    result_html = result_html.replace("{risk_class}", risk_classes[pred_class])
    result_html = result_html.replace("{confidence:.0f}", f"{probs[pred_class] * 100:.0f}")
    result_html = result_html.replace("{description}", risk["desc"])
    result_html = result_html.replace("{low_prob:.0f}", f"{probs[0] * 100:.0f}")
    result_html = result_html.replace("{low_prob}", f"{probs[0] * 100:.0f}")
    result_html = result_html.replace("{med_prob:.0f}", f"{probs[1] * 100:.0f}")
    result_html = result_html.replace("{med_prob}", f"{probs[1] * 100:.0f}")
    result_html = result_html.replace("{high_prob:.0f}", f"{probs[2] * 100:.0f}")
    result_html = result_html.replace("{high_prob}", f"{probs[2] * 100:.0f}")
    result_html = result_html.replace("{metrics}", generate_metrics_html(data))
    result_html = result_html.replace("{recommendations}", generate_recommendations_html(data, pred_class, model, feature_cols))
    
    # Add name to the report
    name_html = f'<div class="person-name">Report for: <strong>{name}</strong></div>'
    
    # Add contradictions check (if any)
    contradictions_html = generate_contradictions_html(data)
    
    result_html = name_html + result_html + contradictions_html
    
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
    model, _, feature_cols = load_model(model_path)
    
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
