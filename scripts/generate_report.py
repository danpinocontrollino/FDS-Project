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


def generate_recommendations_html(data: dict, pred_class: int) -> str:
    """Generate HTML list items for recommendations."""
    recommendations = []
    
    if data.get("sleep_hours", 7) < 7:
        recommendations.append("😴 Try to get at least 7 hours of sleep. Sleep debt compounds over time.")
    
    if data.get("stress_level", 5) > 6:
        recommendations.append("🧘 High stress detected. Consider daily meditation or breathing exercises.")
    
    if data.get("exercise_minutes", 30) < 20:
        recommendations.append("🏃 Aim for at least 20-30 minutes of exercise daily. Even walking helps!")
    
    if data.get("work_hours", 8) > 9:
        recommendations.append("⏰ Working long hours increases burnout risk. Try to set boundaries.")
    
    if data.get("caffeine_mg", 100) > 300:
        recommendations.append("☕ High caffeine intake can affect sleep. Consider reducing after 2 PM.")
    
    if data.get("screen_time_hours", 4) > 6:
        recommendations.append("📱 High screen time. Take regular breaks and try the 20-20-20 rule.")
    
    if data.get("mood_score", 6) < 5:
        recommendations.append("💬 Low mood persisting? Consider talking to someone you trust.")
    
    if not recommendations:
        recommendations.append("✨ Your habits look healthy! Keep maintaining this balance.")
    
    return "\n".join(f"<li>{rec}</li>" for rec in recommendations)


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


def generate_report(csv_path: str, output_path: str, model_path: str = None) -> None:
    """Generate HTML report from CSV data."""
    # Load model
    if model_path is None:
        model_path = "models/saved/lstm_sequence.pt"
    model, _ = load_model(model_path)
    
    # Parse CSV
    df = parse_google_form_csv(csv_path)
    
    # Generate results for each row (use first row for single report)
    row = df.iloc[0]
    data = row.to_dict()
    
    # Create sequence and predict
    sequence = create_weekly_sequence(data)
    pred_class, probs = predict(model, sequence)
    
    risk = RISK_LEVELS[pred_class]
    risk_classes = {0: "low", 1: "medium", 2: "high"}
    
    # Generate result card
    result_html = RESULT_CARD_TEMPLATE.format(
        emoji=risk["emoji"],
        risk_name=risk["name"],
        risk_class=risk_classes[pred_class],
        confidence=probs[pred_class] * 100,
        description=risk["desc"],
        low_prob=probs[0] * 100,
        med_prob=probs[1] * 100,
        high_prob=probs[2] * 100,
        metrics=generate_metrics_html(data),
        recommendations=generate_recommendations_html(data, pred_class),
    )
    
    # Generate full HTML
    html = HTML_TEMPLATE.format(
        date=datetime.now().strftime("%B %d, %Y at %H:%M"),
        results=result_html,
    )
    
    # Save
    Path(output_path).write_text(html)
    print(f"✅ Report saved to {output_path}")
    print(f"   Open in browser: file://{Path(output_path).absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML burnout report")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", default="burnout_report.html", help="Output HTML file")
    parser.add_argument("--model-path", default=None, help="Path to .pt model file")
    
    args = parser.parse_args()
    generate_report(args.csv, args.output, args.model_path)


if __name__ == "__main__":
    main()
