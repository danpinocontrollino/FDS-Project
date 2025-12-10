# Two-Stage Pipeline HTML Report Generation

## Overview

Generate professional HTML reports from the two-stage hybrid prediction pipeline. Reports clearly distinguish between:
- **Stage 1 (Real Data)**: Behavioral forecasting from StudentLife dataset
- **Stage 2 (Synthetic Data)**: Mental health inference from synthetic patterns

## Quick Start

### Generate Reports from Existing Predictions

```bash
# Generate report for specific student
python scripts/generate_two_stage_html.py --student u00

# Generate reports for all students
python scripts/generate_two_stage_html.py --all-students

# Specify custom output directory
python scripts/generate_two_stage_html.py --all-students --output-dir reports/custom_folder
```

### What's Included in Each Report

✅ **Stage 1: Behavioral Predictions** (with uncertainty percentages)
- Sleep hours
- Exercise minutes
- Screen time
- Social interactions
- Steps count

✅ **Stage 2: Mental Health Predictions** (clipped to valid ranges)
- Stress Level (0-10)
- Mood Score (0-10)
- Energy Level (0-10)
- Focus Score (0-10)
- Perceived Stress Scale (0-40)
- Anxiety Score (0-21)
- Depression Score (0-27)
- Job Satisfaction (0-10)

✅ **Risk Assessment**
- At-risk factors (exceeding clinical thresholds)
- Healthy factors (within normal ranges)

✅ **Evidence-Based Recommendations**
- Sleep optimization
- Exercise increase
- Screen time reduction
- Social connection building

✅ **Clear Warnings & Limitations**
- Data source labels (REAL vs SYNTHETIC)
- Uncertainty propagation explained
- Clinical disclaimer

## File Structure

```
reports/two_stage_html/
├── two_stage_profile_u00.html    # Student u00 report
├── two_stage_profile_u01.html    # Student u01 report
└── ...                            # One HTML per student
```

## Key Features

### 1. Data Source Transparency

Every section clearly labeled:
- 🟦 **Stage 1** with `REAL DATA` badge → StudentLife patterns
- 🟩 **Stage 2** with `SYNTHETIC DATA` badge → Synthetic training

### 2. Uncertainty Quantification

- Stage 1 predictions show uncertainty as **percentages** (±11%, ±32%, etc.)
- Average uncertainty displayed prominently
- Warning about error propagation to Stage 2

### 3. Professional Design

- Clean, modern UI with gradient header
- Color-coded risk indicators (red = at-risk, green = healthy)
- Print-friendly layout
- Mobile responsive
- Self-contained HTML (no external dependencies)

### 4. Clinical Validation

- Evidence-based recommendations
- Peer-reviewed research citations
- Clinical thresholds for risk assessment
- Comprehensive disclaimer

## Understanding the Reports

### Stage 1 Uncertainties

Example: **Sleep: 6.5h ±11%**
- Model predicts 6.5 hours of sleep tomorrow
- Uncertainty: ±11% (roughly ±0.7 hours)
- Based on real student behavioral patterns

### Stage 2 Predictions

Example: **Stress Level: 6.5/10 ⚠️**
- Inferred from Stage 1 behavioral predictions
- Trained on synthetic data
- >7/10 = at-risk threshold

### Recommendations

Priority-based:
- **HIGH**: Sleep < 7h, Social < 10 interactions
- **MEDIUM**: Exercise < 30min, Screen time > 8h

Each includes:
- Current vs target metrics
- Scientific evidence
- Specific actionable steps

## Customization Options

### Adjust Risk Thresholds

Edit `RISK_THRESHOLDS` in `generate_two_stage_html.py`:

```python
RISK_THRESHOLDS = {
    "stress_level": 7.0,       # >7/10 = high stress
    "anxiety_score": 10.0,     # >10/21 = moderate anxiety
    "depression_score": 14.0,  # >14/27 = moderate depression
    # ... customize as needed
}
```

### Add Custom Recommendations

Modify `generate_recommendations()` function to add domain-specific advice.

### Change Color Scheme

Edit CSS in `generate_html_report()` function (line ~190).

## Comparison with Single-Model Reports

| Feature | Single-Model (`generate_profile.py`) | Two-Stage (`generate_two_stage_html.py`) |
|---------|--------------------------------------|------------------------------------------|
| **Training Data** | 1.5M synthetic | Stage 1: 674 real + Stage 2: 1.5M synthetic |
| **Predictions** | 8 mental health targets | 5 behavioral + 8 mental health |
| **Uncertainty** | Not quantified | Stage 1 uncertainties as % |
| **Data Source Labels** | Not specified | Clear REAL vs SYNTHETIC badges |
| **Error Propagation** | Not applicable | Explicitly shown and explained |
| **Transparency** | Black-box predictions | Two-stage pipeline visible |

## When to Use Which System

### Use Two-Stage Reports When:
- ✅ You want to demonstrate pipeline transparency
- ✅ Behavioral forecasting is valuable on its own
- ✅ You need to show error propagation
- ✅ Comparing real vs synthetic data impact

### Use Single-Model Reports When:
- ✅ You have direct mental health labels
- ✅ End-to-end prediction is sufficient
- ✅ Job-specific advice is needed
- ✅ Contradiction detection is important

## Technical Details

### Input Requirements

Script reads from: `models/saved/two_stage_predictions.json`

Required JSON structure:
```json
{
  "metadata": {
    "num_students": 10,
    "total_predictions": 598
  },
  "predictions": [
    {
      "student_id": "u00",
      "date": "2013-03-30",
      "stage1_behavioral_predictions": { ... },
      "stage1_uncertainties": { ... },
      "stage2_mental_health_predictions": { ... }
    }
  ]
}
```

### Dependencies

- Python 3.7+
- numpy (for calculations)
- Standard library only (no external HTML frameworks)

### Performance

- ~0.5s per student report
- ~25KB per HTML file
- All 10 students: <5 seconds total

## Troubleshooting

### "File not found: models/saved/two_stage_predictions.json"

**Solution**: Ensure you've copied the correct predictions JSON:
```bash
cp 'models/saved/two_stage_predictions (1).json' models/saved/two_stage_predictions.json
```

### "Student ID not found"

**Solution**: List available students:
```bash
python -c "import json; print(set(p['student_id'] for p in json.load(open('models/saved/two_stage_predictions.json'))['predictions']))"
```

### Reports show all 5.0 values

**Solution**: This is a different comparison view. The two-stage reports use the correct JSON with proper decimal values.

## Future Enhancements

Possible additions:
- [ ] Interactive charts (Chart.js integration)
- [ ] Temporal trends across all days
- [ ] Comparison with previous reports
- [ ] Export to PDF option
- [ ] Multi-user summary dashboard
- [ ] Integration with Google Forms for direct CSV input

## References

- **StudentLife Dataset**: Dartmouth College, 2013 (Stage 1 training)
- **Synthetic Data**: 1.5M records, controlled patterns (Stage 2 training)
- **Clinical Thresholds**: Based on PSS, GAD-7, PHQ-9 scales

## Support

For issues or questions:
1. Check this README
2. Review generated HTML in browser
3. Inspect console for JavaScript errors
4. Verify JSON structure matches expected format

---

**Last Updated**: December 10, 2025
**Version**: 1.0
**Status**: Production Ready ✅
