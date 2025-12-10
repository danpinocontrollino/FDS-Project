# Google Form to HTML Report - Quick Reference

## ✅ System is Working!

Generate comprehensive 60-80KB HTML mental health reports from Google Form CSV exports.

---

## 🚀 Three-Step Process

### 1. Create Google Form
Copy the form template from **`GOOGLE_FORM_INSTRUCTIONS.md`** (see "Example Google Form Setup" section)

**Required questions (8 total):**
1. Email
2. Date
3. Sleep Hours
4. Work Hours
5. Exercise Minutes
6. Steps Count
7. Screen Time Hours
8. Social Interactions

### 2. Collect Data
- Minimum: 7 consecutive days per person
- Daily form (fill once per day) OR weekly form (fill once with 7 days)

### 3. Generate Reports

```bash
# Process all users
python scripts/generate_html_from_form.py --csv your_responses.csv --all-users

# Single user
python scripts/generate_html_from_form.py --csv your_responses.csv --user-id user@example.com

# Weekly format
python scripts/generate_html_from_form.py --csv your_responses.csv --all-users --format weekly
```

**Output:** `reports/form_generated/comprehensive_profile_[user]_[timestamp].html`

---

## 📊 What's in the Reports

✅ **5 Interactive Chart.js visualizations**
✅ **Comprehensive methodology explanation**
✅ **Stage 1 behavioral predictions** (next-day forecasts)
✅ **Stage 2 mental health scores** (8 metrics: stress, mood, energy, focus, PSS-10, GAD-7, PHQ-9, job satisfaction)
✅ **6-10 personalized evidence-based interventions**
✅ **Trend analysis** across all days
✅ **Research citations** (20-30 per report)

---

## 📝 Examples

Test with existing data:

```bash
# Daily format (3 users, 7-9 days each)
python scripts/generate_html_from_form.py --csv data/daily_tracking.csv --all-users

# Weekly format (4 users, 7 days each)
python scripts/generate_html_from_form.py --csv data/martina_7day.csv --all-users
```

---

## 📚 Full Documentation

See **`GOOGLE_FORM_INSTRUCTIONS.md`** for:
- Complete form setup guide
- CSV format specifications
- Troubleshooting
- Privacy & ethics guidelines
- Example workflows

---

## ✨ Key Features

- **Auto-detects CSV format** (daily vs weekly)
- **Flexible column names** (handles variations)
- **Self-contained HTML** (no external dependencies except Chart.js CDN)
- **Works offline** after initial load
- **60-80KB reports** with full interactivity
- **~1,300 lines of HTML** per report

---

## 🎯 Quick Test

```bash
# Generate a report from sample data
python scripts/generate_html_from_form.py --csv data/daily_tracking.csv --user-id sarah@example.com

# Open the report
open reports/form_generated/comprehensive_profile_sarah_example_com_*.html
```

The report should:
- Open in your browser
- Display 5 interactive charts
- Show 6-10 personalized interventions
- Include full methodology section
- Be 60-80KB in size

---

**System Ready! 🎉** Collect data via Google Forms and generate comprehensive mental health reports.
