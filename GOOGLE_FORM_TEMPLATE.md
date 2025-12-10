# Google Form Template - Copy and Paste

## 📋 Form Title
```
Daily Mental Health & Behavior Tracker
```

## 📝 Form Description
```
Please fill this form once per day for at least 7 consecutive days. 
This data will be used to generate your personalized mental health report 
with evidence-based interventions and interactive visualizations.

All responses are confidential and will only be used for report generation.
```

---

## 🔧 Form Settings

### Before adding questions:
1. Click ⚙️ (Settings) in top right
2. ✅ Check "Collect email addresses"
3. ✅ Check "Limit to 1 response" (if using daily format)
4. ✅ Check "Respondents can edit after submit" (recommended)
5. Click "Save"

---

## 📝 Questions to Add (Copy these exactly)

### Question 1: Email
- **Question:** Email Address
- **Type:** Short answer
- **Validation:** Response validation → Text → Email address
- **Required:** ✅ Yes
- **Description:** "Your email will be used to identify your responses across days"

---

### Question 2: Date
- **Question:** What date is this entry for?
- **Type:** Date
- **Required:** ✅ Yes
- **Description:** "Today's date (or the date you're reporting data for)"

---

### Question 3: Sleep
- **Question:** How many hours did you sleep last night?
- **Type:** Short answer
- **Validation:** Response validation → Number → Between 0 and 24
- **Required:** ✅ Yes
- **Description:** "Total hours of sleep (e.g., 7.5). Include naps if they were significant."

---

### Question 4: Work
- **Question:** How many hours did you work today?
- **Type:** Short answer
- **Validation:** Response validation → Number → Between 0 and 24
- **Required:** ✅ Yes
- **Description:** "Total hours spent on work activities (e.g., 8). Include meetings, emails, and tasks."

---

### Question 5: Exercise
- **Question:** How many minutes of exercise did you do today?
- **Type:** Short answer
- **Validation:** Response validation → Number → Between 0 and 300
- **Required:** ✅ Yes
- **Description:** "Total minutes of physical exercise (e.g., 30). Include walking, gym, sports, yoga, etc."

---

### Question 6: Steps
- **Question:** Approximately how many steps did you take today?
- **Type:** Short answer
- **Validation:** Response validation → Number → Between 0 and 50000
- **Required:** ✅ Yes
- **Description:** "Approximate step count (e.g., 8000). Check your phone or fitness tracker if available. Estimate if not sure."

---

### Question 7: Screen Time
- **Question:** How many hours of non-work screen time did you have?
- **Type:** Short answer
- **Validation:** Response validation → Number → Between 0 and 24
- **Required:** ✅ Yes
- **Description:** "Hours of recreational screen time (e.g., 6). Include social media, streaming, gaming, browsing. Exclude work-related screen time."

---

### Question 8: Social
- **Question:** How many meaningful social interactions did you have today?
- **Type:** Short answer
- **Validation:** Response validation → Number → Between 0 and 100
- **Required:** ✅ Yes
- **Description:** "Count of quality interactions (e.g., 15). Include in-person conversations, phone/video calls, messaging conversations. Brief exchanges (hi/bye) don't count."

---

## ✅ Checklist Before Publishing

- [ ] Form title is descriptive
- [ ] Description explains purpose and duration (7+ days)
- [ ] "Collect email addresses" is enabled in Settings
- [ ] All 8 questions are added
- [ ] All questions are marked as Required
- [ ] Number validations are set correctly (ranges)
- [ ] Descriptions/help text are included
- [ ] Preview form to test
- [ ] Test by filling once yourself

---

## 📧 Distribution

### Option 1: Daily Reminder Email Template
```
Subject: Daily Mental Health Tracker - Day [X] of 7

Hi [Name],

Please take 2 minutes to fill today's mental health tracking form:
[FORM LINK]

We're collecting data for [X] more days to generate your personalized report.

Thank you!
```

### Option 2: Weekly Single Email Template
```
Subject: Weekly Mental Health Tracker - Retrospective

Hi [Name],

Please take 10 minutes to fill this week's mental health tracking form:
[FORM LINK]

This form collects 7 days of data at once. Think back over the past week
and provide your best estimates for each day.

Your comprehensive report will be generated within 24 hours of submission.

Thank you!
```

---

## 📊 Collecting Responses

### View Responses
1. Open your form in Google Forms
2. Click "Responses" tab at top
3. See summary statistics and individual responses

### Export to CSV
1. Click "Responses" tab
2. Click the three dots menu (⋮) in top right
3. Select "Download responses (.csv)"
4. Save to your `data/` folder
5. Name it something descriptive (e.g., `team_tracking_dec2024.csv`)

---

## 🚀 Generate Reports

Once you have 7+ days of data per person:

```bash
# Navigate to project directory
cd /workspaces/FDS-Project

# Generate reports for all respondents
python scripts/generate_html_from_form.py --csv data/your_responses.csv --all-users

# Output will be in reports/form_generated/
```

Each person will get a 60-80KB HTML file with:
- ✅ 5 interactive charts
- ✅ Comprehensive methodology
- ✅ 8 mental health predictions
- ✅ 6-10 personalized interventions
- ✅ 20-30 research citations

---

## 💡 Tips for Success

### For Participants
- Fill the form at the same time each day (e.g., before bed)
- Be honest - the system works best with accurate data
- Estimates are okay - don't stress about exact numbers
- Set a daily phone reminder
- The form takes 2 minutes maximum

### For Administrators
- Send daily reminder emails (for Option 1)
- Monitor response rates
- Follow up with non-respondents
- Explain the value of comprehensive reports
- Share sample anonymized reports to build interest

### Data Quality
- ✅ Consistent timing (same time each day)
- ✅ No missing days
- ✅ Honest estimates
- ✅ At least 7 days (more is better)
- ❌ Avoid guessing wildly
- ❌ Don't rush through questions
- ❌ Don't skip days

---

## 📱 Mobile-Friendly

Google Forms works great on phones. Participants can:
- Fill from anywhere
- Takes 2 minutes
- All responses saved
- Can edit after submit (if enabled)

Encourage daily completion from phone during evening routine.

---

## 🎯 Sample Timeline

### Week 1: Setup
- **Monday:** Create form using this template
- **Monday:** Test form yourself
- **Tuesday:** Send invitation to participants
- **Tuesday-Monday:** Collect daily responses (7 days)

### Week 2: Reporting
- **Tuesday:** Export CSV after 7 days complete
- **Tuesday:** Generate reports
- **Wednesday:** Distribute individual HTML reports
- **Thursday:** Schedule follow-up meetings
- **Friday:** Discuss interventions with at-risk individuals

### Weeks 3-4: Intervention Period
- Participants implement recommended changes
- Track compliance with interventions

### Week 5+: Reassessment
- Run form again for 7 days
- Compare new reports to baseline
- Measure improvement

---

## 📖 Example Form URLs (Inspiration)

While you should create your own form, here are similar forms for reference:
- [Daily Wellness Tracker](https://www.google.com/search?q=google+forms+daily+wellness+tracker)
- [Sleep and Productivity Log](https://www.google.com/search?q=google+forms+sleep+tracker)
- [Mental Health Check-In](https://www.google.com/search?q=google+forms+mental+health+check-in)

---

## 🔒 Privacy Settings

### Recommended Settings
- ✅ Only collect email (required for identification)
- ✅ Don't collect names separately (email is sufficient)
- ✅ Inform participants about data usage
- ✅ Store CSV files securely
- ✅ Use institutional Google account (not personal)

### What NOT to collect
- ❌ Full name (email is enough)
- ❌ Phone number
- ❌ Address
- ❌ ID numbers
- ❌ Detailed medical history

---

## ✅ You're Ready!

Copy the 8 questions above into a new Google Form and start collecting data.

**Questions?** Check `GOOGLE_FORM_INSTRUCTIONS.md` for detailed help.

**Technical issues?** See troubleshooting section in main documentation.

**Ready to generate reports?** See `GOOGLE_FORM_QUICK_START.md` for commands.

---

**Form creation time:** ~10 minutes  
**Daily completion time:** ~2 minutes  
**Report generation time:** ~5 seconds per person  
**Value delivered:** Comprehensive personalized mental health insights ✨
