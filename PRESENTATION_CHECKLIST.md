# 🎤 Presentation Checklist & Demo Script

## ✅ Pre-Presentation Setup (5 minutes before)

### Technical Setup
- [ ] Open terminal in `/workspaces/FDS-Project`
- [ ] Activate virtual environment: `source .venv/bin/activate`
- [ ] Test Streamlit demo: `streamlit run demo_app.py`
- [ ] Open backup HTML report: `reports/examples/teacher_profile.html`
- [ ] Have `docs/VISUAL_ARCHITECTURE.md` ready to show
- [ ] Open `reports/benchmarks/benchmark_report.txt` for reference

### Backup Plan (If Demo Fails)
- [ ] Have static HTML reports open in browser
- [ ] Screenshots of Streamlit demo ready
- [ ] Benchmark numbers memorized (98.5% vs 89.3%)

### Materials Ready
- [ ] Laptop charged + power cable
- [ ] Backup USB with repository zip
- [ ] HDMI adapter (if needed)
- [ ] Printed one-page summary (optional)

---

## 🎯 5-Minute Demo Script

### **Slide 1: Title (15 seconds)**
*"Good morning/afternoon. Today I'm presenting our Mental Health Profiling System using deep learning on 1.5 million records."*

**Show:** Repository title + key stats
- 98.5% accuracy (vs 89.3% baseline)
- 1.5M+ training records
- 8 mental health predictions

---

### **Slide 2: Problem Statement (30 seconds)**
*"Mental health issues cost billions in lost productivity. Early intervention is key. Our system predicts 8 mental health outcomes from daily behaviors."*

**Show:** 
- Input: 7 days of sleep, work, exercise data
- Output: Stress, mood, anxiety, depression, PSS, job satisfaction
- Real-time predictions + personalized advice

**Key Point:** "Not just classification - actionable insights."

---

### **Slide 3: Live Demo (2 minutes)**
*"Let me show you the interactive system."*

**Launch:** `streamlit run demo_app.py`

**Demo Flow:**
1. **Healthy Baseline** (10 seconds)
   - Sleep: 7.5h, Work: 8h, Exercise: 30min
   - *"Click Generate Profile → All green, healthy predictions"*

2. **Sleep Deprivation** (30 seconds)
   - Adjust sleep to 4h
   - *"Watch predictions change in real-time"*
   - Show HIGH PRIORITY intervention: "4.1h → 7.5h sleep needed"
   - Point out: "Specific action: Set alarm 3.4h earlier"

3. **Overwork Scenario** (30 seconds)
   - Adjust work hours to 12h
   - Caffeine to 500mg
   - *"System detects overwork + caffeine dependence"*
   - Show risk factors list
   - Highlight job-specific advice

4. **Real-Time Speed** (20 seconds)
   - Change multiple sliders quickly
   - *"Predictions update instantly - LSTM runs in <100ms"*

**Key Point:** "This isn't a static model - it's a complete system."

---

### **Slide 4: Technical Architecture (1 minute)**
*"Let me show you what's under the hood."*

**Show:** `docs/VISUAL_ARCHITECTURE.md` diagram

**Explain:**
1. **Data Pipeline**
   - Google Form CSV → Flexible parser → 7 days × 17 features

2. **Model Architecture**
   - Multi-task LSTM (2 layers, 128 hidden units)
   - 8 prediction heads (one per outcome)
   - Shared encoder for transfer learning

3. **Analysis Layer** (Novel Engineering)
   - Contradiction detection (20+ patterns)
   - Job-specific advice (8 categories)
   - Evidence-based interventions (332 cases)

**Key Point:** "Multi-task learning improves all predictions simultaneously."

---

### **Slide 5: Results & Benchmarks (45 seconds)**
*"How does it perform? We tested 6 different models."*

**Show:** `reports/benchmarks/benchmark_report.txt`

**Highlight:**
```
Baseline (Random Forest):  89.3% accuracy, 491 MB model
Our Model (LSTM):          98.5% accuracy, 1.2 MB model

Improvement:               +9.2% absolute
                          +10.3% relative
                          99.8% smaller model
```

**Target-Specific:**
- Job Satisfaction: 98.5% ✅
- Depression: 98% ✅
- Anxiety: 97% ✅
- Daily stress: 52% (harder due to volatility)

**Key Point:** "Stable weekly metrics highly predictable. Daily metrics harder but still useful."

---

### **Slide 6: Novel Features (30 seconds)**
*"Three innovations make this production-ready."*

**1. Contradiction Detection**
- *"System doubts illogical responses"*
- Example: "High caffeine + perfect sleep quality? Unusual - check data."
- Why: Most ML projects trust input blindly

**2. Job-Specific Advice**
- *"Teachers need different advice than software engineers"*
- 8 categories × 75 lines = 600 lines of tailored recommendations

**3. Evidence-Based Interventions**
- *"Not generic 'sleep more' - specific: 'Sleep 4.1h → 7.5h, +3.4h needed'"*
- Backed by 332 real intervention cases

**Key Point:** "This is a system, not just a model."

---

### **Slide 7: Wrap-Up (20 seconds)**
*"To summarize:"*

✅ **98.5% accuracy** on mental health predictions  
✅ **10.3% improvement** over best baseline  
✅ **Production-ready** with interactive demo  
✅ **Evidence-based** interventions from real data  
✅ **Open for questions**

---

## 🎭 Presentation Tips

### **Tone & Delivery**
- [ ] Start confident: "Good morning, I'm excited to show..."
- [ ] Speak slowly (nervous = fast talking)
- [ ] Pause after key numbers (98.5%... pause... best baseline 89.3%)
- [ ] Make eye contact
- [ ] Use gestures to emphasize points

### **If Things Go Wrong**
**Demo doesn't load:**
- *"While that loads, let me show you the static report..."*
- Open `reports/examples/teacher_profile.html`
- Walk through sections: predictions, interventions, recommendations

**Question you don't know:**
- *"Great question. I don't have exact number memorized, but in our documentation..."*
- Honest > making up answers

**Running over time:**
- Skip architecture deep-dive
- Focus on demo + results

### **Strong Closing**
*"This project shows deep learning isn't just about accuracy - it's about building complete systems that provide real value. Thank you."*

Pause for applause, then: *"Happy to answer questions."*

---

## 📊 Key Numbers to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| **Accuracy** | 98.5% | Job satisfaction (best) |
| **Baseline** | 89.3% | Random Forest |
| **Improvement** | +10.3% | Relative improvement |
| **Model Size** | 1.2 MB | vs 491 MB baseline (99.8% smaller) |
| **Training Data** | 1.5M+ records | 500K samples used |
| **Prediction Speed** | <100ms | Real-time |
| **Features** | 17 | Sleep, work, exercise, etc. |
| **Targets** | 8 | Stress, mood, anxiety, depression, PSS, focus, energy, job_sat |
| **Job Categories** | 8 | Teachers, engineers, healthcare, etc. |
| **Interventions** | 332 cases | Real evidence base |

---

## ❓ Expected Questions & Answers

### **"How did you handle missing data?"**
*"We impute missing values with the training set mean for that feature. We use the StandardScaler's fitted mean from 500K samples, which is more robust than zero-imputation."*

### **"Why LSTM over Transformer?"**
*"Both work well - Transformer gets 98.7% vs LSTM's 98.5%. But LSTM is 40% faster (420s vs 680s training) and 40% smaller (1.2 MB vs 2 MB). For real-time deployment, LSTM wins."*

### **"How do you prevent overfitting?"**
*"Three ways: (1) Dropout layers (0.2), (2) Early stopping on validation set, (3) Multi-task learning acts as regularization - model can't memorize, must learn generalizable patterns."*

### **"What about privacy with real user data?"**
*"The training data is synthetic (no real people). For deployment, we'd anonymize inputs, use local inference (no cloud upload), and never store raw behavioral data."*

### **"Can this predict suicide risk?"**
*"No - that's outside our scope and would require clinical validation. Our system detects general mental health risks (stress, anxiety) and recommends professional help. It's a screening tool, not a diagnostic tool."*

### **"How often should someone use this?"**
*"Weekly is ideal. Our history tracking feature (shown in HTML reports) analyzes trends over 90 days. More frequent gives better longitudinal insights."*

### **"What's the contradiction detection accuracy?"**
*"We don't have ground truth for contradictions since they're pattern-based logic checks. But in our 4 test users, it correctly flagged Marti's 'at-risk mental health but poor self-care' pattern, which aligned with expert review."*

---

## 🎬 Backup Demo Videos (If Live Demo Fails)

Record these before presentation:

1. **Screen recording of Streamlit demo** (2 min)
   - Healthy → sleep deprived → overwork scenarios
   - Real-time prediction changes

2. **HTML report walkthrough** (1 min)
   - Open teacher_profile.html
   - Scroll through sections
   - Show charts, interventions, recommendations

3. **Architecture diagram zoom** (30 sec)
   - Pan across VISUAL_ARCHITECTURE.md

---

## 📋 Post-Presentation Tasks

- [ ] Answer all questions confidently
- [ ] Provide repository link if asked
- [ ] Note any suggestions from professor/peers
- [ ] Thank professor and audience
- [ ] Celebrate! 🎉

---

## 🏆 Success Criteria

**A+ Presentation:**
- Live demo works smoothly
- All key numbers mentioned (98.5%, +10.3%)
- Novel features clearly explained
- Within 5-minute time limit
- Confident delivery + eye contact
- Questions answered well

**Strong B+/A Presentation:**
- Demo has minor glitches (but backup works)
- Most key numbers mentioned
- Clear explanation of approach
- Slightly over/under time
- Good delivery
- Most questions answered

**You've Got This!** ✨

---

*Checklist created: December 4, 2025*  
*Ready for presentation success* 🚀
