# Dataset Citation

## Primary Dataset

**Work-Life Balance Synthetic Daily Wellness Dataset**

**Author:** Wafaa El-Husseini  
**Published:** 2024  
**Platform:** Kaggle  
**URL:** https://www.kaggle.com/datasets/wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset  
**License:** Community Data License Agreement - Sharing, Version 1.0

### Citation Format (APA):
```
El-Husseini, W. (2024). Work-Life Balance Synthetic Daily Wellness Dataset. 
Kaggle. https://www.kaggle.com/datasets/wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset
```

### Citation Format (BibTeX):
```bibtex
@misc{elhusseini2024worklife,
  author = {El-Husseini, Wafaa},
  title = {Work-Life Balance Synthetic Daily Wellness Dataset},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset}
}
```

## Dataset Description

The dataset contains synthetic (privacy-preserving) daily wellness data for approximately 2,100 workers tracked over time, totaling 1.5M+ records. It includes:

- **Behavioral Features (17):** sleep hours, work hours, exercise, screen time, caffeine, social interactions, etc.
- **Mental Health Outcomes (8):** stress, mood, energy, focus, PSS, anxiety, depression, job satisfaction
- **Demographics:** age, profession, work mode, chronotype
- **Interventions (332 cases):** therapy, diet coaching, exercise plans, meditation, sick leave, vacation, workload caps

### Key Characteristics:
- **Synthetic Data:** Generated to preserve privacy while maintaining realistic correlations
- **Longitudinal:** Multiple observations per user over weeks/months
- **Multi-Dimensional:** Combines behavioral, psychological, and demographic data
- **Intervention Tracking:** Includes real intervention outcomes for evidence-based recommendations

## Academic Integrity Statement

This project is submitted as part of the **Foundations of Data Science** course requirements. All data sources are properly cited, and the analysis represents original work by the FDS Project team.

### Our Contributions:
- Multi-task LSTM architecture for 8 simultaneous mental health predictions
- Flexible Google Form CSV parser with fuzzy column matching
- Job-specific recommendation system (8 categories)
- Contradiction detection engine (20+ patterns)
- Evidence-based behavioral intervention system
- Interactive Streamlit demo application
- Comprehensive HTML report generation with Chart.js
- Longitudinal mental health trend analysis

### External Resources Used:
- **Dataset:** Wafaa El-Husseini (Kaggle) - properly cited above
- **Deep Learning Framework:** PyTorch (open-source, BSD license)
- **Visualization:** Chart.js (MIT license, CDN)
- **Web Framework:** Streamlit (Apache 2.0 license)
- **Data Processing:** pandas, NumPy, scikit-learn (BSD licenses)

All code is original unless otherwise noted in comments. No plagiarism of academic papers or other student work.

---

*Last Updated: December 4, 2025*  
*FDS Project Team*  
*Repository: github.com/danpinocontrollino/FDS-Project*
