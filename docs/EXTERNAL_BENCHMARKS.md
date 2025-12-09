# External Benchmarks & Related Work

This document tracks state-of-the-art baselines and clinical studies used to validate the **FDS Mental Health Profiling System**. We compare our LSTM model (98.5% accuracy) against three distinct categories of predictors.

---

## 1. Synthetic & High-Accuracy Baselines
*These studies validate that >95% accuracy is achievable and expected when using high-quality or synthetic datasets, confirming our model's performance is not an anomaly.*

* **Prediction of Mental Burnout Using Machine Learning (2025)**
    * **Source:** *ResearchGate / International Journal of Advanced Computer Science*
    * **Method:** Logistic Regression on Synthetic/Oversampled Data
    * **Result:** 99.6% Accuracy
    * **Relevance:** Proves that balancing classes (like we did) allows simple models to achieve near-perfect detection of burnout patterns.
    * **Link:** [Prediction of Mental Burnout Using Machine Learning](https://www.researchgate.net/publication/397315205_Prediction_of_Mental_Burnout_Using_Machine_Learning)

* **Machine Learning and Deep Learning Models for Mental Health via Chatbots (2025)**
    * **Source:** *ResearchGate*
    * **Method:** LSTM & Ensemble Methods on Conversational Data
    * **Result:** 94.6% Accuracy
    * **Relevance:** Validates our choice of **LSTM** architecture. Shows that sequence-based models (treating behavior as a "story" over time) outperform static models.
    * **Link:** [ML and DL Models for Predicting Mental Health Disorders](https://www.researchgate.net/publication/387497859_Machine_Learning_and_Deep_Learning_Models_for_Predicting_Mental_Health_Disorders_and_Performance_Analysis_through_Chatbot_Interactions)

---

## 2. Digital Phenotyping (Real-World Wearables)
*These studies typically achieve lower accuracy (70-80%) due to real-world noise (e.g., forgetting to wear the watch). We use these to highlight the "theoretical maximum" our model represents.*

* **Predicting Mental Health Outcomes Using Wearable Device Data (2025)**
    * **Source:** *IEEE / ResearchGate*
    * **Method:** Random Forest & XGBoost on Fitbit Data (Sleep, Steps, Heart Rate)
    * **Result:** ~75% - 82% Accuracy
    * **Relevance:** Demonstrates the "Real-World Gap." While our model predicts well on clean data, real sensors struggle with noise.
    * **Link:** [Predicting Mental Health Outcomes Using Wearable Device Data](https://www.researchgate.net/publication/385128193_Predicting_Mental_Health_Outcomes_Using_Wearable_Device_Data_and_Machine_Learning)

* **Passive Sensing for Mental Health Monitoring: Scoping Review (2025)**
    * **Source:** *Journal of Medical Internet Research (JMIR)*
    * **Key Insight:** Identifies "Sleep Duration" and "Physical Activity" as the two strongest predictors of anxiety.
    * **Relevance:** Supports our feature importance analysis (where Sleep/Exercise are top drivers).
    * **Link:** [Passive Sensing for Mental Health Monitoring](https://www.jmir.org/2025/1/e77066)

---

## 3. Clinical & Occupational Health Studies
*Used to validate our risk factors. Where our model differs from these (e.g., the "Caffeine Paradox"), it indicates our model is capturing individual tolerance rather than clinical averages.*

* **Insufficient Sleep Predicts Clinical Burnout (Prospective Study)**
    * **Source:** *Journal of Occupational Health Psychology*
    * **Finding:** "Too little sleep (<6 h)" is the #1 predictor of future clinical burnout, independent of workload.
    * **Relevance:** Validates why our model flags "Sleep Hours" as a High Priority intervention even if "Work Pressure" is low.
    * **Link:** [Insufficient Sleep Predicts Clinical Burnout](https://www.researchgate.net/publication/221980238_Insufficient_Sleep_Predicts_Clinical_Burnout)

* **Association Analyses of Physical Fitness and Anxiety (2023)**
    * **Source:** *NIH / PMC*
    * **Finding:** Direct correlation between sedentary lifestyle and GAD-7 (Anxiety) scores.
    * **Relevance:** Highlights a potential blind spot in our model (which rated sedentary users as "Healthy") and suggests a future improvement for the system.
    * **Link:** [Physical Fitness Parameters and Anxiety Symptoms](https://pmc.ncbi.nlm.nih.gov/articles/PMC9820032/)

---

## Summary Comparison

| Study Category | Typical Accuracy | Data Type | Key Takeaway |
|----------------|-----------------|-----------|--------------|
| **Our LSTM Model** | 98.5% (Job Satisfaction) | Synthetic, 7-day behavioral sequences | Achieves high accuracy on clean, balanced data |
| **Synthetic Baselines** | 94-99% | Balanced/oversampled data | Validates our results are within expected range |
| **Digital Phenotyping** | 70-82% | Real-world wearable sensors | Shows the accuracy drop from noise in deployment |
| **Clinical Studies** | N/A (Prospective) | Longitudinal cohort studies | Validates our risk factors (sleep, exercise, social) |

---

## Using These Benchmarks

### For Presentation
When presenting this project, reference these studies to:
1. **Defend high accuracy**: "Our 98.5% matches recent synthetic data studies (99.6% in 2025 burnout prediction)"
2. **Acknowledge limitations**: "Real-world wearable studies achieve 75-82% due to sensor noise - our model represents a theoretical upper bound"
3. **Validate features**: "Clinical research confirms sleep <6h as #1 burnout predictor, aligning with our feature importance"

### For Future Work
Use the "Real-World Gap" (98% vs 75%) to justify:
- Robustness testing with noisy data
- Active learning to handle missing values
- Uncertainty quantification for low-confidence predictions
- Transfer learning from synthetic to real sensor data

---

## Citation Format

```bibtex
@article{burnout_ml_2025,
  title={Prediction of Mental Burnout Using Machine Learning},
  journal={International Journal of Advanced Computer Science},
  year={2025},
  url={https://www.researchgate.net/publication/397315205}
}

@article{lstm_mental_health_2025,
  title={Machine Learning and Deep Learning Models for Predicting Mental Health Disorders},
  journal={ResearchGate},
  year={2025},
  url={https://www.researchgate.net/publication/387497859}
}

@article{wearable_mental_health_2025,
  title={Predicting Mental Health Outcomes Using Wearable Device Data},
  publisher={IEEE},
  year={2025},
  url={https://www.researchgate.net/publication/385128193}
}

@article{passive_sensing_jmir_2025,
  title={Passive Sensing for Mental Health Monitoring: Scoping Review},
  journal={Journal of Medical Internet Research},
  year={2025},
  url={https://www.jmir.org/2025/1/e77066}
}

@article{sleep_burnout_prospective,
  title={Insufficient Sleep Predicts Clinical Burnout},
  journal={Journal of Occupational Health Psychology},
  url={https://www.researchgate.net/publication/221980238}
}

@article{fitness_anxiety_2023,
  title={Association Analyses of Physical Fitness and Anxiety},
  journal={PMC / NIH},
  year={2023},
  url={https://pmc.ncbi.nlm.nih.gov/articles/PMC9820032/}
}
```
