# Stroke Prediction with Machine Learning

Can we predict from basic health data whether a patient will have a stroke? This project builds a complete scikit-learn pipeline, compares logistic regression with a random forest and chooses the decision threshold with the medical context in mind.

| | |
|---|---|
| Course | Machine Learning 1, B.Sc. Artificial Intelligence, Landshut University of Applied Sciences |
| Semester | 3rd semester, winter 2025/26 |
| Team | Adam, Farouk |
| Grade | 1.3 |
| Language | Notebook and slides are written in German |

## Results

| Model | Test ROC-AUC |
|---|---:|
| **Logistic regression** (class weights balanced) | **0.844** |
| Random forest | 0.786 |

- **Final model:** logistic regression with a threshold of 0.4. It finds 42 of the 50 stroke cases in the test set (recall 0.84)
- **Cross-validation:** 5-fold stratified, mean ROC-AUC 0.837 ± 0.022, so the result is stable and not a lucky split
- **Overfitting check:** train ROC-AUC 0.847 vs. test 0.844, no sign of overfitting
- **Honest limitation:** with only about 5 % stroke cases, high recall comes at the cost of many false alarms (precision 0.12, F1 0.21)

## What the project covers

- **Exploratory data analysis** of 5,110 patients with 10 features such as age, glucose level, BMI, hypertension and smoking status
- **Preprocessing in a pipeline:** median imputation and scaling for numerical features, one-hot encoding for categorical ones, a stratified train/test split, so there is no data leakage
- **Model choice with reasons:** KNN, naive Bayes, logistic regression and random forest are compared on paper before training
- **Threshold analysis:** from 0.5 to 0.4 the number of missed strokes drops from 10 to 8 for 61 more false alarms; in medicine, missing a stroke is the more expensive error
- **Comparison model:** the random forest only reaches a useful recall with extreme thresholds and stays clearly behind

## Files

| File | Content |
|---|---|
| `stroke_prediction.ipynb` | the full notebook with all outputs |
| `ML1_Praesentation.pdf` | presentation slides |
| `requirements.txt` | Python packages |

## Run it yourself

1. Download `healthcare-dataset-stroke-data.csv` from the Kaggle [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). The data is not included in this repository.
2. Save it next to the notebook as `stroke.csv`.
3. Install the packages and start Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook stroke_prediction.ipynb
```
