# **Compliance Radar — Organizational Risk & Integrity Analysis**

*Corporate Compliance & Operational Risk Detection*

---

## **1. Project Overview**

This project analyzes the `org_compliance_data.db` dataset — a comprehensive organizational data source containing department-level operational metrics, reporting patterns, financial indicators, audit results, and anonymized engagement scores.
The objective is to design a **data-driven analytical framework** capable of identifying potential non-compliance signals, uncovering key explanatory factors, and generating actionable recommendations to strengthen corporate integrity.

The analysis integrates:

* Statistical reasoning
* Predictive modeling
* Ethical and interpretability insights
* Evidence-based governance recommendations

---

## **2. Problem Definition**

After examining the structure of the data, this project frames the main task as a **binary classification problem**:

> **Goal:** Predict whether a department is *high-risk* (`is_high_risk = 1`) or *not high-risk* (`is_high_risk = 0`) based on operational, managerial, engagement, and compliance metrics.

The `high_risk_departments` table is used to create the binary label.
Additional exploratory components (e.g., clustering for pattern discovery) may also be included to complement the supervised task.

---

## **3. Dataset Description**

The database contains four tables:

### **3.1 departments (709 rows, 37 columns)**

The core dataset with features describing:

* Department structure, size, type
* Manager/supervisor experience
* Training, reporting, and audit metrics
* Operational health and risk exposure indicators

### **3.2 high_risk_departments (201 rows, 37 columns)**

Subset of departments flagged as high-risk.
Used to create the binary target variable.

### **3.3 risk_summary_by_division (2 rows, 8 columns)**

Aggregated insights per division (Corporate HQ vs Regional Operations).

### **3.4 data_dictionary (39 rows, 4 columns)**

Provides definitions and data types for all fields.

---

## **4. Data Cleaning & Integrity Resolution**

A key data issue identified early was **duplicate department IDs** in `departments` with differing values.
Rather than removing any rows — which would distort the dataset and break alignment with the high-risk table — we **preserved all rows** and added a flag to track identifier conflicts.

**Final decision:**
All department records were kept; no row deletions performed.
A new feature (`id_conflict' flag)  indicates whether a 'dept_id' appears multiple times with inconsistent attributes.

This preserves data fidelity while enabling the model to learn from potential instability signals.

---

## **5. Feature Engineering**

Key feature additions include:

* **Binary target:** `is_high_risk` (from high_risk_departments)
* **Duplicate-ID indicator:** `id_conflict` flag
* Possible derived features (to be completed during EDA):

  * Reporting gap ratios
  * Audit score changes (Q1 → Q2)
  * Normalized violations
  * Interaction/engagement composite metrics

---

## **6. Exploratory Data Analysis (EDA)**

The EDA section includes:

* Distribution plots of numerical variables
* Missingness heatmaps and patterns
* Correlation matrices and risk-factor clustering
* Group-based comparisons (divisions, categories, types)
* Visual contrasts between high-risk vs low-risk departments

*Figures will be inserted here:*

```
📌 Placeholder for EDA images  
(e.g., histograms, boxplots, heatmaps, risk score distributions)
```

---

## **7. Problem Modeling Approach**

### **Model Type:**

**Binary classification**

### **Models Tested:**

1. **Logistic Regression** (interpretable baseline)
2. **Random Forest Classifier**
3. **XGBoost / Gradient Boosting Classifier**

Each model is evaluated first using default hyperparameters on a **validation split**, then fully optimized using **cross-validation**.

---

## **8. Preprocessing Pipeline**

Steps include:

* Missing value imputation
* Outlier detection and treatment
* One-hot encoding of categorical variables
* Scaling of numerical features where appropriate
* Train/validation/test splitting
* Feature importance & SHAP interpretability steps

---

## **9. Hyperparameter Tuning**

Cross-validation is performed (GridSearchCV or RandomizedSearchCV).
For each model, the following hyperparameters are tuned:

Examples:

* **Logistic Regression:** C, penalty, solver
* **Random Forest:** n_estimators, max_depth, min_samples_split, min_samples_leaf
* **XGBoost:** learning_rate, max_depth, subsample, colsample_bytree, n_estimators

*Full hyperparameter tables will be added after experimentation.*

---

## **10. Evaluation Metrics**

Primary metrics:

* **F1-score**
* **Precision, Recall** (especially important for identifying high-risk departments)
* **ROC-AUC**
* **Confusion matrix**

---

## **11. Results & Model Comparison**

```
📌 Placeholder for performance table  
📌 Placeholder for ROC curves  
📌 Placeholder for confusion matrix figures  
```

---

## **12. Insights & Interpretation**

This section will include:

* SHAP value analysis
* Feature importance rankings
* Interpretation of key variables driving risk
* Potential causal or governance explanations

---

## **13. Ethical Considerations**

Topics addressed:

* Risk of unfairly labeling departments without context
* Interpretability requirements for governance decisions
* Avoiding automation bias (model outputs must not replace compliance audit judgment)
* Transparency in how features influence predictions

---

## **14. Recommendations & Actionable Findings**

* Which operational metrics most strongly predict risk
* Early-warning indicators
* Governance improvements
* Training or reporting processes that reduce violations

---

## **15. Repository Structure**

Compliance-Radar-ML/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── main.ipynb
│
├── reports/
│
├── images/
│
└── README.md


---

## **16. Contributors**

Student 1: 
Student 2:
Student 3:
Student 4:

Bachelor’s in Artificial Intelligence & Management
Luiss Guido Carli University





##  TEAM B DRAFT ##



## 2. Methods
### 2.1 Problem Formulation
In this project, the predictive task focuses on determining whether an organizational department should be classified as high-risk based on its operational, financial, managerial, and audit-related attributes.
The target variable, is_high_risk, is a binary indicator taking values:
	•	1 — department identified as high-risk
	•	0 — department not identified as high-risk
Given the discrete and dichotomous nature of the outcome, the task is formally defined as a binary classification problem.
This formulation aligns with the aim of supporting compliance officers in identifying units that may require additional oversight or early intervention.
The classification framework also allows the integration of both:
	•	interpretable linear models (useful in governance contexts), and
	•	more expressive non-linear models (to improve predictive reliability).
This formalization sets the methodological foundation for the modelling strategy described in the following sections.
### 2.2 Model Definition 
To address the classification task, we adopt a modelling strategy that evaluates a diverse set of baseline and advanced algorithms.
The goal is to balance interpretability, predictive performance, and robustness, which are essential in compliance-driven environments.
We therefore assess the following models:
Baseline 0 — Majority Class Classifier
A naive classifier that always predicts the most frequent class in the dataset.
It provides the minimum benchmark against which all other models must be compared to demonstrate meaningful predictive value.
Model 1 — Logistic Regression
A linear and highly interpretable classifier commonly used in risk assessment contexts.
It represents an essential baseline, enabling transparent inspection of the effect and directionality of input variables.
Model 2 — Random Forest Classifier
A non-linear ensemble model composed of multiple decision trees.
Random Forests naturally model feature interactions, are robust to noise and outliers, and provide reliable feature importance measures.
Model 3 — XGBoost Classifier
A high-performance gradient boosting model designed for structured (tabular) data.
XGBoost generally achieves superior predictive accuracy and integrates seamlessly with SHAP values, enabling precise model interpretability.
By combining simple, interpretable baselines with more complex ensemble techniques, this modelling design ensures both methodological rigor and practical relevance for corporate compliance decision-making.
### 2.6 Preprocessing Pipeline
To ensure that the modeling process relies on clean, structured, and machine-learning–ready data, we designed a preprocessing pipeline that standardizes all variables before training. The goal is to guarantee comparability across features, avoid data leakage, and align the modelling procedure with the requirements of supervised learning.
The preprocessing workflow includes the following steps:
1. Train–Test Split
We begin by separating the dataset into training and test subsets.
The split is stratified on is_high_risk to preserve the proportion of high-risk and low-risk departments in both partitions.
The training set is then used for all model development, including cross-validation and hyperparameter tuning, while the test set is kept strictly for final evaluation.
2. Handling Missing Values
Missing values are imputed to avoid discarding data and to maintain consistency across models.
Numerical features are imputed using the median, which is robust to outliers, while categorical features are imputed using the most frequent category.
3. Outlier Treatment
Outliers identified during the EDA are addressed using a combination of:
	•	Winsorization for extremely skewed numerical values, or
	•	Robust scaling for variables with heavy tails
This prevents extreme observations from disproportionately influencing model behaviour.
4. Encoding Categorical Features
Categorical variables are transformed using One-Hot Encoding, which creates binary dummy variables without imposing ordinal relationships.
This representation ensures compatibility with both linear models (e.g., Logistic Regression) and tree-based ensembles (Random Forest, XGBoost).
5. Scaling Numerical Variables
For scaling, we employ the StandardScaler (mean = 0, standard deviation = 1) on all continuous features.
Scaling is essential for models that are sensitive to feature magnitudes, such as Logistic Regression and gradient boosting methods, and improves optimization stability.
6. Final Feature Matrix Construction
After preprocessing, all numerical and encoded categorical features are combined into a unified feature matrix X, while the target vector y = is_high_risk is extracted separately.
This final dataset is used for all subsequent modelling experiments.


## 3. Experimental Design
The experimental design outlines how the classification models are evaluated, compared, and validated. The goal is to ensure methodological rigor, prevent overfitting, and provide a transparent framework for assessing predictive performance.
### 3.1 Purpose of experiments
The primary objective of the experiments is to evaluate multiple classification algorithms and determine which model provides the most reliable predictions of departmental compliance risk.
Each experiment is designed to:
	•	assess baseline vs. advanced model performance,
	•	understand how different hyperparameters affect accuracy,
	•	measure the stability and generalizability of the models.
### 3.2 Baseline Comparison
All models are compared against:
	•	Majority Class Baseline, which predicts the most common class, and
	•	Default versions of Logistic Regression, Random Forest, and XGBoost.
This ensures that improvements are meaningfully attributable to model capability rather than dataset imbalance or random variation.
### 3.3 Evaluation Metrics
We adopt a set of metrics that reflect both global performance and the model’s ability to correctly identify high-risk departments.
	•	Accuracy – overall correctness
	•	Precision – reliability of high-risk predictions
	•	Recall – sensitivity; ability to detect all high-risk departments
	•	F1-score – balance between precision and recall
	•	ROC-AUC – ranking performance across thresholds
These metrics are particularly relevant in a compliance context, where both false positives (mislabeling a safe department) and false negatives (missing an at-risk department) have operational implications.
### 3.4 Validation Strategy
All hyperparameter tuning is conducted using k-fold cross-validation on the training data. This prevents overfitting and provides robust estimates of model performance across multiple data partitions.
### 3.5 Hyperparameter Tuning
GridSearchCV (or RandomizedSearchCV for efficiency) is used to tune:
	•	Logistic Regression (C, penalty)
	•	Random Forest (n_estimators, max_depth, min_samples_split)
	•	XGBoost (learning_rate, max_depth, subsample, n_estimators)
For each model, the best-performing configuration is identified according to the F1-score or ROC-AUC metric, depending on the imbalance of the target variable.



## 4 Results & Model Comparison
This section presents the empirical performance of the classification models described in Section 2.5. All evaluations are conducted on the held-out test set, after hyperparameter tuning performed exclusively on the training data to ensure methodological rigor and avoid data leakage. Our analysis compares the majority class baseline, Logistic Regression, Random Forest, and XGBoost using standard metrics for binary classification in compliance-risk detection.
### 4.1 Overall Classification Performance
Table 1 reports the main evaluation metrics—Accuracy, Precision, Recall, F1-score, and ROC-AUC—for all models.
Across all metrics, the machine-learning models significantly outperform the majority class baseline, confirming that the input features contain meaningful predictive information.
The majority class baseline attains relatively high accuracy due to class imbalance, but its recall and F1-score for the high-risk category are extremely low.
This demonstrates that accuracy alone is insufficient for evaluating performance in compliance-related tasks, where identifying high-risk units is essential.
Compared to the baseline:
	•	Logistic Regression improves performance but remains limited by linear decision boundaries.
	•	Random Forest achieves stronger recall and F1 performance, indicating improved detection of high-risk departments.
	•	XGBoost delivers the best overall results, achieving the highest F1-score and ROC-AUC and consistently improving both precision and recall.
These results indicate that more flexible non-linear models are better suited for modeling departmental compliance risk.
### 4.2 Confusion Matrix Analysis
Figure 1 displays the confusion matrices for the three main classifiers.
These visualizations provide a detailed view of the types of errors each model commits.
	•	Logistic Regression tends to produce fewer false positives but misses a greater number of high-risk departments (higher false negatives).
	•	Random Forest strikes a more balanced trade-off, correctly identifying a larger share of high-risk cases.
	•	XGBoost, while slightly more aggressive, achieves the highest number of correctly detected high-risk units, reducing both types of errors overall.
In a compliance context, missing a high-risk department (false negative) poses a greater operational threat than incorrectly flagging a low-risk department (false positive). Under this criterion, Random Forest and especially XGBoost are better aligned with real-world priorities.
### 4.3 ROC Curves and Threshold Behaviour
Figure 2 presents the ROC curves of the three classifiers. The ROC curve illustrates model performance across all possible classification thresholds. XGBoost consistently dominates the other classifiers across the entire curve, achieving the highest ROC-AUC score. This indicates that XGBoost has the strongest ability to correctly rank high-risk and low-risk departments regardless of threshold selection. Random Forest also provides strong separation power, while Logistic Regression performs acceptably but shows limitations tied to its linear structure. The superior ROC-AUC of the ensemble models reflects their ability to model complex, non-linear patterns in departmental operational data.
### 4.4 Feature Importance and SHAP Interpretability
To interpret the best-performing model (XGBoost), we compute feature importance scores and SHAP (SHapley Additive exPlanations) values. Figure 3 displays the SHAP summary plot, which quantifies the marginal impact of each feature on model predictions.
The SHAP analysis highlights several variables as key drivers of high-risk classification:
	•	Audit- and violation-related indicators, which strongly increase the predicted risk level
	•	Training insufficiency measures, where lower training hours or compliance engagement elevate risk probability
	•	Reporting gaps, such as under-submitted reports relative to expected totals
	•	Department instability signals, including duplicate or inconsistent identifiers (captured by the id_conflict flag)
Unlike traditional feature importance measures, SHAP values provide local interpretability, showing not only which features matter most but also how they push each individual prediction toward high or low risk.
This interpretability component is crucial in compliance applications, where explainable decisions are required to support governance processes, internal audits, and regulatory expectations.
### 4.5  Summary of Findings
Overall, the empirical evidence supports the following conclusions:
	1.	All machine-learning models outperform the baseline, confirming the presence of predictive structure in the dataset.
	2.	XGBoost is the strongest model across all metrics, offering the best trade-off between precision and recall.
	3.	Random Forest provides robust performance while maintaining strong interpretability.
	4.	Logistic Regression is useful primarily as an interpretable baseline, but its predictive limitations make it less suitable for operational deployment.
	5.	SHAP analysis confirms operational plausibility: audit issues, violations, and training/reporting deficiencies are among the top contributors to risk.
These findings support the use of machine-learning–based early-warning systems for compliance risk monitoring, complemented by interpretability tools that ensure transparency and accountability.



## 5 Conclusion and ethical considerations
### 5.1 General Conclusions
This project developed a machine-learning framework to identify departments with a high likelihood of compliance risk. Through a structured methodological pipeline—data preparation, feature engineering, model training, and interpretability analysis—we evaluated four classifiers with different levels of complexity and transparency.
The empirical results demonstrate that the dataset contains strong predictive signals:
	•	All machine-learning models outperform the majority baseline, confirming that operational, reporting, and audit variables are relevant predictors of risk.
	•	XGBoost achieves the highest overall performance, offering the best balance between precision and recall, and the strongest ROC-AUC.
	•	Random Forest provides stable and interpretable performance, making it suitable when slightly higher transparency is required.
	•	Logistic Regression, while less accurate, remains an important reference due to its interpretability and simplicity.
The SHAP analysis confirms that the models rely on meaningful operational indicators such as violations, audit findings, training gaps, reporting delays, and stability issues. These insights can directly support early-warning systems, targeted interventions, and data-driven governance policies. Overall, the project provides a solid foundation for integrating machine learning into compliance-risk monitoring while ensuring methodological rigor and interpretability.
### 5.2 Limitations and Future Work
Despite the strong results, several limitations remain:
	•	Data structure constraints: The dataset is static and does not include temporal evolution, which limits the ability to detect trends or risk progression over time.
	•	Potential hidden biases: Models may inherit biases present in the data (e.g., inconsistent reporting practices).
	•	Limited external validation: The evaluation is based on a single split of organizational data; testing on other companies or years would improve robustness.
	•	Feature completeness: Additional variables related to employee behavior, audits over time, cultural dynamics, or risk-mitigation actions could further enhance model quality.
Future work could include:
	•	incorporating time-series patterns into the analysis,
	•	developing model-agnostic drift-detection tools,
	•	integrating NLP signals from audit notes or reports,
	•	and deploying the model in a real-time monitoring dashboard.
### 5.3  Ethical and Governance Considerations
Using machine learning for compliance monitoring introduces important ethical implications.
First, false negatives—failing to detect a risky department—may expose the organization to operational, legal, or reputational harm. Therefore, models must be calibrated with particular attention to recall and must not replace human oversight.
Second, false positives—incorrectly flagging a low-risk department—can unfairly stigmatize units, disrupt internal dynamics, and lead to unnecessary investigations. Thus, model outputs should be used as decision-support tools, not automated judgments.
Third, transparency is essential.
Through SHAP explainability, decision-makers can understand which features influence each prediction, reducing the risk of opaque or unjustified decisions. This aligns with principles of fairness, accountability, and responsible AI.
Finally, governance frameworks should ensure:
	•	periodic model audits,
	•	monitoring of prediction biases,
	•	clear documentation of assumptions,
	•	and combined use of quantitative and qualitative evidence.
The objective is not to replace human judgment, but to augment compliance processes with transparent, interpretable, and data-informed insights.

