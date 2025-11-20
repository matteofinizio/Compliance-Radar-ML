# 📊 Compliance Radar – Organizational Risk & Integrity Analysis  
### Machine Learning Project – Academic Year 2025/2026  

---

## 👥 Team Members  
- **Team Captain:** _Name (Student ID)_  
- **Member 2:** _Name (Student ID)_  
- **Member 3:** _Name (Student ID)_  
- **Member 4:** _Name (Student ID)_  

---

# 1. 📝 Introduction  

**Compliance Radar** is an analytical framework designed to identify potential non-compliance risks within an organization using the dataset **org_compliance_data.db**.  
The project aims to:

- Detect departments with elevated operational, financial, or ethical risk indicators  
- Understand the drivers behind compliance inconsistencies  
- Build predictive + interpretative models  
- Provide evidence-based recommendations for reducing risk  
- Combine **statistical reasoning**, **ethical insights**, and **explainability**

Instead of focusing solely on prediction, this project emphasizes **interpretability**, **transparency**, and **practical insights** to support organizational accountability.

---

# 2. ⚙️ Methods  

This section describes:
- Data sources used  
- Data cleaning and preprocessing  
- Feature engineering choices  
- Model selection rationale  
- Environment setup  
- System design overview  

---

## 2.1 📁 Dataset Overview  
The SQLite database contains four core tables:

1. **departments** – operational, structural, and behavioral indicators  
2. **high_risk_departments** – subset marked as high-risk  
3. **risk_summary_by_division** – aggregate division-level summaries  
4. **data_dictionary** – definitions and metadata for all features  

We integrate these tables using `dept_id` as the primary key (validated through inspection and uniqueness checks).

---

## 2.2 🧹 Data Preprocessing  

Steps include:
- Missing value imputation (strategy depends on feature type)  
- Outlier detection & handling  
- Encoding categorical variables (One-Hot or Target Encoding)  
- Scaling numeric features  
- Merging tables using `dept_id`  
- Label construction for risk classification/regression  

_Figure Placeholder: Pipeline Flowchart_  
(Insert diagram from `/images/data_pipeline.png`)

---

## 2.3 🧠 Model Design  

Since the problem involves both **prediction** and **interpretation**, we use:

- **Binary classification models** (High-risk vs Low-risk)  
- **Regression models** (Predicting risk score)  
- **Explainability tools**: SHAP, feature importances, partial dependence  
- **Baseline models**: Logistic Regression, Decision Tree  
- **Advanced models**: Random Forest, XGBoost, Gradient Boosting  

We justify choices based on:
- Interpretability  
- Performance  
- Ability to handle tabular data  
- Robustness to multicollinearity and missingness  

---

## 2.4 🔧 Environment Reproducibility  

To recreate our environment, run:

```bash
conda env create -f environment.yml
