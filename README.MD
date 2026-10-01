# 🚀 MLPROJECT — End-to-End Student Performance Prediction

🔗 **GitHub Repository:**
https://github.com/MSIVAPAPARAO13/MLPROJECT

An end-to-end **Machine Learning regression project** that predicts a student's **Mathematics Score** using demographic and academic attributes.

The project demonstrates a complete ML engineering workflow from **data ingestion and preprocessing to model training, evaluation, serialization, Flask deployment, and real-time prediction**.

---

## 📌 Project Overview

The system predicts a student's mathematics score using:

* Gender
* Race / ethnicity
* Parental level of education
* Lunch type
* Test preparation course
* Reading score
* Writing score

### ML Pipeline

```text
Raw Student Dataset
        ↓
Data Ingestion
        ↓
Train / Test Split
        ↓
Data Transformation
        ↓
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Multiple Regression Models
        ↓
GridSearchCV Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Serialization
        ↓
Flask Prediction Pipeline
        ↓
Real-Time Math Score Prediction
```

---

# ✨ Features

* 📥 Automated data ingestion
* 📊 1,000-record student dataset
* ✂️ 80/20 train-test split
* 🧹 Missing-value handling
* 🔤 One-hot encoding for categorical features
* 📏 StandardScaler for numerical features
* 🧩 ColumnTransformer preprocessing
* 🤖 Multi-model regression comparison
* 🔍 3-fold GridSearchCV hyperparameter tuning
* 📈 MAE, RMSE and R² evaluation
* 🏆 Automated best-model selection
* 💾 Model and preprocessor serialization
* 🌐 Flask prediction application
* 🎨 Bootstrap-based web interface
* 📝 Structured logging
* ⚠️ Custom exception handling
* ☁️ AWS Elastic Beanstalk-compatible configuration

---

# 📊 Dataset

The project uses a student performance dataset containing:

| Feature                       | Description               |
| ----------------------------- | ------------------------- |
| `gender`                      | Student gender            |
| `race_ethnicity`              | Race / ethnicity group    |
| `parental_level_of_education` | Parent education level    |
| `lunch`                       | Lunch type                |
| `test_preparation_course`     | Test preparation status   |
| `reading_score`               | Reading examination score |
| `writing_score`               | Writing examination score |
| `math_score`                  | Target variable           |

### Dataset Size

```text
Total Records: 1,000
Training: 800
Testing: 200
```

---

# 🧠 Machine Learning Models

The model-training notebook compares the following regression algorithms:

1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. K-Neighbors Regressor
5. Decision Tree Regressor
6. Random Forest Regressor
7. XGBoost Regressor
8. CatBoost Regressor
9. AdaBoost Regressor

---

# ⚙️ Data Preprocessing

## Numerical Features

Numerical features are processed using:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

Current numerical features:

```text
reading_score
writing_score
```

## Categorical Features

Categorical features use:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

Categorical features:

```text
gender
race_ethnicity
parental_level_of_education
lunch
test_preparation_course
```

---

# 🧩 ColumnTransformer

The numerical and categorical pipelines are combined through:

```python
ColumnTransformer
```

This ensures that preprocessing is consistent across training and inference.

The fitted preprocessing object is saved as:

```text
artifacts/preprocessor.pkl
```

---

# 🔬 Model Training

The training pipeline evaluates multiple models and performs hyperparameter tuning using:

```text
GridSearchCV
cv = 3
```

The project evaluates models using:

```text
MAE
RMSE
R² Score
```

The best-performing model is selected using test-set R² and serialized as:

```text
artifacts/model.pkl
```

---

# 📈 Model Results

### Test Set Performance

| Model             |         R² |       RMSE |        MAE |
| ----------------- | ---------: | ---------: | ---------: |
| Ridge             | **0.8806** | **5.3904** | **4.2111** |
| Linear Regression |     0.8797 |     5.4096 |     4.2259 |
| CatBoost          |     0.8516 |     6.0086 |     4.6125 |
| Random Forest     |     0.8496 |     6.0500 |     4.7320 |
| AdaBoost          |     0.8464 |     6.1136 |     4.7583 |
| XGBoost           |     0.8278 |     6.4733 |     5.0577 |
| Lasso             |     0.8253 |     6.5197 |     5.1579 |
| KNN               |     0.7839 |     7.2516 |     5.6160 |
| Decision Tree     |     0.7191 |     8.2671 |     6.6550 |

> Note: Ridge achieves the highest score in the notebook comparison. The current automated `ModelTrainer` candidate dictionary does not include Ridge, so its production candidate set currently selects from seven models, where Linear Regression has the highest documented test R².

---

# 📁 Project Structure

```text
MLPROJECT/
│
├── .ebextensions/
│   └── python.config
│
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── catboost_info/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   └── time_left.tsv
│
├── notebook/
│   ├── data/
│   │   └── stud.csv
│   └── 2. MODEL TRAINING.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── .gitignore
├── README.MD
├── application.py
├── requirements.txt
└── setup.py
```

---

# 🧱 Architecture

## 1. Data Ingestion

`src/components/data_ingestion.py`

Responsibilities:

* Load the student dataset
* Store raw data
* Create train/test datasets
* Save the resulting CSV files inside `artifacts/`
* Trigger transformation and model-training stages

---

## 2. Data Transformation

`src/components/data_transformation.py`

Responsibilities:

* Identify numerical and categorical features
* Handle missing values
* Apply StandardScaler
* Apply OneHotEncoder
* Build ColumnTransformer
* Fit transformation on training data
* Transform test data
* Save the preprocessing object

---

## 3. Model Training

`src/components/model_trainer.py`

Responsibilities:

* Load transformed training/test arrays
* Define multiple regression models
* Configure hyperparameter grids
* Run GridSearchCV
* Compare model R² scores
* Select the best candidate
* Save the trained model
* Calculate final R²

---

## 4. Prediction Pipeline

`src/pipeline/predict_pipeline.py`

Responsibilities:

* Load the serialized model
* Load the preprocessing object
* Convert form data into a Pandas DataFrame
* Apply the saved preprocessing pipeline
* Generate model predictions

---

## 5. Flask Application

`application.py`

Routes:

```text
GET /
```

Displays the landing page.

```text
GET /predictdata
```

Displays the prediction form.

```text
POST /predictdata
```

Receives student inputs and returns the predicted mathematics score.

---

# 🌐 Web Application

The frontend is built using:

```text
HTML
Bootstrap 5
Jinja2
```

The prediction form accepts:

```text
Gender
Race / Ethnicity
Parental Education
Lunch Type
Test Preparation Course
Reading Score
Writing Score
```

The predicted mathematics score is displayed dynamically.

The UI also provides:

```text
Predicted Maths Score
PASS / FAIL Status
```

The current interface considers:

```text
Score >= 35 → PASS
Score < 35  → FAIL
```

---

# 💾 Model Artifacts

The project stores trained artifacts in:

```text
artifacts/
```

### Model

```text
artifacts/model.pkl
```

Contains the selected trained regression model.

### Preprocessor

```text
artifacts/preprocessor.pkl
```

Contains the fitted preprocessing pipeline.

### Dataset Splits

```text
artifacts/data.csv
artifacts/train.csv
artifacts/test.csv
```

---

# 📝 Logging

The project contains a centralized logging utility.

Logs are generated with:

```text
Timestamp
Line Number
Module Name
Log Level
Message
```

This helps trace pipeline execution and diagnose runtime failures.

---

# ⚠️ Custom Exception Handling

`src/exception.py` implements a custom exception class that captures:

* Python file name
* Line number
* Error message

This improves debugging across ingestion, transformation, training and inference stages.

---

# 🛠️ Tech Stack

## Programming

* Python

## Data Science

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost
* CatBoost

## ML Algorithms

* Linear Regression
* Lasso
* Ridge
* KNN
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* CatBoost
* AdaBoost

## Web

* Flask
* HTML5
* Bootstrap 5
* Jinja2

## Deployment Configuration

* AWS Elastic Beanstalk configuration
* WSGI

## Development

* Jupyter Notebook
* Git
* GitHub

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/MSIVAPAPARAO13/MLPROJECT.git
```

```bash
cd MLPROJECT
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Data Ingestion & Training

From the project root:

```bash
python src/components/data_ingestion.py
```

This starts the pipeline:

```text
Dataset
 ↓
Data Ingestion
 ↓
Train/Test Split
 ↓
Data Transformation
 ↓
Model Training
 ↓
Best Model
 ↓
artifacts/model.pkl
```

---

# ▶️ Run Flask Application

```bash
python application.py
```

Open:

```text
http://127.0.0.1:5000
```

Prediction page:

```text
http://127.0.0.1:5000/predictdata
```

---

# ☁️ AWS Elastic Beanstalk Configuration

The project contains:

```text
.ebextensions/python.config
```

with the WSGI configuration:

```text
WSGIPath: application:app
```

This provides an AWS Elastic Beanstalk-compatible Flask application entry point.

---

# 🔐 API Keys

This project does **not require third-party API keys** for its core functionality.

```text
External API Keys: None
```

The model runs locally using:

```text
model.pkl
preprocessor.pkl
```

---

# 📊 Engineering Concepts Demonstrated

* End-to-End Machine Learning
* Regression
* Feature Engineering
* Data Preprocessing
* ColumnTransformer
* One-Hot Encoding
* Standardization
* Hyperparameter Tuning
* GridSearchCV
* Cross-Validation
* Model Comparison
* Model Selection
* Model Serialization
* Prediction Pipelines
* Flask Integration
* Logging
* Custom Exception Handling
* Package Structuring
* WSGI Deployment Configuration

---

# 🎯 Use Case

The platform can be used as an educational and analytical tool to estimate student mathematics performance from academic and demographic features.

It demonstrates how a machine learning model can be taken from:

```text
Raw Data
   ↓
Model Development
   ↓
Evaluation
   ↓
Serialization
   ↓
Web Application
   ↓
Real-Time Prediction
```

---

# 🔮 Future Improvements

* Add Docker support
* Add automated CI/CD
* Add unit and integration tests
* Add experiment tracking
* Add model explainability using SHAP
* Add prediction history
* Add database persistence
* Add REST API endpoints
* Add monitoring and model-drift detection
* Add improved validation and error messages
* Add cloud deployment automation
* Add automated retraining pipeline

---

# 👨‍💻 Author

**Siva Paparao Medisetti**

GitHub:
https://github.com/MSIVAPAPARAO13

Project:
https://github.com/MSIVAPAPARAO13/MLPROJECT

---

# ⭐ Project Highlight

> **Built an end-to-end student performance prediction system using a modular ML pipeline, preprocessing automation, multi-model regression benchmarking, GridSearchCV tuning, model serialization, and Flask-based real-time inference, achieving approximately 0.88 R² on the held-out test set.**

---

## 📌 Repository

🔗 https://github.com/MSIVAPAPARAO13/MLPROJECT
