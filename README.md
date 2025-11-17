❤️ Heart Disease Prediction – Machine Learning Web Application

A complete end-to-end machine learning project that predicts the likelihood of heart disease using nine powerful classification algorithms. This system includes full model training, evaluation, saving, backend integration with Flask, and a modern, beautifully designed user interface built using pure HTML and CSS.

📌 Project Overview

This project implements a full workflow to analyze and predict heart disease using the Kaggle Heart Disease Dataset. It includes:

Data preprocessing and feature engineering

Training 9 major ML classification models

Storing trained models (.pkl) and evaluation metrics (.json)

A professional Flask-based web application

A modern, responsive UI for model selection and prediction

It is designed to be clean, organized, and easy to use—perfect for academic work, ML portfolios, and real-world demonstrations.

🧠 Dataset Information

Dataset:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Features (13 inputs):

Age

Sex (1 = male, 0 = female)

Chest pain type

Resting blood pressure

Cholesterol level

Fasting blood sugar

Resting ECG results

Maximum heart rate achieved

Exercise-induced angina

ST depression induced by exercise

Slope of peak exercise ST segment

Number of major vessels (0–3)

Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)

Target:

0 → No heart disease

1 → Heart disease present

🤖 Machine Learning Models Implemented

This project includes nine fully trained and evaluated classification models:

K-Nearest Neighbors (KNN)

Logistic Regression

Naive Bayes

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

Extreme Gradient Boosting (XGBoost)

Support Vector Machine (SVM)

Each model is saved as:

.pkl → Trained model file

.json → Accuracy score, confusion matrix, and classification report

📊 Model Evaluation Metrics

For every algorithm, the project stores:

Accuracy Score

Confusion Matrix (TP, TN, FP, FN)

Detailed Classification Report

Precision

Recall

F1-score

Support

These metrics help compare models directly through the UI without re-running training.

🧪 Training Pipeline

The training process includes:

Handling missing values

Data normalization/scaling (for certain models)

Model-specific hyperparameter tuning

Generating evaluation metrics

Exporting metrics to .json

Saving models using pickle.dump()

This ensures consistent deployment and reproducibility.

🖥 Flask Web Application

The backend is built using Flask, which handles:

Loading the appropriate model

Receiving input values

Performing real-time predictions

Fetching JSON metrics

Rendering results on the UI

🔧 Backend Technologies

Python

Flask

NumPy

Pandas

Scikit-Learn

XGBoost

🎨 Frontend Interface (HTML + CSS)

The application features a beautiful, modern dark-theme UI designed using:

Pure HTML

Pure CSS

No JavaScript required

✨ UI Features

Clean, professional layout

Elegant dark gradient background

Smooth card hover effects

Form with 13 input fields

Dropdown to select ML model

Real-time prediction display

Confusion matrix & classification report viewer

Fully responsive design for all screen sizes

The entire interface is crafted to look premium, user-friendly, and visually appealing.

📁 Folder Structure
project/
│── app.py
│── requirements.txt
│── models/
│     ├── knn.pkl
│     ├── svm.pkl
│     ├── random_forest.pkl
│     └── ...
│
│── metrics/
│     ├── knn.json
│     ├── svm.json
│     ├── xgboost.json
│     └── ...
│
│── templates/
│     └── index.html
│
│── static/
      └── style.css
🚀 How to Run the Project
1️⃣ Install all dependencies
pip install -r requirements.txt

2️⃣ Run Flask app
python app.py

3️⃣ Open in browser
http://127.0.0.1:5000/

📦 Requirements
Flask
numpy
pandas
scikit-learn
xgboost

🌟 Highlights

✔ 9 classification models

✔ Clean dataset preprocessing

✔ Machine learning metrics stored in JSON

✔ Beautiful HTML/CSS front-end

✔ Fully functional Flask backend

✔ Real-time prediction workflow

✔ Professional UI and project structure

✔ Great for portfolios & ML case studies

✨ Author

Manish D
Created with ❤️ passion for Machine Learning & Web Development
