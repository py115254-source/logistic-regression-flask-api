
# Iris Logistic Regression REST API

## Project Overview
This project deploys a Machine Learning Logistic Regression model as a REST API using Flask. The model predicts the species of an Iris flower using four input features.

## Technologies Used
- Python
- Scikit-learn
- Flask
- NumPy
- Joblib
- REST API

## Model Performance
- Model: Logistic Regression
- Dataset: Iris Dataset
- Test Accuracy: 96.67%
- Train-Test Split: 80:20
- Random State: 42

## Project Structure
```text
logistic-regression-flask-api/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── models/
    └── iris_model.joblib
```

## Installation
Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the Model
```bash
python train_model.py
```

## Run the API
```bash
python app.py
```

The API will run at:
http://127.0.0.1:5000

## API Endpoints

### Home
- Method: GET
- URL: `/`

### Predict Iris Species
- Method: POST
- URL: `/predict`
- Content-Type: `application/json`

### Example Request
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### Example Response
```json
{
  "prediction": "setosa",
  "class_id": 0,
  "probabilities": {
    "setosa": 0.9784,
    "versicolor": 0.0216,
    "virginica": 0.0
  }
}
```

*Note: The actual probabilities may vary slightly.*

## Features
- Trains a Logistic Regression model.
- Saves the trained model using Joblib.
- Provides predictions through a Flask REST API.
- Returns predicted species and class probabilities.
- Validates API input.

## Author
Machine Learning Project
