
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/iris_model.joblib")

class_names = ["setosa", "versicolor", "virginica"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris Logistic Regression API is running!",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "features" not in data:
        return jsonify({
            "error": "Send JSON with a features list containing 4 numbers."
        }), 400

    try:
        features = np.asarray(data["features"], dtype=float)

        if features.shape != (4,) or not np.isfinite(features).all():
            raise ValueError

        prediction = int(model.predict([features])[0])
        probabilities = model.predict_proba([features])[0]

        return jsonify({
            "prediction": class_names[prediction],
            "class_id": prediction,
            "probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(class_names, probabilities)
            }
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "features must contain exactly 4 valid numbers."
        }), 400


if __name__ == "__main__":
    app.run(debug=False)
