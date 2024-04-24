from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

app = Flask(__name__)
CORS(app)

data = pd.read_csv('Dataset_spine.csv')

model = joblib.load("random_forest_model.pkl")

label_encoder = LabelEncoder()
data['Class_att'] = label_encoder.fit_transform(data['Class_att'])

@app.route('/normality', methods=['POST'])
def predict():
    data = request.get_json()

    new_data = pd.DataFrame(data)

    predictions = model.predict(new_data)
    predicted_classes = label_encoder.inverse_transform(predictions)

    return jsonify({"predictions": predicted_classes.tolist()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)