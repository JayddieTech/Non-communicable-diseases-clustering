from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = kmeans  # Assuming this is the trained model
scaler = StandardScaler()  # Assume scaler is already trained

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    # Parse input JSON request
    data = request.json
    age = data['Age']
    systolic_pressure = data['systolic_pressure']
    diastolic_pressure = data['diastolic_pressure']

    # Prepare data for prediction
    new_data = np.array([[age, systolic_pressure, diastolic_pressure]])
    new_data_scaled = scaler.transform(new_data)
    
    # Predict cluster
    cluster = model.predict(new_data_scaled)

    return jsonify({'Cluster': int(cluster[0])})

if __name__ == '__main__':
    app.run(debug=True)
