# Import necessary libraries
from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans
import joblib  # Assuming you're using joblib to save/load your model

app = Flask(__name__)

# Load the pre-trained KMeans model
model = joblib.load('kmeans_clustering_model.pkl')  

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    data = request.get_json()
    age = data['Age']
    systolic_pressure = data['systolic_pressure']
    diastolic_pressure = data['diastolic_pressure']
    
    # Create a feature array from the input data
    features = np.array([[age, systolic_pressure, diastolic_pressure]])
    
    # Predict the cluster
    cluster = model.predict(features)
    
    return jsonify({'Cluster': int(cluster[0])})

if __name__ == '__main__':
    app.run(debug=True)
