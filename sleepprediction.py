import pickle
import numpy as np
import pandas as pd

# Load the trained models
with open('linear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('decision_tree_model.pkl', 'rb') as f:
    tree_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    forest_model = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Convert hours to seconds
def hours_to_seconds(hours):
    return hours * 3600

# Take inputs from user
TimeInBed = hours_to_seconds(float(input("Enter Time in Bed (hours): ")))
TimeAsleep = hours_to_seconds(float(input("Enter Time Asleep (hours): ")))
TimeBeforeSleep = float(input("Enter Time Before Sleep (seconds): "))
Snore = int(input("Enter Snore (0 or 1): "))
Alarm = int(input("Enter Alarm (0 or 1): "))
Steps = int(input("Enter Steps: "))

# Create input array for prediction with feature names
input_data = np.array([[Steps, Alarm, TimeInBed, TimeAsleep, TimeBeforeSleep, Snore]])
feature_names = ['Steps', 'Alarm', 'TimeInBed', 'TimeAsleep', 'TimeBeforeSleep', 'Snore']
input_data = pd.DataFrame(data=input_data, columns=feature_names)

# Predict sleep quality using all models
linear_pred = linear_model.predict(input_data)[0]
tree_pred = tree_model.predict(input_data)[0]
forest_pred = forest_model.predict(input_data)[0]
best_pred=best_model.predict(input_data)[0]

# Print the predicted sleep quality from each model
print("Predicted Sleep Quality:")
print("Linear Regression:", linear_pred)
print("Decision Tree:", tree_pred)
print("Random Forest:", forest_pred)
print("Best Model:",best_pred)

