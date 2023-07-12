import pandas as pd
import joblib
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

model = joblib.load('best_model.pkl')
# Load the dataset
data = pd.read_csv('sleepdata.csv')


# Function to update the model with feedback data
# Update the sleepdata.csv file with new data

# Update the sleepdata.csv file with new data
def update_model(user_inputs):
    # Load the existing data
    data = pd.read_csv('sleepdata.csv')

    # Create a DataFrame from user_inputs
    feedback_data = pd.DataFrame(user_inputs, index=[0])

    # Append feedback_data to the existing data
    updated_data = pd.concat([data, feedback_data], ignore_index=True)

    # Save the updated data to sleepdata.csv
    updated_data.to_csv('sleepdata.csv', index=False)

    # Split the updated data into predictors and target variable
    predictors = ['Steps', 'Alarm', 'TimeInBed', 'TimeAsleep', 'TimeBeforeSleep', 'SnoreTime']
    X = updated_data[predictors]
    y = updated_data['SleepQuality']

    # Retrain the model with the updated data
    model.fit(X, y)

    # Save the updated model
    dump(model, 'best_model.pkl')


# Function to predict sleep quality based on user inputs
def predict_sleep_quality(user_inputs):
    input_data = pd.DataFrame(data=[user_inputs],
                              columns=['Steps', 'Alarm', 'TimeInBed', 'TimeAsleep', 'TimeBeforeSleep', 'SnoreTime'])
    predicted_quality = model.predict(input_data)[0]
    return predicted_quality


# Command line interface
while True:
    print("\n===== Sleep Quality Predictor =====")
    print("Enter your sleep-related information (press 'q' to quit):")

    # Get user inputs
    steps = input("Steps: ")
    if steps.lower() == 'q':
        break
    steps = int(steps)

    alarm = input("Alarm (0 or 1): ")
    if alarm.lower() == 'q':
        break
    alarm = int(alarm)

    time_in_bed = input("Time in Bed (seconds): ")
    if time_in_bed.lower() == 'q':
        break
    time_in_bed = int(time_in_bed)

    time_asleep = input("Time Asleep (seconds): ")
    if time_asleep.lower() == 'q':
        break
    time_asleep = int(time_asleep)

    time_before_sleep = input("Time Before Sleep (seconds): ")
    if time_before_sleep.lower() == 'q':
        break
    time_before_sleep = int(time_before_sleep)

    snore = input("Snore (0 or 1): ")
    if snore.lower() == 'q':
        break
    snore = int(snore)

    # Predict sleep quality
    user_inputs = {'Steps': steps, 'Alarm': alarm, 'TimeInBed': time_in_bed, 'TimeAsleep': time_asleep,
                   'TimeBeforeSleep': time_before_sleep, 'SnoreTime': snore}
    predicted_quality = predict_sleep_quality(user_inputs)

    # Provide feedback based on prediction
    if predicted_quality >= 70:
        print("Your sleep quality is good. Keep it up!")
    else:
        print("Your sleep quality needs improvement. Consider making changes to your sleep routine.")

    # Ask for user feedback
    feedback = input("Did the prediction match your actual sleep quality? (y/n): ")

    # Update the model based on feedback
    if feedback.lower() == 'y':
        user_inputs['SleepQuality'] = predicted_quality
    else:
        actual_quality = input("Enter your actual sleep quality: ")
        if actual_quality.lower() == 'q':
            break
        actual_quality = int(actual_quality)
        user_inputs['SleepQuality'] = actual_quality

    update_model(user_inputs)
