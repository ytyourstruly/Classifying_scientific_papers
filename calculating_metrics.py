import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Function to calculate metrics for a given CSV file
def calculate_metrics(csv_file):
    # Extract model name from filename (assumes the format is evaluation_modelname.csv)
    model_name = os.path.basename(csv_file).split('_')[1].replace('.csv', '')
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Define the actual and predicted labels
    y_true = df['truth']  # Replace 'Actual' with the column for actual labels
    y_pred = df['prediction']  # Replace 'Predicted' with the column for predicted labels
    
    # Define pos_label based on the model (you can modify this logic based on your needs)
    pos_label = 'Human'
        # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    
    # Print the results
    print(f"Metrics for {model_name} ({csv_file}):")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print("\n")

# List of your CSV files (update with your actual filenames)
csv_files = ["evaluation_fakespot.csv", "evaluation_hellosimpleai-detector.csv", "evaluation_openai-detector.csv"]  # Replace with your actual file paths

# Iterate over the files and calculate metrics
for csv_file in csv_files:
    calculate_metrics(csv_file)
