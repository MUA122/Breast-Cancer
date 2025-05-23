from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and column names
model = joblib.load('full_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Define encoding mappings (adjust based on your preprocessing)
encoding_maps = {
    'age_group_5_years': lambda x: min(max(int(float(x) - 19) // 5, 1), 11),  # Bin age to 1–11 (20–24 to 70–74)
    'race_eth': {'White': 1, 'Black': 2, 'Hispanic': 3, 'Asian': 4, 'Other': 5, 'Unknown': 6},
    'first_degree_hx': {'No': 0, 'Yes': 1},
    'age_menarche': {'<12': 0, '12–14': 1, '>14': 2},
    'age_first_birth': {'<20': 0, '20–24': 1, '25–29': 2, '30–34': 3, '>34': 4},
    'BIRADS_breast_density': {'1': 1, '2': 2, '3': 3, '4': 4},
    'current_hrt': {'No': 0, 'Yes': 1},
    'menopaus': {'Pre': 1, 'Peri': 2, 'Post': 3},
    'bmi_group': {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4},
    'biophx': {'No': 0, 'Yes': 1}
}

@app.route('/')
def home():
    return render_template('index.html', columns=model_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {}
    try:
        for column in model_columns:
            value = request.form.get(column)
            if column == 'age_group_5_years':
                # Validate age input
                try:
                    value = float(value)
                    if not (20 <= value <= 74):
                        raise ValueError
                except (ValueError, TypeError):
                    return render_template('result.html', error=f"Age must be between 20 and 74.")
            else:
                # Validate categorical input
                if value not in encoding_maps[column]:
                    return render_template('result.html', error=f"Invalid value for {column}. Choose from {list(encoding_maps[column].keys())}.")

            # Convert to encoded value
            encoded_value = encoding_maps[column](value) if callable(encoding_maps[column]) else encoding_maps[column][value]
            input_data[column] = encoded_value

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure columns match the model's expected input
        input_df = input_df[model_columns]

        # Predict probability
        prob = model.predict_proba(input_df)[0][1]  # Probability of class 1 (breast cancer)
        probability = round(prob * 100, 2)
        print(f"Predicted probability: {probability}%")  # Debug logging

        # Return result
        return render_template('result.html', probability=probability)

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)  # Debug logging
        return render_template('result.html', error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)