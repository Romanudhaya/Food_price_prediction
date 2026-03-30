from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model artifacts
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
columns = data['columns']
mean = data['mean']
std = data['std']
categorical_options = data['categorical_options']


@app.route('/')
def home():
    return render_template('index.html', options=categorical_options)


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    # Convert numeric fields
    num_features = [
        'num_items', 'avg_item_price', 'discount_percent',
        'delivery_distance_km', 'delivery_rating',
        'customer_age', 'num_previous_orders'
    ]

    for col in num_features:
        form_data[col] = float(form_data[col])

    # Create input vector
    input_dict = {col: 0 for col in columns}

    # Fill numeric
    for col in num_features:
        input_dict[col] = (form_data[col] - mean[col]) / std[col]

    # One-hot encoding
    for cat_col in categorical_options:
        value = form_data[cat_col]
        col_name = f"{cat_col}_{value}"
        if col_name in input_dict:
            input_dict[col_name] = 1

    # Convert to array
    input_array = np.array([list(input_dict.values())])

    prediction = model.predict(input_array)[0]

    return render_template('index.html',
                           prediction=round(prediction, 2),
                           options=categorical_options)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)