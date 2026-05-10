from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    return ("Churn" if prediction == 1 else "No Churn"), probability


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        input_data = {
            'gender':           form['gender'],
            'SeniorCitizen':    int(form['SeniorCitizen']),
            'Partner':          form['Partner'],
            'Dependents':       form['Dependents'],
            'tenure':           float(form['tenure']),
            'PhoneService':     form['PhoneService'],
            'MultipleLines':    form['MultipleLines'],
            'InternetService':  form['InternetService'],
            'OnlineSecurity':   form['OnlineSecurity'],
            'OnlineBackup':     form['OnlineBackup'],
            'DeviceProtection': form['DeviceProtection'],
            'TechSupport':      form['TechSupport'],
            'StreamingTV':      form['StreamingTV'],
            'StreamingMovies':  form['StreamingMovies'],
            'Contract':         form['Contract'],
            'PaperlessBilling': form['PaperlessBilling'],
            'PaymentMethod':    form['PaymentMethod'],
            'MonthlyCharges':   float(form['MonthlyCharges']),
            'TotalCharges':     float(form['TotalCharges']),
        }

        prediction, probability = make_prediction(input_data)

        return jsonify({
            'prediction':  prediction,
            'probability': f"{probability * 100:.1f}%",
            'color':       'red' if prediction == 'Churn' else 'green'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False)