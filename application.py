from flask import (
    Flask,
    request,
    render_template
)
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    data = CustomData(
        transaction_amount=float(request.form.get('transaction_amount')),
        quantity=int(request.form.get('quantity')),
        customer_age=int(request.form.get('customer_age')),
        account_age_days=int(request.form.get('account_age_days')),
        payment_method=request.form.get('payment_method'),
        product_category=request.form.get('product_category'),
        customer_location=request.form.get('customer_location'),
        device_used=request.form.get('device_used'),
        transaction_date=request.form.get('transaction_date'),
    )

    pred_df = data.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()
    preds, probas = predict_pipeline.predict(pred_df)

    result_label = "Fraudulent" if preds[0] == 1 else "Legitimate"
    fraud_probability = round(probas[0], 4)

    return render_template(
        'home.html',
        results=result_label,
        probability=fraud_probability,
        form_data=request.form
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
