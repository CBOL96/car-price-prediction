from joblib import load
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify

modelo = load("modelo_car_prediction.joblib")

app = Flask('car_price_prediction')



@app.route('/prediction', methods=['POST'])
def predict_price():
    customer = pd.DataFrame([request.get_json()])
    prediction = modelo.predict(customer)


    result = {
        "predicted_price": float(prediction[0])
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port = 9696)








