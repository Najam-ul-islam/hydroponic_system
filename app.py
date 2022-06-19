from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
model = loaded_model = joblib.load("model_final.sav")


app = Flask(__name__)


@app.route("/predict", methods=["GET"])
def predict():
    if request.method == 'GET':
        ppm = request.args.get('PPM')
        wl = request.args.get('Water_Level')
        tmp = request.args.get('Temp')
        hum = request.args.get('Humidity')
        ldr = request.args.get('LDR')
        df = pd.DataFrame([[ppm, wl, tmp, hum, ldr]], columns=[
                          'PPM', 'Water_Level', 'Temp', 'Humidity', 'LDR'], index=['input'], dtype=np.float64)
        prediction = model.predict(df)[0]
        original_input = {'PPM': ppm, 'Water_Level': wl,
                          'Temp': tmp, 'Humidity': hum, 'LDR': ldr, 'Label': prediction}
    return jsonify(original_input)


if __name__ == '__main__':
    app.run(debug=True)
