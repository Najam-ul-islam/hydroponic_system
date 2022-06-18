from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
model = loaded_model = joblib.load("model_final.sav")


app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        ppm = request.form['PPM']
        wl = request.form['Water Level']
        tmp = request.form['Temp']
        hum = request.form['Humidity']
        ldr = request.form['LDR']
        df = pd.DataFrame([[ppm, wl, tmp, hum, ldr]], columns=[
                          'PPM', 'Water Level', 'Temp', 'Humidity', 'LDR'], index=['input'], dtype=np.float64)
        prediction = model.predict(df)[0]
        original_input = {'PPM': ppm, 'Water Level': wl,
                          'Temp': tmp, 'Humidity': hum, 'LDR': ldr, 'Result': prediction}
    return jsonify(original_input)


if __name__ == '__main__':
    app.run(debug=True)
