
import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform(
            [[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]]
        )
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])

    return render_template('home.html', results=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)