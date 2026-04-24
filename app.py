from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('model_kucing.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = [[
        data['nafsu_makan'],
        data['muntah'],
        data['lesu'],
        data['demam'],
        data['diare'],
        data['batuk'],
        data['bersin'],
        data['berat_badan_turun'],
        data['mata_berair']
    ]]

    pred = model.predict(input_data)
    hasil = le.inverse_transform(pred)

    return jsonify({'penyakit': str(hasil[0])})

import os
if __name__=='__main__':
    port = int(os.environ.get('PORT,5000))
app.run(host='0.0.0.0', port=port)
