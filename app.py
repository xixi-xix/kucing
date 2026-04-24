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

    return jsonify({'penyakit': hasil[0]})

if __name__ == '__main__':
    app.run(debug=True)