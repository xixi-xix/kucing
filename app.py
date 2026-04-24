from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# =========================
# LOAD MODEL (WAJIB DI ATAS)
# =========================
model = joblib.load('model_kucing.pkl')
le = joblib.load('label_encoder.pkl')

# =========================
# HOME ROUTE
# =========================
@app.route('/')
def home():
    return jsonify({
        "status": "aktif",
        "message": "API Kucing berjalan 🚀"
    })

# =========================
# PREDICT ROUTE
# =========================
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

    return jsonify({
        'penyakit': str(hasil[0])
    })

# =========================
# RUN (LOCAL + RAILWAY)
# =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
