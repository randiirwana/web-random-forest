from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Batasi ukuran file ke 16MB

# Pastikan folder uploads ada
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Folder upload berhasil dibuat di: {UPLOAD_FOLDER}")
except Exception as e:
    logger.error(f"Gagal membuat folder upload: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Gagal membaca gambar dari path: {image_path}")
            return None
        
        # Resize gambar ke ukuran yang sama
        img = cv2.resize(img, (64, 64))
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ekstrak fitur histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Normalisasi histogram
        hist = hist / hist.sum()
        
        return hist
    except Exception as e:
        logger.error(f"Error dalam ekstraksi fitur: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analisis')
def analisis():
    return render_template('analisis.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logger.warning("Tidak ada file dalam request")
            return jsonify({'error': 'Tidak ada file yang diunggah'})
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Nama file kosong")
            return jsonify({'error': 'Tidak ada file yang dipilih'})
        
        if file and allowed_file(file.filename):
            # Generate nama file unik dengan timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Simpan file
            try:
                file.save(filepath)
                logger.info(f"File berhasil disimpan di: {filepath}")
            except Exception as e:
                logger.error(f"Gagal menyimpan file: {str(e)}")
                return jsonify({'error': 'Gagal menyimpan file'})
            
            # Ekstrak fitur
            features = extract_features(filepath)
            if features is None:
                return jsonify({'error': 'Gagal memproses gambar'})
            
            # Load model
            try:
                model = joblib.load('model.joblib')
            except Exception as e:
                logger.error(f"Gagal memuat model: {str(e)}")
                return jsonify({'error': 'Model tidak ditemukan'})
            
            # Prediksi
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            # Dapatkan probabilitas untuk kelas yang diprediksi
            pred_prob = probability[model.classes_.tolist().index(prediction)]
            
            # Konversi label ke Bahasa Indonesia
            label_map = {
                'cardboard': 'Kardus',
                'glass': 'Kaca',
                'metal': 'Logam',
                'paper': 'Kertas',
                'plastic': 'Plastik',
                'trash': 'Sampah'
            }
            
            result = {
                'prediction': label_map.get(prediction, prediction),
                'probability': float(pred_prob),
                'image_path': f'/static/uploads/{filename}'
            }
            
            logger.info(f"Prediksi berhasil: {result}")
            return jsonify(result)
        
        logger.warning(f"Format file tidak didukung: {file.filename}")
        return jsonify({'error': 'Format file tidak didukung'})
    
    except Exception as e:
        logger.error(f"Error dalam proses prediksi: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan dalam memproses request'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 