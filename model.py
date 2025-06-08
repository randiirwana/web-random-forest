import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import cv2
import joblib

def extract_features(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Tidak dapat membaca gambar: {image_path}")
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

def generate_and_save_plots(model, X_test, y_test, y_pred):
    # Buat folder static/plots jika belum ada
    os.makedirs('static/plots', exist_ok=True)
    
    # 1. Confusion Matrix (Interaktif dengan Plotly)
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_
    
    fig_cm = go.Figure(data=go.Heatmap(z=cm,
                                     x=labels,
                                     y=labels,
                                     colorscale='Blues'))
    
    fig_cm.update_layout(
        title='Confusion Matrix - Klasifikasi Sampah',
        xaxis_title='Prediksi',
        yaxis_title='Sebenarnya',
        xaxis_nticks=len(labels),
        yaxis_nticks=len(labels),
        template='plotly_white'
    )
    fig_cm.write_html('static/plots/confusion_matrix.html', auto_open=False)
    
    # 2. Feature Importance Plot (Interaktif dengan Plotly)
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": [f"Histogram Bin {i}" for i in range(len(feature_importances))],
        "Importance": feature_importances
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(20)
    
    fig_fi = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
                   title="20 Fitur Terpenting pada Random Forest",
                   color="Importance", color_continuous_scale=px.colors.sequential.Viridis)
    
    fig_fi.update_layout(
        yaxis_title="Fitur",
        xaxis_title="Tingkat Kepentingan",
        yaxis_autorange="reversed", # Untuk menampilkan fitur terpenting di atas
        template='plotly_white'
    )
    fig_fi.write_html('static/plots/feature_importance.html', auto_open=False)

# 1. Baca dataset labels
print("Membaca dataset labels...")
df = pd.read_csv('dataset_labels.csv')

# 2. Ekstrak fitur dari gambar
print("Mengekstrak fitur dari gambar...")
features = []
valid_indices = []

for idx, row in df.iterrows():
    # Dapatkan label dan nama file
    label = row['label']
    filename = row['filename']
    
    # Buat path lengkap ke gambar
    img_path = os.path.join('images', 'dataset-resized', label, filename)
    
    # Pastikan path gambar benar
    if not os.path.exists(img_path):
        print(f"File tidak ditemukan: {img_path}")
        continue
        
    feature = extract_features(img_path)
    if feature is not None:
        features.append(feature)
        valid_indices.append(idx)

if len(features) == 0:
    print("Tidak ada gambar yang berhasil diproses!")
    print("Pastikan file gambar berada di direktori 'images/dataset-resized'")
    exit()

# Filter dataframe untuk gambar yang valid
df = df.iloc[valid_indices]
X = np.array(features)
y = df['label']

print(f"\nJumlah gambar yang berhasil diproses: {len(features)}")

# 3. Eksplorasi dataset
print("\nInfo Dataset:")
print(df.info())
print("\nDistribusi Kelas:")
print(df['label'].value_counts())

# 4. Pisahkan data latih dan uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Buat model Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 6. Latih model
print("\nMelatih model...")
model.fit(X_train, y_train)

# 7. Prediksi data uji
y_pred = model.predict(X_test)

# 8. Evaluasi Model
print("\nAkurasi Model:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# 9. Generate dan simpan plot
generate_and_save_plots(model, X_test, y_test, y_pred)

# 10. Simpan model
joblib.dump(model, 'model.joblib')
print("\nModel berhasil disimpan ke 'model.joblib'")