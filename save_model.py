import joblib
from model import model

# Simpan model
joblib.dump(model, 'model.joblib')
print("Model berhasil disimpan ke model.joblib") 