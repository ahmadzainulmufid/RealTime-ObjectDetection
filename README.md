# RealTime-ObjectDetection 🧠🎥

Proyek ini merupakan sistem **deteksi objek secara real-time** berbasis YOLOv11x, dengan antarmuka web menggunakan **Flask** dan dukungan webcam.

## 🗂️ Struktur Proyek
📁 templates/
    └── index.html         # Template tampilan web
📄 app.py                  # Script Flask

---
⚙️ Instalasi

1. Clone Repo
git clone https://github.com/ahmadzainulmufid/RealTime-ObjectDetection.git
cd RealTime-ObjectDetection

2. Buat Virtual Env dan Install Dependencies
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows

pip install ultralytics opencv-python flask

3. Unduh Model best.pt
Karena GitHub tidak mendukung file >100MB, silakan unduh model dari link berikut:

🔗 https://drive.google.com/drive/folders/1TZZx6EYl8qgsQHu19PFTMKCMtXCUg_se Letakkan file best.pt di root folder proyek ini.

---
🚀 Jalankan Aplikasi
python app.py
Akses di browser: http://localhost:5050

🧠 Model
Model best.pt merupakan hasil training YOLOv11x pada dataset OK/NG dengan 2 class:

Class 0: NG (Not Good)
Class 1: OK

Dataset diproses dan dibagi dalam rasio 80/10/10: Train / Val / Test.
