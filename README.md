import Image from 'next/image';

# 🧠 Copy-Move Forgery Detection (MICC-F220 Dataset)

A full-stack web application for detecting copy-move image forgeries using deep learning with VGG16, built with:

- 🔥 PyTorch + Flask for backend
- ⚛️ Next.js + React for frontend
- 🧠 Real-time metrics: accuracy, precision, recall, F1-score
- 🖼️ Bounding box forgery visualization on images
- 📸 MICC-F220 dataset (original and tampered images)

---

## 🗂️ Project Structure

```bash
CopyMoveForgeryDetection/
├── backend/             # Flask + PyTorch server
│   ├── app.py
│   ├── train_model.py
│   ├── predict.py
│   ├── utils.py
│   ├── draw.py
│   ├── model_loader.py
│   ├── history_plot.py
│   ├── static/              # Input/output images and stats
│   ├── model_output/        # Trained model, metrics, features
│   │   ├── vgg_forgery_model.pkl
│   │   ├── metrics.json
│   │   ├── X.npy, Y.npy
│   │   └── vgg_history.pckl
│   └── requirements.txt
├── frontend/           # Next.js frontend
│   ├── pages/
│   ├── components/
│   ├── styles/
│   ├── public/
│   │   └── placeholder.jpg
│   ├── package.json
│   └── next.config.js
├── MICC-F220/          # ⚠️ Not uploaded to GitHub. Must add locally
│   ├── Au/             # Authentic/original images
│   └── Tu/             # Tampered images
├── .gitignore
└── README.mdx
