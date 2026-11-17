PLEASE DOWNLOAD ZIP FILE FROM RELIZE V2.0
# PlanetX 🚀

**AI-powered exoplanet detection using BLS and Convolutional Neural Networks (CNN).**

PlanetX is a project developed for the **NASA Space Apps Challenge 2026**. It explores how machine learning can assist in identifying potential exoplanet transit signals in stellar light curves.

## 🌌 About the Project

PlanetX combines two approaches:

- **Box Least Squares (BLS):** searches stellar light curves for periodic transit-like dips in brightness.
- **Convolutional Neural Network (CNN):** analyzes prepared light-curve signals and classifies them as `CONFIRMED` or `FALSE POSITIVE`.

The system first checks for a convincing periodic transit signal using BLS. If a suitable signal is detected, the processed light curve is passed to the CNN for classification.

## ✨ Features

- Upload stellar light curves in CSV format.
- Search for periodic transit signals using BLS.
- Analyze candidate signals with a CNN.
- Display detection results and model scores.
- Provide information about the analysis.

## 🛠️ Technologies

- Python
- FastAPI
- PyTorch
- Lightkurve
- NumPy
- Pandas
- HTML, CSS and JavaScript

## 📁 Project Structure

```text
PlanetX/
├── Backend/
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── planetx_cnn.pth
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
│
├── PlanetX_Training.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sashabogomolova2008-cmd/NasaSpaceAppChallenge2026.git
cd PlanetX
```

### 2. Set up the backend

```bash
cd Backend
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 4. Run the frontend

Open the `frontend` folder in Visual Studio Code and launch `index.html` using the Live Server extension.

Make sure the backend is running before analyzing a light curve.

## 📊 Input Data

The application accepts CSV files containing two columns:

- `time` — observation time.
- `flux` — measured stellar brightness.

The input data must contain sufficient valid observations for the analysis.

## 🧠 Machine Learning Model

The CNN is trained using labeled examples of confirmed exoplanets and false positives from the Kepler Objects of Interest (KOI) dataset.

The model currently uses two classes:

- `CONFIRMED`
- `FALSE POSITIVE`

The displayed model scores are outputs of the neural network and should not be interpreted as calibrated probabilities of a planet's existence.

## ⚠️ Scientific Limitations

- A periodic transit-like signal is not proof of an exoplanet.
- Eclipsing binaries and other astrophysical phenomena can produce false-positive signals.
- Failure to detect a transit does not prove that a star has no planets.
- CNN predictions depend on the quality and representativeness of the training data.

## 📓 Training

The `PlanetX_Training.ipynb` notebook contains the model training and data preparation workflow.

## 📄 License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
