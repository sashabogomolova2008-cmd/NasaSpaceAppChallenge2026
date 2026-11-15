
from pathlib import Path

import io
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import load_planetx_model
from preprocessing import find_transit, prepare_signal


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "planetx_cnn.pth"

INPUT_LENGTH = 2000

CLASS_NAMES = {
    0: "CONFIRMED",
    1: "FALSE POSITIVE"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI(
    title="PlanetX API",
    description="Exoplanet transit detection and CNN classification",
    version="1.0.0"
)

# Разрешаем локальному frontend обращаться к backend.
# Для разработки с Live Server используем localhost:5500.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Загружаем модель один раз при запуске сервера
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл обученной модели: {MODEL_PATH}"
    )

model = load_planetx_model(
    str(MODEL_PATH),
    device
)


@app.get("/")
def home():
    return {
        "project": "PlanetX",
        "status": "running",
        "message": "PlanetX backend is ready."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/analyze")
async def analyze_light_curve(
    file: UploadFile = File(...)
):

    # Проверяем формат файла
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Пожалуйста, загрузите CSV-файл."
        )

    try:
        content = await file.read()

        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Файл слишком большой. Максимум 20 MB."
            )

        df = pd.read_csv(io.BytesIO(content))

        # Нормализуем названия столбцов
        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        if "time" not in df.columns or "flux" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV должен содержать столбцы time и flux."
            )

        time = pd.to_numeric(
            df["time"],
            errors="coerce"
        ).to_numpy(dtype=float)

        flux = pd.to_numeric(
            df["flux"],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid = np.isfinite(time) & np.isfinite(flux)

        time = time[valid]
        flux = flux[valid]

        if len(time) < 100:
            raise HTTPException(
                status_code=400,
                detail="Нужно минимум 100 корректных измерений."
            )

        if len(time) > 500000:
            raise HTTPException(
                status_code=400,
                detail="Слишком много измерений для этого прототипа."
            )

        # Сначала ищем транзит с помощью BLS
        transit = find_transit(time, flux)

        if not transit["found"]:

            return {
                "status": "no_transit",
                "message": "Убедительный периодический транзит не найден.",
                "bls": transit,
                "neural_network": None
            }

        # Если BLS нашёл сигнал, готовим его для CNN
        signal = prepare_signal(
            time=time,
            flux=flux,
            period=transit["period_days"],
            epoch=transit["epoch"],
            duration_days=transit["duration_days"]
        )

        # Форма входа CNN: [batch, channels, length]
        input_tensor = torch.tensor(
            signal,
            dtype=torch.float32
        ).reshape(1, 1, INPUT_LENGTH).to(device)

        # Предсказание
        model.eval()

        with torch.no_grad():

            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        confirmed_probability = float(
            probabilities[0].item() * 100
        )

        false_positive_probability = float(
            probabilities[1].item() * 100
        )

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        predicted_label = CLASS_NAMES[predicted_class]

        return {
            "status": "transit_found",
            "message": "Периодический транзитоподобный сигнал найден.",
            "bls": transit,
            "neural_network": {
                "predicted_class": predicted_label,
                "confirmed_probability": round(
                    confirmed_probability, 2
                ),
                "false_positive_probability": round(
                    false_positive_probability, 2
                ),
                "probabilities_are_model_scores": True
            }
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки файла: {str(error)}"
        )
