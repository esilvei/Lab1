import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.preprocessor import ImageProcessor


class FechaduraBiometricaPipeline:
    """
    Pipeline unificado de inferência.
    Encapsula o pré-processamento (CLAHE, Haar Cascade) e a Tiny-CNN.
    """

    def __init__(self, model_path: str, img_size: int = 32, threshold: float = 0.5):
        # AVISO: O 'threshold' ideal não necessariamente é 0.5.
        # Após treinar otimizando AUC, deve-se gerar a Curva ROC de Validação
        # E definir o threshold exato (ex: 0.65) onde o False Positive Rate chega a 0%
        # Mas mantendo o True Positive Rate alto o suficiente para os alunos.
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        self.model = tf.keras.models.load_model(model_path)

        self.processor = ImageProcessor(img_size)

        self.img_size = img_size
        self.threshold = threshold

    def predizer_imagem(self, frame: np.ndarray):
        """
        Recebe um frame BGR (ex: da webcam), processa e retorna a predição.
        """
        face_crop = self.processor.detect_and_crop(frame)

        if face_crop is None:
            return {"status": "Nenhuma face detectada", "autorizado": False, "probabilidade": 0.0, "face_crop": None}

        rosto_norm = face_crop.astype("float32") / 255.0

        rosto_input = np.expand_dims(rosto_norm, axis=(0, -1))

        pred_prob = self.model.predict(rosto_input, verbose=0)[0][0]
        autorizado = bool(pred_prob >= self.threshold)

        return {
            "status": "Sucesso",
            "autorizado": autorizado,
            "probabilidade": float(pred_prob),
            "face_crop": face_crop
        }