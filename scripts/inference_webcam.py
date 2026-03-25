import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.config import Config

cfg = Config()


def iniciar_inferencia():
    model_path = cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
    if not model_path.exists():
        print(f"Erro: Modelo não encontrado em {model_path}")
        return

    model = tf.keras.models.load_model(str(model_path))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    print("Iniciando Webcam... Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

            rosto_crop = gray[y1:y2, x1:x2]
            rosto_resized = cv2.resize(rosto_crop, (cfg.IMG_SIZE, cfg.IMG_SIZE))
            rosto_norm = rosto_resized.astype("float32") / 255.0
            rosto_input = np.expand_dims(rosto_norm, axis=(0, -1))  # (1, 32, 32, 1)

            pred_prob = model.predict(rosto_input, verbose=0)[0][0]

            if pred_prob > 0.8:
                label = f"AUTORIZADO ({pred_prob * 100:.1f}%)"
                color = (0, 255, 0)  # Verde
            else:
                label = f"NEGADO ({pred_prob * 100:.1f}%)"
                color = (0, 0, 255)  # Vermelho

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Fechadura Biometrica - Tiny-CNN', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_inferencia()