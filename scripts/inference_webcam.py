import cv2
import numpy as np
import tensorflow as tf
import json
from src.config import Config
from src.preprocessor import ImageProcessor
from src.model_io import load_tinycnn_model

cfg = Config()

def iniciar_inferencia():
    # 1. Carrega o modelo final
    model_path = cfg.MODEL_PATH
    if not model_path.exists():
        print(f"Erro Crítico: Modelo não encontrado em {model_path}")
        return

    model = load_tinycnn_model(model_path, compile_model=False)
    processor = ImageProcessor(cfg.IMG_SIZE)
    class_names = None
    if cfg.CLASS_MAP_PATH.exists():
        with open(cfg.CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        class_names = [None] * len(class_indices)
        for name, idx in class_indices.items():
            class_names[int(idx)] = name

    confidence_threshold = 0.55

    cap = cv2.VideoCapture(0)
    print("Iniciando Webcam... Clique na janela do vídeo e pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Converte para escala de cinza e aplica CLAHE (como no treino)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        # 3. Detecta múltiplos rostos na imagem
        faces = processor.face_cascade.detectMultiScale(gray_clahe, 1.2, 5, minSize=(60, 60))

        if len(faces) == 0:
            # Texto caso ninguém esteja na câmera
            cv2.putText(frame, "Procurando rosto...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for (x, y, w, h) in faces:
            # 4. Extração com padding (idêntico ao dataset)
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray_clahe.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray_clahe.shape[1], x + w + pad)

            rosto_crop = gray_clahe[y1:y2, x1:x2]
            rosto_resized = cv2.resize(rosto_crop, (cfg.IMG_SIZE, cfg.IMG_SIZE))

            # 5. Normalizao e Predio
            rosto_norm = rosto_resized.astype("float32") / 255.0
            rosto_input = np.expand_dims(rosto_norm, axis=(0, -1))
            probs = model.predict(rosto_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])

            if class_names and pred_idx < len(class_names):
                raw_class = class_names[pred_idx]
                is_unknown = raw_class == cfg.UNKNOWN_CLASS_NAME
                
                # Remove o prefixo de ID (ex: "10_Naira_Beatriz" -> "Naira Beatriz")
                if "_" in raw_class:
                    display_name = raw_class.split("_", 1)[1].replace("_", " ")
                else:
                    display_name = raw_class
            else:
                display_name = f"classe {pred_idx}"
                is_unknown = pred_idx == 0

            # 6. Lgica de Cores e Texto
            if is_unknown or pred_conf < confidence_threshold:
                label = f"NEGADO {display_name} ({pred_conf * 100:.1f}%)"
                color = (0, 0, 255)
            else:
                label = f"LIBERADO {display_name} ({pred_conf * 100:.1f}%)"
                color = (0, 255, 0)

            # --- DESENHO DO QUADRADO E TEXTO ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Opcional: Mostra o que a rede está a ver num ecrã separado
            cv2.imshow('Visao da CNN (32x32)', cv2.resize(rosto_resized, (160, 160)))

        cv2.imshow('Fechadura Biometrica', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_inferencia()