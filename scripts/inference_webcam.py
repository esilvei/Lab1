import cv2
import numpy as np
import tensorflow as tf
from src.config import Config
from src.preprocessor import ImageProcessor

cfg = Config()

def iniciar_inferencia():
    # 1. Carrega o modelo final
    model_path = cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
    if not model_path.exists():
        print(f"Erro Crítico: Modelo não encontrado em {model_path}")
        return

    model = tf.keras.models.load_model(str(model_path))
    processor = ImageProcessor(cfg.IMG_SIZE)

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

            # 5. Normalização e Predição
            rosto_norm = rosto_resized.astype("float32") / 255.0
            rosto_input = np.expand_dims(rosto_norm, axis=(0, -1))
            pred_prob = model.predict(rosto_input, verbose=0)[0][0]

            # 6. Lógica de Cores e Texto
            if pred_prob > 0.5:
                label = f"AUTORIZADO ({pred_prob * 100:.1f}%)"
                color = (0, 255, 0) # Verde
            else:
                label = f"NEGADO ({pred_prob * 100:.1f}%)"
                color = (0, 0, 255) # Vermelho

            # --- DESENHO DO QUADRADO E TEXTO ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Opcional: Mostra o que a rede está a ver num ecrã separado
            cv2.imshow('Visao da CNN (32x32)', cv2.resize(rosto_resized, (160, 160)))

        # Exibe o frame final
        cv2.imshow('Fechadura Biometrica', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_inferencia()