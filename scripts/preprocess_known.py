import sys
from pathlib import Path
import cv2
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import Config

cfg = Config()


def aplicar_augmentation(imagem):
    img_aug = imagem.copy()

    if random.random() > 0.5:
        img_aug = cv2.flip(img_aug, 1)

    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    img_aug = cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)

    return img_aug


def processar_lote_videos(frames_pular=1, max_frames_por_pessoa=500):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    videos = list(cfg.RAW_AUTORIZADO_DIR.glob("*.mp4"))

    if not videos:
        print(f"Nenhum vídeo '.mp4' encontrado em:\n{cfg.RAW_AUTORIZADO_DIR}")
        return

    for video_path in videos:
        nome_pessoa = video_path.stem
        output_dir = cfg.INTERIM_AUTORIZADO_DIR / nome_pessoa
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Extraindo rostos de: {nome_pessoa}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir o vídeo {video_path.name}.")
            continue

        frames_lidos = 0
        rostos_salvos = 0
        imagens_extraidas = []

        while True:
            ret, frame = cap.read()
            if not ret or rostos_salvos >= max_frames_por_pessoa:
                break

            frames_lidos += 1
            if frames_lidos % frames_pular != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                pad = int(w * 0.15)
                y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

                rosto_crop = gray[y1:y2, x1:x2]
                rosto_final = cv2.resize(rosto_crop, (cfg.IMG_SIZE, cfg.IMG_SIZE))

                caminho_salvar = output_dir / f"{nome_pessoa}_original_{rostos_salvos:04d}.jpg"
                cv2.imwrite(str(caminho_salvar), rosto_final)

                imagens_extraidas.append(rosto_final)
                rostos_salvos += 1
                break

        cap.release()
        print(f"    Extraídos {rostos_salvos} quadros originais do vídeo.")

        if rostos_salvos < max_frames_por_pessoa and len(imagens_extraidas) > 0:
            faltam = max_frames_por_pessoa - rostos_salvos
            print(f"    Faltam {faltam} fotos para a meta. Completando de forma sintética...")

            for i in range(faltam):
                img_base = random.choice(imagens_extraidas)
                img_aug = aplicar_augmentation(img_base)

                caminho_aug = output_dir / f"{nome_pessoa}_aug_{i:04d}.jpg"
                cv2.imwrite(str(caminho_aug), img_aug)
                rostos_salvos += 1

        print(f"    Concluído! Total de {rostos_salvos} fotos finalizadas na pasta {output_dir.name}")


if __name__ == "__main__":
    processar_lote_videos(frames_pular=1, max_frames_por_pessoa=400)
    print("\nProcessamento em lote finalizado com sucesso!")