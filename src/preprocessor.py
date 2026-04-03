import cv2
import numpy as np
import random
import shutil
from pathlib import Path


class ImageProcessor:
    def __init__(self, img_size=32):
        self.img_size = img_size
        # Carrega o classificador oficial para detecção facial [cite: 19]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, frame):
        """Detecta face, aplica CLAHE para iluminação e faz o crop com padding oficial."""
        if frame is None:
            return None

        # Conversão para escala de cinza conforme requisito da FPGA [cite: 20]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # OTIMIZAÇÃO: CLAHE (Normalização de iluminação adaptativa)
        # Fundamental para que a rede reconheça faces em diferentes condições de luz [cite: 19]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Detecção facial sincronizada com os parâmetros de treino (1.2, 5) [cite: 7, 8]
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Enquadramento com 15% de padding para capturar bordas do rosto [cite: 7, 19]
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

            roi = gray[y1:y2, x1:x2]
            return cv2.resize(roi, (self.img_size, self.img_size))
        return None

    def apply_augmentation(self, image):
        """Augmentation avançada para melhorar a acurácia em ângulos variados."""
        img_aug = image.copy()

        # 1. Rotação leve (Essencial para inclinação da cabeça na webcam)
        angle = random.randint(-15, 15)
        m_rot = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1)
        img_aug = cv2.warpAffine(img_aug, m_rot, (self.img_size, self.img_size))

        # 2. Espelhamento horizontal (50% de chance)
        if random.random() > 0.5:
            img_aug = cv2.flip(img_aug, 1)

        # 3. Variação de Zoom e Escala
        zoom = random.uniform(0.9, 1.1)
        new_size = int(self.img_size * zoom)
        img_aug = cv2.resize(img_aug, (new_size, new_size))

        # Crop ou Padding para manter o tamanho 32x32 [cite: 20]
        if zoom > 1:
            start = (new_size - self.img_size) // 2
            img_aug = img_aug[start:start + self.img_size, start:start + self.img_size]
        else:
            pad = (self.img_size - new_size) // 2
            img_aug = cv2.copyMakeBorder(img_aug, pad, self.img_size - new_size - pad, pad,
                                         self.img_size - new_size - pad, cv2.BORDER_CONSTANT, value=0)

        # 4. Ajuste de Brilho/Contraste (Simula variações de ambiente)
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        return cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)

class DataPreprocessor:
    """Orquestrador modular para processamento de dados Autorizados e Desconhecidos."""

    def __init__(self, config, processor, extractor):
        self.cfg = config
        self.processor = processor
        self.extractor = extractor

    def clear_interim(self):
        """Prepara as pastas temporárias limpando execuções anteriores[cite: 13]."""
        if self.cfg.INTERIM_DIR.exists():
            shutil.rmtree(self.cfg.INTERIM_DIR)
        for d in [self.cfg.INTERIM_AUTORIZADO_DIR, self.cfg.NEGADOS_INTERIM_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def process_authorized(self, max_fotos=400):
        """Processa vídeos e fotos da equipe completando a meta com augmentation[cite: 7, 19]."""
        print("\n[PREPROCESS] Processando Equipe (Classe 1)...")
        for item in self.cfg.RAW_AUTORIZADO_DIR.iterdir():
            nome_limpo = self.extractor.sanitize_name(item.name)
            if nome_limpo != item.name:
                item.rename(item.with_name(nome_limpo))

            nome_pessoa = item.stem
            dest = self.cfg.INTERIM_AUTORIZADO_DIR / nome_pessoa
            dest.mkdir(parents=True, exist_ok=True)

            rostos = []
            if item.suffix.lower() == '.mp4':
                cap = cv2.VideoCapture(str(item))
                while len(rostos) < max_fotos:
                    ret, frame = cap.read()
                    if not ret: break
                    f = self.processor.detect_and_crop(frame)
                    if f is not None: rostos.append(f)
                cap.release()
            elif item.is_dir():
                fotos = [p for p in item.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                for f_p in fotos:
                    if len(rostos) >= max_fotos: break
                    f = self.processor.detect_and_crop(cv2.imread(str(f_p)))
                    if f is not None: rostos.append(f)

            # Salva fotos base
            for i, r in enumerate(rostos[:max_fotos]):
                cv2.imwrite(str(dest / f"{i:04d}.jpg"), r)

            # Aplica Augmentation para atingir a meta de fotos [cite: 7]
            if 0 < len(rostos) < max_fotos:
                for i in range(max_fotos - len(rostos)):
                    cv2.imwrite(str(dest / f"aug_{i:04d}.jpg"),
                                self.processor.apply_augmentation(random.choice(rostos)))

    def process_unknowns(self, ratio=1.5):
        """Processa Selfies + LFW e aplica Augmentation.
           Focamos em usar APENAS as faces do LFW/Selfies como 'Hard Negatives'.
        """
        print("\n[PREPROCESS] Processando Desconhecidos (Classe 0)...")
        total_auth = len(list(self.cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")))
        meta = int(total_auth * ratio)

        # Suporte ao LFW e Selfies minerados recursivamente 
        image_paths = []
        for d in [self.cfg.RAW_DIR / "selfies", self.cfg.RAW_DIR / "lfw_extracted"]:
            if d.exists():
                image_paths.extend(
                    [p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

        random.shuffle(image_paths)
        count = 0
        for p in image_paths:
            if count >= meta: break
            try:
                # Carregamento via numpy para evitar erro com caracteres especiais no LFW [cite: 19]
                img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                f = self.processor.detect_and_crop(img)

                if f is not None:
                    # Salva face original do estranho
                    cv2.imwrite(str(self.cfg.NEGADOS_INTERIM_DIR / f"unknown_{count:05d}.jpg"), f)
                    count += 1

                    # OTIMIZAÇÃO: Augmentation para a Classe 0 (Duplica a diversidade negativa)
                    if count < meta:
                        f_aug = self.processor.apply_augmentation(f)
                        cv2.imwrite(str(self.cfg.NEGADOS_INTERIM_DIR / f"unknown_aug_{count:05d}.jpg"), f_aug)
                        count += 1
            except (cv2.error, OSError):
                continue

