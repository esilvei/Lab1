import cv2
import numpy as np
import random
import shutil
from pathlib import Path

class ImageProcessor:
    def __init__(self, img_size=32):
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, frame):
        """Detecta face, aplica CLAHE para iluminação e faz o crop com padding oficial."""
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

            roi = gray[y1:y2, x1:x2]
            return cv2.resize(roi, (self.img_size, self.img_size))
        return None

    def apply_augmentation(self, image):
        """Augmentation avançada para melhorar a acurácia em ângulos variados."""
        img_aug = image.copy()

        angle = random.randint(-15, 15)
        m_rot = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1)
        img_aug = cv2.warpAffine(img_aug, m_rot, (self.img_size, self.img_size))

        if random.random() > 0.5:
            img_aug = cv2.flip(img_aug, 1)

        zoom = random.uniform(0.9, 1.1)
        new_size = int(self.img_size * zoom)
        img_aug = cv2.resize(img_aug, (new_size, new_size))

        if zoom > 1:
            start = (new_size - self.img_size) // 2
            img_aug = img_aug[start:start + self.img_size, start:start + self.img_size]
        else:
            pad = (self.img_size - new_size) // 2
            img_aug = cv2.copyMakeBorder(img_aug, pad, self.img_size - new_size - pad, pad,
                                         self.img_size - new_size - pad, cv2.BORDER_CONSTANT, value=0)

        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        return cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)

    def generate_synthetic_background(self):
        """Gera fundos para a Classe 0 para evitar falsos positivos com paredes."""
        cor_base = random.randint(40, 230)
        imagem = np.full((self.img_size, self.img_size), cor_base, dtype=np.float32)

        tipo_grad = random.choice(['horizontal', 'vertical', 'nenhum'])
        if tipo_grad != 'nenhum':
            luz = random.uniform(-40, 40)
            grad = np.linspace(0, luz, self.img_size)
            imagem = imagem + (grad if tipo_grad == 'horizontal' else grad[:, np.newaxis])

        ruido = np.random.normal(0, random.uniform(2.0, 15.0), (self.img_size, self.img_size))
        return np.clip(imagem + ruido, 0, 255).astype(np.uint8)


class DataPreprocessor:
    """Orquestrador modular para processamento de dados Autorizados e Desconhecidos."""

    def __init__(self, config, processor, extractor):
        self.cfg = config
        self.processor = processor
        self.extractor = extractor

    def clear_interim(self):
        """Prepara as pastas temporárias limpando execuções anteriores."""
        if self.cfg.INTERIM_DIR.exists():
            shutil.rmtree(self.cfg.INTERIM_DIR)
        for d in [self.cfg.INTERIM_AUTORIZADO_DIR, self.cfg.NEGADOS_INTERIM_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def process_authorized(self, max_fotos=400):
        """Processa vídeos e fotos da equipa completando a meta com augmentation."""
        print("\n[PREPROCESS] Processando Equipa (Classe 1)...")

        for item in self.cfg.RAW_AUTORIZADO_DIR.iterdir():
            nome_limpo = self.extractor.sanitize_name(item.name)
            if nome_limpo != item.name:
                novo_caminho = item.with_name(nome_limpo)

                if not novo_caminho.exists():
                    item.rename(novo_caminho)
                else:
                    print(f" -> Aviso: Ignorando renomeação. O caminho {novo_caminho.name} já existe.")

        for item in self.cfg.RAW_AUTORIZADO_DIR.iterdir():
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
                    try:
                        img_array = np.fromfile(str(f_p), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                        f = self.processor.detect_and_crop(img)
                        if f is not None: rostos.append(f)
                    except Exception:
                        continue

            for i, r in enumerate(rostos[:max_fotos]):
                cv2.imwrite(str(dest / f"{i:04d}.jpg"), r)

            if 0 < len(rostos) < max_fotos:
                for i in range(max_fotos - len(rostos)):
                    cv2.imwrite(str(dest / f"aug_{i:04d}.jpg"),
                                self.processor.apply_augmentation(random.choice(rostos)))

    def process_unknowns(self, ratio=1.5):
        """Processa Selfies + LFW e aplica Augmentation para robustez da Classe 0."""
        print("\n[PREPROCESS] Processando Desconhecidos (Classe 0)...")
        total_auth = len(list(self.cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")))
        meta = int(total_auth * ratio)

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
                img_array = np.fromfile(str(p), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                f = self.processor.detect_and_crop(img)

                if f is not None:
                    cv2.imwrite(str(self.cfg.NEGADOS_INTERIM_DIR / f"unknown_{count:05d}.jpg"), f)
                    count += 1

                    if count < meta:
                        f_aug = self.processor.apply_augmentation(f)
                        cv2.imwrite(str(self.cfg.NEGADOS_INTERIM_DIR / f"unknown_aug_{count:05d}.jpg"), f_aug)
                        count += 1
            except (cv2.error, OSError):
                continue