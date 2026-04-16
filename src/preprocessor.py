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

    @staticmethod
    def _is_authorized_source(path_obj):
        return path_obj.is_dir() or path_obj.suffix.lower() == '.mp4'

    @staticmethod
    def _safe_imread(path_obj):
        """Leitura robusta para caminhos com acentos/Unicode no Windows."""
        try:
            return cv2.imdecode(np.fromfile(str(path_obj), np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error):
            return None

    @staticmethod
    def _unique_name(base_name, used_names):
        name = base_name
        idx = 2
        while name in used_names:
            name = f"{base_name}_{idx}"
            idx += 1
        used_names.add(name)
        return name

    def _estimate_authorized_target(self, base_counts):
        """Estima um alvo unico e conservador para todas as classes autorizadas."""
        if not base_counts:
            return 0

        counts = np.array(base_counts, dtype=np.float32)
        strategy = getattr(self.cfg, "AUTHORIZED_TARGET_STRATEGY", "quantile")

        if strategy == "quantile":
            target = int(np.quantile(counts, getattr(self.cfg, "AUTHORIZED_TARGET_QUANTILE", 0.25)))
        elif strategy == "median":
            target = int(np.median(counts))
        elif strategy == "mean":
            target = int(np.mean(counts))
        else:
            target = int(getattr(self.cfg, "AUTHORIZED_TARGET_FALLBACK", 500))

        target_min = int(getattr(self.cfg, "AUTHORIZED_TARGET_MIN", 400))
        target_max = int(getattr(self.cfg, "AUTHORIZED_TARGET_MAX", 600))

        target = max(target_min, min(target, target_max))
        return target

    def clear_interim(self):
        """Prepara as pastas temporárias limpando execuções anteriores[cite: 13]."""
        if self.cfg.INTERIM_DIR.exists():
            shutil.rmtree(self.cfg.INTERIM_DIR)
        for d in [self.cfg.INTERIM_AUTORIZADO_DIR, self.cfg.NEGADOS_INTERIM_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def process_authorized(self, target_fotos=None):
        """Processa vídeos e fotos da equipe balanceando todas as classes para um alvo unico."""
        print("\n[PREPROCESS] Processando Equipe (Classe 1)...")
        fontes = [p for p in self.cfg.RAW_AUTORIZADO_DIR.iterdir() if self._is_authorized_source(p)]
        fontes.sort(key=lambda p: p.name.lower())
        print(f"[PREPROCESS] Fontes autorizadas encontradas: {len(fontes)}")

        used_names = set()
        classes_buffer = []

        for item in fontes:
            raw_name = item.stem if item.is_file() else item.name
            nome_base = self.extractor.sanitize_name(raw_name)
            if not nome_base:
                continue

            nome_pessoa = self._unique_name(nome_base, used_names)
            print(f"  -> Lendo fonte: {item.name} -> classe {nome_pessoa}")
            rostos = []
            if item.suffix.lower() == '.mp4':
                cap = cv2.VideoCapture(str(item))
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    f = self.processor.detect_and_crop(frame)
                    if f is not None: rostos.append(f)
                cap.release()
            elif item.is_dir():
                fotos = [p for p in item.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                for f_p in fotos:
                    f = self.processor.detect_and_crop(self._safe_imread(f_p))
                    if f is not None: rostos.append(f)

            if rostos:
                classes_buffer.append((nome_pessoa, rostos))

        base_counts = [len(rostos) for _, rostos in classes_buffer]
        if target_fotos is None:
            target_fotos = self._estimate_authorized_target(base_counts)

        print(f"[PREPROCESS] Alvo automatico de faces por classe autorizada: {target_fotos}")

        total_classes = 0
        for nome_pessoa, rostos in classes_buffer:
            dest = self.cfg.INTERIM_AUTORIZADO_DIR / nome_pessoa
            dest.mkdir(parents=True, exist_ok=True)

            base_detectadas = len(rostos)
            if base_detectadas >= target_fotos:
                rostos_base = random.sample(rostos, target_fotos)
                num_aug = 0
            else:
                rostos_base = list(rostos)
                num_aug = target_fotos - base_detectadas

            for i, r in enumerate(rostos_base):
                cv2.imwrite(str(dest / f"{i:04d}.jpg"), r)

            if num_aug > 0 and rostos_base:
                for i in range(num_aug):
                    cv2.imwrite(
                        str(dest / f"aug_{i:04d}.jpg"),
                        self.processor.apply_augmentation(random.choice(rostos_base))
                    )

            total_final = len(list(dest.glob("*.jpg")))

            if total_final > 0:
                total_classes += 1
                print(
                    f"  -> Classe autorizada: {nome_pessoa} | "
                    f"base detectadas: {base_detectadas} | "
                    f"aug geradas: {num_aug} | total final: {total_final}"
                )

        print(f"[PREPROCESS] Classes autorizadas processadas: {total_classes}")

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

