import cv2
import numpy as np
import random
import os
import sys
from src.config import Config
from src.data_utils import DataExtractor
from src.preprocessor import ImageProcessor
from src.dataset_manager import DatasetManager
from src.engine import ModelEngine
from src.evaluator import ModelEvaluator
from src.model import build_tiny_cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

class MIFExporter:
    def __init__(self, bit_width=8, frac_bits=7):
        self.bit_width = bit_width
        self.frac_bits = frac_bits
        self.max_val = (2 ** (bit_width - 1)) - 1
        self.min_val = -(2 ** (bit_width - 1))

    def to_fixed_point_hex(self, value):
        fixed_val = int(np.round(value * (2 ** self.frac_bits)))
        fixed_val = max(min(fixed_val, self.max_val), self.min_val)
        if fixed_val < 0:
            fixed_val = (1 << self.bit_width) + fixed_val
        return format(fixed_val, f'0{self.bit_width // 4}X')

    def export(self, model, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n[MIF] Exportando pesos quantizados (Q1.7)...")

        for layer in model.layers:
            weights = layer.get_weights()
            if not weights: continue

            for i, data in enumerate(weights):
                suffix = "weights" if i == 0 else "biases"
                filename = f"{layer.name}_{suffix}.mif"
                flat_data = data.flatten()

                mif_lines = [
                    f"DEPTH = {len(flat_data)};",
                    f"WIDTH = {self.bit_width};",
                    "ADDRESS_RADIX = HEX;",
                    "DATA_RADIX = HEX;",
                    "CONTENT BEGIN",
                    ""
                ]

                for addr, val in enumerate(flat_data):
                    mif_lines.append(f"{addr:X} : {self.to_fixed_point_hex(val)};")

                mif_lines.append("END;")

                with open(output_dir / filename, "w") as f:
                    f.write("\n".join(mif_lines))
                print(f" -> {filename} gerado.")


def main():
    cfg = Config()
    extractor = DataExtractor()
    processor = ImageProcessor(cfg.IMG_SIZE)
    ds_manager = DatasetManager(cfg)
    evaluator = ModelEvaluator(cfg)
    random.seed(42)

    print("\n" + "=" * 50)
    print("         INICIANDO PIPELINE      ")
    print("=" * 50)

    tar_path = cfg.RAW_DIR / "Selfie-dataset.tar.gz"
    if tar_path.exists():
        extractor.extract_tar(tar_path, cfg.RAW_DIR / "selfies", limit=7000)

    itens_raw = list(cfg.RAW_AUTORIZADO_DIR.iterdir())
    for item in itens_raw:
        nome_pessoa = extractor.sanitize_name(item.stem if item.is_file() else item.name)
        output_dir = cfg.INTERIM_AUTORIZADO_DIR / nome_pessoa
        output_dir.mkdir(parents=True, exist_ok=True)
        rostos = []
        if item.suffix.lower() == '.mp4':
            cap = cv2.VideoCapture(str(item))
            while len(rostos) < 400:
                ret, frame = cap.read()
                if not ret: break
                f = processor.detect_and_crop(frame)
                if f is not None: rostos.append(f)
            cap.release()
        elif item.is_dir():
            fotos = [p for p in item.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
            for f_p in fotos:
                if len(rostos) >= 400: break
                img = cv2.imread(str(f_p))
                f = processor.detect_and_crop(img)
                if f is not None: rostos.append(f)

        for i, r in enumerate(rostos[:400]):
            cv2.imwrite(str(output_dir / f"{i:04d}.jpg"), r)

        if 0 < len(rostos) < 400:
            for i in range(400 - len(rostos)):
                cv2.imwrite(str(output_dir / f"aug_{i:04d}.jpg"), processor.apply_augmentation(random.choice(rostos)))
        print(f" -> {nome_pessoa}: Processado")

    total_auth = len(list(cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")))
    count_0 = 0
    selfie_files = list((cfg.RAW_DIR / "selfies").glob("*.jpg"))
    random.shuffle(selfie_files)

    for f_p in selfie_files:
        if count_0 >= (total_auth * 2 - 300): break
        img = cv2.imdecode(np.fromfile(str(f_p), np.uint8), cv2.IMREAD_COLOR)
        f = processor.detect_and_crop(img)
        if f is not None:
            cv2.imwrite(str(cfg.NEGADOS_INTERIM_DIR / f"{count_0:05d}.jpg"), f)
            count_0 += 1

    for i in range(300):
        fundo = np.clip(np.full((32, 32), random.randint(40, 200)) + np.random.normal(0, 10, (32, 32)), 0, 255).astype(
            np.uint8)
        cv2.imwrite(str(cfg.NEGADOS_INTERIM_DIR / f"fundo_{i:04d}.jpg"), fundo)

    ds_manager.clean_processed()
    ds_manager.split_data(list(cfg.NEGADOS_INTERIM_DIR.glob("*.jpg")), "0_desconhecido")
    ds_manager.split_data(list(cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")), "1_autorizado")

    engine = ModelEngine(cfg, build_tiny_cnn)
    history, model = engine.train()

    evaluator.plot_training_history(history)
    evaluator.evaluate_on_test_set()
    mif_exporter = MIFExporter(bit_width=8, frac_bits=7)
    mif_exporter.export(model, cfg.PROJECT_ROOT / "export")

    print("\n" + "=" * 50)
    print("      PIPELINE CONCLUÍDA      ")
    print("=" * 50)


if __name__ == "__main__":
    main()