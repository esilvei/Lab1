import sys
from pathlib import Path
import os
import cv2

# Adiciona o diretório raiz do projeto ao sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import Config

cfg = Config()

def preprocess_unknown():
    src_dir = cfg.RAW_DIR / "lfwpeople" / "lfw_funneled"
    # Salva agora em data/interim/0_desconhecido
    interim_dir = project_root / "data" / "interim" / "0_desconhecido"
    interim_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in src_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"Total de imagens encontradas: {len(image_paths)}")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Erro ao ler {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        # Salva com o mesmo nome do arquivo original
        dst_path = interim_dir / img_path.name
        cv2.imwrite(str(dst_path), resized)
    print(f"Imagens processadas salvas em: {interim_dir}")

if __name__ == "__main__":
    preprocess_unknown()
