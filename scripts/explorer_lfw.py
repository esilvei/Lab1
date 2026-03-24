import sys
from pathlib import Path
import os
import shutil
import random
import cv2
import matplotlib.pyplot as plt
import kagglehub
import tarfile

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import Config

cfg = Config()


def baixar_e_explorar_dataset():
    print("1. Baixando o dataset LFW via KaggleHub...")
    cache_path = kagglehub.dataset_download("atulanandjha/lfwpeople")

    destino_raw = cfg.RAW_DIR / "lfwpeople"
    print(f"2. Copiando os arquivos para a sua pasta: {destino_raw}...")
    shutil.copytree(cache_path, destino_raw, dirs_exist_ok=True)

    tgz_file = destino_raw / "lfw-funneled.tgz"
    if tgz_file.exists():
        print(f"3. Extraindo o arquivo compactado: {tgz_file.name}...")
        with tarfile.open(tgz_file, "r:gz") as tar:
            tar.extractall(path=destino_raw)
        print("Extração concluída!\n")

    print("4. Procurando imagens...")
    image_paths = [
        str(p) for p in destino_raw.rglob('*')
        if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]

    if not image_paths:
        print(f"ERRO: Nenhuma imagem encontrada em {destino_raw}.")
        return

    pessoas = {}
    for img_path in image_paths:
        nome_pessoa = Path(img_path).parent.name
        if nome_pessoa in pessoas:
            pessoas[nome_pessoa] += 1
        else:
            pessoas[nome_pessoa] = 1

    print(f"\n--- RESUMO DO DATASET EM DATA/RAW ---")
    print(f"Total de imagens: {len(image_paths)}")
    print(f"Total de pessoas diferentes: {len(pessoas)}")

    pessoas_ordenadas = sorted(pessoas.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 pessoas com mais fotos:")
    for nome, qtd in pessoas_ordenadas[:5]:
        print(f" - {nome}: {qtd} fotos")

    print("\nGerando visualização de 5 amostras aleatórias...")
    amostras = random.sample(image_paths, 5)

    plt.figure(figsize=(15, 6))
    plt.suptitle("LFW Dataset: Visão Original vs. Visão da FPGA (32x32 Grayscale)", fontsize=16)

    for i, img_path in enumerate(amostras):
        img = cv2.imread(img_path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        nome_pessoa = Path(img_path).parent.name
        nome_curto = nome_pessoa[:12] + "..." if len(nome_pessoa) > 12 else nome_pessoa

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Original\n{nome_curto}", fontsize=10)
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(resized, cmap='gray')
        plt.title("Entrada (32x32)", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    baixar_e_explorar_dataset()