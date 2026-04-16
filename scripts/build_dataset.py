import shutil
import random
from src.config import Config
cfg = Config()

random.seed(42)

def split_and_copy(files: list, class_name: str, train_ratio=0.8, val_ratio=0.1):
    """
    Divide uma lista de arquivos e copia para as pastas de treino, validação e teste.
    """
    random.shuffle(files)
    total = len(files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        dest_dir = cfg.PROCESSED_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in split_files:
            shutil.copy(f, dest_dir / f.name)

    print(f"Classe {class_name[:20]:<20} | Treino: {len(splits['train']):<4} | Val: {len(splits['val']):<4} | Teste: {len(splits['test']):<4} | Total: {total}")
    return total

def build_dataset():
    cfg.validate()
    if not cfg.INTERIM_DIR.exists():
        print(f"ERRO: A pasta {cfg.INTERIM_DIR} não existe. Processe os dados interim primeiro.")
        return

    if cfg.PROCESSED_DIR.exists():
        print(f"Limpando dados processados anteriores em: {cfg.PROCESSED_DIR.name}...")
        shutil.rmtree(cfg.PROCESSED_DIR)

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    totals = {}

    # --- 1. PROCESSAR CLASSE 0 (DESCONHECIDOS) ---
    print("\n--- Organizando Classe 0 (Negados) ---")
    negados_files = list(cfg.NEGADOS_INTERIM_DIR.glob("*.jpg"))
    if negados_files:
        totals["Classe 0"] = split_and_copy(negados_files, "0_desconhecido")
    else:
        print("Aviso: Nenhuma imagem encontrada para a Classe 0.")

    # --- 2. PROCESSAR CLASSES AUTORIZADAS ---
    if cfg.INTERIM_AUTORIZADO_DIR.exists():
        class_dirs = sorted([d for d in cfg.INTERIM_AUTORIZADO_DIR.iterdir() if d.is_dir()])
        if not class_dirs:
            print("Aviso: Nenhuma pasta de aluno encontrada para classes autorizadas.")
        if cfg.is_binary_mode:
            print("\n--- Organizando Classe 1 (Autorizados - Binario) ---")
            autorizados = []
            for class_dir in class_dirs:
                autorizados.extend(list(class_dir.rglob("*.jpg")))
            if autorizados:
                totals["Classe 1"] = split_and_copy(autorizados, cfg.BINARY_AUTHORIZED_CLASS_NAME)
        else:
            print("\n--- Organizando Classes de Autorizados (Multiclasse) ---")
            for idx, class_dir in enumerate(class_dirs, start=1):
                autorizados_files = list(class_dir.rglob("*.jpg"))
                class_name = f"{idx}_{class_dir.name}"
                if autorizados_files:
                    totals[f"Classe {class_name}"] = split_and_copy(autorizados_files, class_name)

    print("\n" + "="*50)
    modo = "BINARIO" if cfg.is_binary_mode else "MULTICLASSE"
    print(f"RESUMO DO DATASET PROCESSADO ({modo})")
    print("="*50)
    for cls, count in totals.items():
        percentage = (count / sum(totals.values())) * 100
        print(f"{cls}: {count} imagens ({percentage:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    print(f"Iniciando construção do dataset em: {cfg.PROCESSED_DIR}")
    build_dataset()
    print("\nSucesso! O dataset está pronto para o treinamento da Tiny-CNN.")