import os
import random
import kagglehub
import shutil
from src.config import Config
from src.data_utils import DataExtractor
from src.preprocessor import ImageProcessor, DataPreprocessor
from src.dataset_manager import DatasetManager
from src.engine import ModelEngine
from src.evaluator import ModelEvaluator
from src.model import build_tiny_cnn
from src.export_mif import export_model_to_mif

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'




def main():
    cfg = Config()
    extractor = DataExtractor()
    img_processor = ImageProcessor(cfg.IMG_SIZE)
    data_preprocessor = DataPreprocessor(cfg, img_processor, extractor)
    ds_manager = DatasetManager(cfg)
    evaluator = ModelEvaluator(cfg)
    random.seed(42)

    print("\n" + "=" * 50)
    print("         INICIANDO PIPELINE BIOMÉTRICA (ARTEFATO 1)      ")
    print("=" * 50)

    # 1. GESTÃO DE DATASETS EXTERNOS (Classe 0)
    print("\n[PASSO 1] Gerenciando fontes de Desconhecidos...")

    # Extração das Selfies locais (.tar.gz)
    tar_selfies = cfg.RAW_DIR / "Selfie-dataset.tar.gz"
    if tar_selfies.exists():
        extractor.extract_tar(tar_selfies, cfg.RAW_DIR / "selfies", limit=3000)

    lfw_download_path = kagglehub.dataset_download("atulanandjha/lfwpeople")
    lfw_raw_folder = cfg.RAW_DIR / "lfw_extracted"

    if not lfw_raw_folder.exists():
        print("  -> Extraindo LFW para a estrutura do projeto...")
        shutil.copytree(lfw_download_path, lfw_raw_folder)

    # 2. PRÉ-PROCESSAMENTO (Raw -> Interim)
    data_preprocessor.clear_interim()

    # Processa fotos/vídeos da equipe (Classe 1)
    data_preprocessor.process_authorized(max_fotos=800)

    # Processa minerando faces do LFW e Selfies (Classe 0)
    data_preprocessor.process_unknowns(ratio=1.5)

    # 3. ORGANIZA‡ƒO DO DATASET (Interim -> Processed)
    print("\n[PASSO 3] Organizando Dataset (Split Treino/Validao/Teste)...")
    ds_manager.clean_processed()
    ds_manager.split_data(list(cfg.NEGADOS_INTERIM_DIR.glob("*.jpg")), "0_desconhecido")
    ds_manager.split_data(list(cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")), "1_autorizado")

    # 4. TREINAMENTO E TUNING
    print("\n[PASSO 4] Iniciando Treinamento com Keras Tuner...")
    engine = ModelEngine(cfg, build_tiny_cnn)
    history, model = engine.train()

    # 5. AVALIAÇÃO E EXPORTAÇÃO
    print("\n[PASSO 5] Gerando relatórios e arquivos para FPGA...")
    evaluator.plot_training_history(history)
    evaluator.evaluate_on_test_set()

    # Exportação para arquivos .mif quantizados em Q1.7
    export_model_to_mif()

    print("\n" + "=" * 50)
    print("      PIPELINE CONCLUÍDA COM SUCESSO (ARTEFATO 1)      ")
    print("=" * 50)


if __name__ == "__main__":
    main()