import os
from pathlib import Path

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init_paths()
        return cls._instance

    def _init_paths(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.INTERIM_DIR = self.DATA_DIR / "interim"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.LFW_TAR = self.RAW_DIR / "lfw.tgz"
        self.LFW_RAW_DIR = self.RAW_DIR / "lfw_extracted"
        self.RAW_AUTORIZADO_DIR = self.RAW_DIR / "1_autorizado"
        self.INTERIM_AUTORIZADO_DIR = self.INTERIM_DIR / "1_autorizado"
        self.NEGADOS_INTERIM_DIR = self.INTERIM_DIR / "0_desconhecido"

        self.TUNER_LOGS_DIR = self.PROJECT_ROOT / "tuner_logs"

        self.IMG_SIZE = 32
        self.CHANNELS = 1
        
        # Flag para controlar se fazemos busca de hiperparâmetros ou se treinamos direto com os melhores conhecidos
        self.RUN_HYPERPARAMETER_SEARCH = False

    def setup_directories(self):
        dirs = [
            self.RAW_DIR, self.INTERIM_DIR, self.PROCESSED_DIR,
            self.RAW_AUTORIZADO_DIR, self.INTERIM_AUTORIZADO_DIR, self.NEGADOS_INTERIM_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)