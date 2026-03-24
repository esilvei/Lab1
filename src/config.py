import os
from pathlib import Path


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent

        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.INTERIM_DIR = self.DATA_DIR / "interim"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"

        self.RAW_AUTORIZADO_DIR = self.RAW_DIR / "1_autorizado"
        self.INTERIM_AUTORIZADO_DIR = self.INTERIM_DIR / "1_autorizado"

        self.NEGADOS_INTERIM_DIR = self.INTERIM_DIR / "0_desconhecido"
        self.NEGADOS_PROCESSED_DIR = self.PROCESSED_DIR / "0_desconhecido"

        self.IMG_SIZE = 32

        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_AUTORIZADO_DIR.mkdir(parents=True, exist_ok=True)
        self.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        self.INTERIM_AUTORIZADO_DIR.mkdir(parents=True, exist_ok=True)
        self.NEGADOS_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.NEGADOS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)