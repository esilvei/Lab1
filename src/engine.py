import mlflow
import mlflow.keras
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelEngine:
    """
    Motor de treino otimizado para a Tiny-CNN.
    Implementa pesos manuais para reduzir Falsos Positivos e busca exaustiva de hiperparâmetros.
    """
    def __init__(self, config, model_builder):
        self.cfg = config
        self.model_builder = model_builder
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("Fechadura_Biometrica_Otimizada")

    def get_generators(self):
        """Prepara os geradores de dados com normalização 1/255."""
        gen = ImageDataGenerator(rescale=1. / 255)

        train = gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'train',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=True
        )

        val = gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'val',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=False
        )
        return train, val

    def train(self):
        train_gen, val_gen = self.get_generators()

        cw = {0: 1.0, 1: 4.0}

        tuner = kt.Hyperband(
            self.model_builder,
            objective=kt.Objective('val_precision', direction='max'),
            max_epochs=20,
            factor=3,
            directory='tuner_logs',
            project_name='tiny_cnn_fine_tuning'
        )

        with mlflow.start_run(run_name="Optimized_Training_Precision"):
            stop_search = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            )

            print("\n[PASSO 4.1] Iniciando busca (Objetivo: MAX val_precision)...")
            tuner.search(train_gen, validation_data=val_gen, callbacks=[stop_search])

            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)

            mlflow.keras.autolog(log_models=True)

            print("\n[PASSO 4.2] Iniciando ajuste fino do modelo final...")
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=50,
                class_weight=cw,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', # Loss geral ajuda a evitar overfitting
                    patience=15,
                    restore_best_weights=True
                )]
            )

            model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
            model.save(str(model_path))
            print(f" -> Modelo final guardado em: {model_path}")

        return history, model