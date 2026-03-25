import mlflow
import mlflow.keras
import keras_tuner as kt
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ModelEngine:
    def __init__(self, config, model_builder):
        self.cfg = config
        self.model_builder = model_builder
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("Detector_Alunos")

    def get_generators(self):
        gen = ImageDataGenerator(rescale=1. / 255)

        train = gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'train',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32
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

        # Cálculo de pesos
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_gen.classes),
            y=train_gen.classes
        )
        cw = dict(enumerate(weights))

        tuner = kt.Hyperband(
            self.model_builder,
            objective='val_accuracy',
            max_epochs=15,
            factor=3,
            directory='tuner_logs',
            project_name='tiny_cnn_solid'
        )

        with mlflow.start_run(run_name="Final_Training_Run"):
            # 1. Busca de Hiperparâmetros
            tuner.search(train_gen, validation_data=val_gen,
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

            # 2. Constrói o melhor modelo encontrado
            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)

            # 3. Configura o Autolog
            mlflow.keras.autolog(log_models=True)

            # 4. TREINAMENTO
            print("\nIniciando o ajuste fino do modelo final...")
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=20,
                class_weight=cw,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            )

            # 5. Salvamento
            model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
            model.save(str(model_path))

        return history, model