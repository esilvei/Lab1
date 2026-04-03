import mlflow
import mlflow.keras
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelEngine:
    """
    Motor de treino otimizado para a Tiny-CNN.
    Implementa pesos manuais e busca de hiperparâmetros focada no RECALL (detecção de autorizados).
    """
    def __init__(self, config, model_builder):
        self.cfg = config
        self.model_builder = model_builder
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("Fechadura_Biometrica_Otimizada")

    def get_generators(self):
        """Prepara os geradores de dados com augmentation dinâmico para iluminação."""

        # OTIMIZAÇÃO: Augmentation em tempo real apenas para o TREINO
        # brightness_range: 0.5 (muito escuro) até 1.5 (muito claro/estourado)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            brightness_range=[0.6, 1.4],
            rotation_range=10,
            horizontal_flip=True
        )

        # Validação NUNCA deve ter augmentation, apenas rescale
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train = train_datagen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'train',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=True
        )

        val = val_datagen.flow_from_directory(
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

        # Compute class weights dynamically based on actual sample counts
        n_train = train_gen.n
        class_counts = np.bincount(train_gen.classes)
        cw = {i: n_train / (len(class_counts) * c) for i, c in enumerate(class_counts) if c > 0}

        tuner = kt.Hyperband(
            self.model_builder,
            objective=kt.Objective('val_auc', direction='max'),
            max_epochs=20,
            factor=3,
            directory='tuner_logs',
            project_name='tiny_cnn_fine_tuning'
        )

        with mlflow.start_run(run_name="Optimized_Training_AUC"):
            stop_search = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=10
            )

            print("\n[PASSO 4.1] Iniciando busca (Objetivo: MAX val_auc)...")
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
                    monitor='val_auc',
                    mode='max',
                    patience=15,
                    restore_best_weights=True
                )]
            )

            model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
            model.save(str(model_path))
            print(f" -> Modelo final guardado em: {model_path}")

        return history, model