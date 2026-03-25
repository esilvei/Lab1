import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import mlflow
import mlflow.keras
import keras_tuner as kt
from src.config import Config
from src.model import build_tiny_cnn

cfg = Config()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Fechadura_Biometrica_V3_75_25")


def load_data():
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_gen = datagen.flow_from_directory(
        cfg.PROCESSED_DIR / "train",
        target_size=(32, 32),
        color_mode="grayscale",
        class_mode="binary",
        batch_size=32,
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        cfg.PROCESSED_DIR / "val",
        target_size=(32, 32),
        color_mode="grayscale",
        class_mode="binary",
        batch_size=32,
        shuffle=False
    )
    return train_gen, val_gen


def run_training():
    train_data, val_data = load_data()

    # --- CÁLCULO DE PESOS (COMPENSA O 75/25) ---
    labels = train_data.classes
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    cw = dict(enumerate(weights))

    print(f"\nPesos aplicados para equilibrar o 75/25: {cw}")

    tuner = kt.Hyperband(
        build_tiny_cnn,
        objective='val_accuracy',
        max_epochs=15,
        factor=3,
        directory=str(cfg.PROJECT_ROOT / 'tuner_logs'),
        project_name='tiny_cnn_v3'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    with mlflow.start_run(run_name="Weighted_Training_3to1"):
        print("Buscando hiperparâmetros...")
        tuner.search(train_data, validation_data=val_data, callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)

        mlflow.log_params(best_hps.values)
        mlflow.keras.autolog()

        print("\nTreinando modelo final...")
        model.fit(
            train_data,
            epochs=30,
            validation_data=val_data,
            class_weight=cw,
            callbacks=[stop_early]
        )

        model_path = cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
        model.save(str(model_path))
        print(f"\nSucesso! Modelo salvo em: {model_path}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    run_training()