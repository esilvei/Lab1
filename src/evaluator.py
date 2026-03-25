import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ModelEvaluator:
    def __init__(self, config):
        self.cfg = config

    def plot_training_history(self, history):
        if isinstance(history, tuple):
            history = history[0]

        stats = history.history if hasattr(history, 'history') else history

        acc = stats.get('accuracy') or stats.get('acc')
        val_acc = stats.get('val_accuracy') or stats.get('val_acc')
        loss = stats.get('loss')
        val_loss = stats.get('val_loss')

        if acc is None:
            print("Erro: Chaves de acurácia não encontradas no histórico.")
            return

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Treino')
        plt.plot(epochs_range, val_acc, label='Validação')
        plt.title('Acurácia')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Treino')
        plt.plot(epochs_range, val_loss, label='Validação')
        plt.title('Loss')
        plt.legend()
        plt.show()

    def evaluate_on_test_set(self):
        model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
        model = tf.keras.models.load_model(str(model_path))

        test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            self.cfg.PROCESSED_DIR / "test",
            target_size=(self.cfg.IMG_SIZE, self.cfg.IMG_SIZE),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=False
        )

        y_pred_prob = model.predict(test_gen)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        class_names = list(test_gen.class_indices.keys())

        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()