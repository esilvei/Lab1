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

        precision = stats.get('precision')
        val_precision = stats.get('val_precision')

        if loss is None:
            print("Erro: Chaves não encontradas no histórico.")
            return

        epochs_range = range(len(loss))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, loss, label='Treino')
        plt.plot(epochs_range, val_loss, label='Validação')
        plt.title('Loss')
        plt.legend()

        if acc is not None:
            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, acc, label='Treino')
            plt.plot(epochs_range, val_acc, label='Validação')
            plt.title('Acurácia')
            plt.legend()

        if precision is not None:
            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, precision, label='Treino')
            plt.plot(epochs_range, val_precision, label='Validação')
            plt.title('Precisão (Foco: Segurança)')
            plt.legend()

        plt.tight_layout()

        img_path = self.cfg.PROJECT_ROOT / "reports" / "historico_treinamento.png"
        img_path.parent.mkdir(exist_ok=True)
        plt.savefig(str(img_path))
        print(f"\nGráficos de treinamento salvos em: {img_path}")
        plt.show()

    def evaluate_on_test_set(self):
        print("\nCarregando o modelo final para avaliação...")
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

        print("\nGerando predições para o conjunto de teste...")
        y_pred_prob = model.predict(test_gen)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        class_names = list(test_gen.class_indices.keys())

        print("\n--- Relatório de Classificação ---")
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')

        img_path = self.cfg.PROJECT_ROOT / "reports" / "matriz_confusao.png"
        img_path.parent.mkdir(exist_ok=True)
        plt.savefig(str(img_path))
        print(f"\nGráfico da Matriz de Confusão salvo em: {img_path}")
        plt.show()