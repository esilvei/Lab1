import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelEvaluator:
    def __init__(self, config):
        self.cfg = config

    def plot_training_history(self, history):
        if isinstance(history, tuple):
            history = history[0]

        stats = history.history if hasattr(history, 'history') else history

        loss = stats.get('loss')
        val_loss = stats.get('val_loss')

        train_auc = stats.get('auc')
        val_auc = stats.get('val_auc')

        if loss is None:
            print("Erro: Chaves não encontradas no histórico.")
            return

        epochs_range = range(len(loss))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='Treino')
        plt.plot(epochs_range, val_loss, label='Validação')
        plt.title('Loss')
        plt.legend()

        if train_auc is not None:
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, train_auc, label='Treino')
            plt.plot(epochs_range, val_auc, label='Validação')
            plt.title('AUC (Foco: Separabilidade)')
            plt.legend()

        plt.tight_layout()

        img_path = self.cfg.PROJECT_ROOT / "reports" / "historico_treinamento.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(img_path))
        print(f"\nGráficos de treinamento salvos em: {img_path}")
        plt.close()

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
        y_pred_prob = model.predict(test_gen).flatten()
        y_true = test_gen.classes
        class_names = list(test_gen.class_indices.keys())

        # --- Análise ROC e AUC ---
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Buscando o Threshold ideal para segurança (FPR próximo de 0)
        # Vamos buscar o limiar onde o FPR seja baixíssimo (ex: <= 1%) 
        # Isso garante que um professor não autorizado raramente seja aprovado
        target_fpr = 0.01
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[-1]
        else:
            best_idx = np.argmax(tpr - fpr) # Fallback alternativo
        
        optimal_threshold = thresholds[best_idx]
        
        print(f"\n--- Análise Limiar Otimizado ---")
        print(f"Test AUC: {roc_auc:.4f}")
        print(f"Limiar (Threshold) Seguro Sugerido para a Fechadura: {optimal_threshold:.4f}")
        print(f"Com esse limiar teremos ~{(fpr[best_idx]*100):.1f}% Falsos Positivos e {tpr[best_idx]*100:.1f}% de Recall Genuíno.")

        # Plot ROC Curve
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', s=100, label=f'Limiar {optimal_threshold:.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        img_path_roc = self.cfg.PROJECT_ROOT / "reports" / "curva_roc.png"
        plt.savefig(str(img_path_roc))
        plt.close()
        print(f"Gráfico da Curva ROC salvo em: {img_path_roc}")

        # Avaliação final usando o novo threshold seguro
        y_pred = (y_pred_prob > optimal_threshold).astype(int)

        print(f"\n--- Relatório de Classificação (Limiar de {optimal_threshold:.2f}) ---")
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão (Limiar Seguro: {optimal_threshold:.2f})')

        img_path_cm = self.cfg.PROJECT_ROOT / "reports" / "matriz_confusao.png"
        plt.savefig(str(img_path_cm))
        plt.close()
        print(f"Gráficos salvos com sucesso na pasta reports/!")
