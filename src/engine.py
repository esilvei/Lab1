import mlflow
import mlflow.keras
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TinyCNNHyperModel(kt.HyperModel):
    """
    Subclasse do HyperModel que permite não só modificar a arquitetura no método build(),
    mas também injetar hiperparâmetros dinâmicos customizados no treinamento (.fit()).
    Neste caso, ele tunará o multiplicador de punição para a Classe Zero (Falso Positivos).
    """
    def __init__(self, model_builder, base_class_weight):
        super().__init__()
        self.model_builder = model_builder
        self.base_class_weight = base_class_weight

    def build(self, hp):
        return self.model_builder(hp)

    def fit(self, hp, model, *args, **kwargs):
        # A punição dupla (2.0) e intermediária (1.5) dominou o ranking vs a proporção neutra (1.0).
        # Vamos focar essa faixa de restrição para Falsos Positivos.
        pena_invasor = hp.Choice('peso_classe_0', values=[1.5, 2.0, 2.5])
        
        # Reescreve o dicionário antes de passar pro Keras model.fit() original
        cw_tunado = {
            0: self.base_class_weight[0] * pena_invasor,
            1: self.base_class_weight[1]
        }
        kwargs['class_weight'] = cw_tunado
        
        return model.fit(*args, **kwargs)


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
        """Prepara os geradores de dados com normalizao 1/255 e augmentation."""
        train_gen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            brightness_range=[0.6, 1.4], # Simulação de iluminação ambiente do laboratório
            horizontal_flip=True
        )
        val_gen = ImageDataGenerator(rescale=1. / 255)

        train = train_gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'train',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=True
        )

        val = val_gen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'val',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=False
        )
        return train, val

    def train(self):
        from sklearn.utils import class_weight
        train_gen, val_gen = self.get_generators()

        # Calcula automaticamente os pesos base usando a proporção do dataset
        labels = train_gen.classes
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        base_cw = dict(enumerate(weights))

        # Passamos a nossa classe Especial para que os pesos sejam tunados dentro dela
        hypermodel_wrapper = TinyCNNHyperModel(self.model_builder, base_cw)

        # A rede possui pouquíssimos parâmetros. O RandomSearch avalia todo o horizonte da trial.
        tuner = kt.RandomSearch(
            hypermodel_wrapper,
            objective=kt.Objective('val_auc', direction='max'),
            max_trials=16, # Busca aleatória com profundidade suficiente do espaço de combinações
            directory=str(self.cfg.PROJECT_ROOT / 'tuner_logs'),
            project_name='tiny_cnn_search',
            overwrite=False # Mantém os dados da busca para análises futuras
        )

        with mlflow.start_run(run_name="Optimized_Training_AUC"):
            stop_search = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=20 # Dobrando a paciência para que as LRs baixinhas tenham tempo de mostrar seu potencial antes de serem cortadas.
            )

            print("\n[PASSO 4.1] Iniciando busca (Objetivo: MAX val_auc para separabilidade)...")
            # Aumentamos as épocas para 100 limitando estritamente a janela de generalização no longo prazo.
            tuner.search(train_gen, validation_data=val_gen, epochs=100, callbacks=[stop_search])

            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)

            # Imprime as informações dos parâmetros vencendores escolhidos (incluindo o peso testado)
            peso_escolhido = best_hps.get('peso_classe_0')
            print(f" -> O Tuner determinou o multiplicador de punição: {peso_escolhido}x para Desconhecidos.")
            
            cw_final = {
                0: base_cw[0] * peso_escolhido,
                1: base_cw[1]
            }

            mlflow.keras.autolog(log_models=True)

            print("\n[PASSO 4.2] Iniciando ajuste fino do modelo final usando a melhor arquitetura encontrada...")
            history = model.fit(
                train_gen, 
                validation_data=val_gen, 
                epochs=100,
                class_weight=cw_final, # Passando o peso vitorioso dinâmico para o treino final
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc', # Alteramos para early stop monitorar o AUC garantindo que grave o ápice de separabilidade
                    mode='max',
                    patience=20,
                    restore_best_weights=True
                )]
            )

            model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
            model.save(str(model_path))
            print(f" -> Modelo final guardado em: {model_path}")

        return history, model