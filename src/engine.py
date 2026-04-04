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
        pena_invasor = hp.Choice('peso_classe_0', values=[1.5, 2.5])

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
            brightness_range=[0.6, 1.4], # Simulação de iluminação ambiente variada
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

        labels = train_gen.classes
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        base_cw = dict(enumerate(weights))

        if self.cfg.RUN_HYPERPARAMETER_SEARCH:
            hypermodel_wrapper = TinyCNNHyperModel(self.model_builder, base_cw)

            tuner = kt.RandomSearch(
                hypermodel_wrapper,
                objective=kt.Objective('val_auc', direction='max'),
                max_trials=4,
                directory=str(self.cfg.PROJECT_ROOT / 'tuner_logs'),
                project_name='tiny_cnn_search',
                overwrite=False
            )

            with mlflow.start_run(run_name="Optimized_Training_AUC"):
                stop_search = tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    mode='max',
                    patience=20 # Dobrando a paciência para que as LRs baixinhas tenham tempo de mostrar seu potencial antes de serem cortadas.
                )

                print("\n[PASSO 4.1] Iniciando busca (Objetivo: MAX val_auc para separabilidade)...")
                tuner.search(train_gen, validation_data=val_gen, epochs=200, callbacks=[stop_search])

                best_hps = tuner.get_best_hyperparameters()[0]
                model = tuner.hypermodel.build(best_hps)

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
                    epochs=200,
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
        else:
            print("\n[PASSO 4.1] Busca de Hiperparmetros desativada. Utilizando os melhores parmetros fixos (L2=0, Drop=0.3, Adam 0.001, PesoC0 1.5)...")
            best_hps = kt.HyperParameters()
            best_hps.Fixed('l2_reg', 0.0)
            best_hps.Fixed('dropout', 0.3)
            best_hps.Fixed('learning_rate', 0.001)
            best_hps.Fixed('optimizer', 'adam')
            best_hps.Fixed('peso_classe_0', 1.5)

            model = self.model_builder(best_hps)
            peso_escolhido = best_hps.get('peso_classe_0')
            cw_final = {
                0: base_cw[0] * peso_escolhido,
                1: base_cw[1]
            }

            with mlflow.start_run(run_name="Training_Top_HPs"):
                mlflow.keras.autolog(log_models=True)

                print("\n[PASSO 4.2] Iniciando treinamento do modelo direto com os hiperparmetros ttimos...")
                history = model.fit(
                    train_gen, 
                    validation_data=val_gen, 
                    epochs=200,
                    class_weight=cw_final,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_auc',
                        mode='max',
                        patience=20,
                        restore_best_weights=True
                    )]
                )

                model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
                model.save(str(model_path))
                print(f" -> Modelo final guardado em: {model_path}")

            return history, model
