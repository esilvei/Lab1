import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_tiny_cnn(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(32, 32, 1)))

    # Focaremos em L2 nula ou baixíssima, pois a ausência de L2 (0.0) garantiu o topo absoluto.
    hp_l2 = hp.Choice('l2_reg', values=[0.0, 1e-4])

    model.add(layers.Conv2D(
        filters=4,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(hp_l2),
        name='conv2d_hardware'
    ))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    # Reduzindo o range. Foco no Dropout vencedor e suas adjacências.
    hp_dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.4])
    model.add(layers.Dropout(rate=hp_dropout))


    model.add(layers.Dense(1, activation='sigmoid'))

    # As taxas altas dominaram os primeiros lugares absolutamente (ex: 2e-3).
    # Vamos explorar esse teto.
    # REFLEXÃO: Para dar a chance de taxas menores e mais seguras convergirem, voltamos a 
    # incluir taxas como 5e-4 e 1e-4, casando com um número maior de épocas na Engine.
    hp_lr = hp.Choice('learning_rate', values=[2e-3, 1e-3, 5e-4, 1e-4])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])

    if hp_optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=hp_lr)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=hp_lr)

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.FalsePositives(name='fp')
        ]
    )

    return model