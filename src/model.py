import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_tiny_cnn(hp):
    model = keras.Sequential()

    # ENTRADA: 32x32 pixels, 1 canal de cor (Grayscale)
    model.add(layers.Input(shape=(32, 32, 1)))

    # CONVOLUÇÃO: 4 filtros 3x3
    model.add(layers.Conv2D(
        filters=4,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        name='conv2d_hardware'
    ))

    # POOLING: Reduz a imagem pela metade (Max Pooling 2x2)
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    # Keras Tuner: Busca a melhor quantidade de neurônios na camada oculta
    hp_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(layers.Dense(units=hp_units, activation='relu'))

    # SAÍDA BINÁRIA: 1 neurônio (0 = Invasor/Fundo, 1 = Autorizado)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Keras Tuner: Busca a melhor taxa de aprendizado
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model