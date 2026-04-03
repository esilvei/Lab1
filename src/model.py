import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_tiny_cnn(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(32, 32, 1)))

    hp_l2 = hp.Choice('l2_reg', values=[1e-3, 1e-4])

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

    hp_dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
    model.add(layers.Dropout(rate=hp_dropout))

    model.add(layers.Dense(1, activation='sigmoid'))

    hp_lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model