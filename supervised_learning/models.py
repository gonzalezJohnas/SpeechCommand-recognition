import tensorflow as tf
from tensorflow.python.keras import layers, Model
from tensorflow.keras.backend import squeeze

def get_model(name, output_dim, features_dim):
    if name == "cnn":
        return conv_net(output_dim, features_dim)
    elif name == "lstm":
        return conv_lstm_net(output_dim, features_dim)
    elif name == "attention_lstm":
        return conv_net_lstm_attention(output_dim, features_dim)
    else:
        return None
def simple_lstm(output_dim=11):
    model = tf.keras.Sequential()

    # Add a LSTM layer with 128 internal units.

    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True),
                                   input_shape=(32, 26)))

    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.7)))

    model.add(layers.Dense(80, activation='relu'))

    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dropout(rate=1 - 0.7))

    # Add a Dense layer with 10 units and softmax activation.
    model.add(layers.Dense(output_dim, activation='softmax'))

    return model


def conv_net_lstm_attention(output_dim=11, mel_dim=(32, 26, 1)):
    inputs = layers.Input((mel_dim))

    x = layers.Conv2D(filters=10, kernel_size=(5, 1), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=40, kernel_size=(5, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=1, kernel_size=(5, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)


    x = layers.Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (x)

    x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)

    xFirst = layers.Lambda(lambda q: q[:, 16])(x)  # [b_s, vec_dim]
    query = layers.Dense(128)(xFirst)

    # dot product attention
    attScores = layers.Dot(axes=[1, 2])([query, x])
    attScores = layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = layers.Dense(64, activation='relu')(attVector)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32)(x)
    output = layers.Dense(output_dim, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model


def conv_net(output_dim=11, mel_dim=(32, 26, 1)):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding='same', input_shape=mel_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Dropout(rate=1 - 0.7))

    model.add(layers.Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=1 - 0.7))

    model.add(layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=1 - 0.7))

    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dense(20, activation='relu'))

    model.add(layers.Dropout(rate=1 - 0.7))

    # Add a Dense layer with 10 units and softmax activation.
    model.add(layers.Dense(output_dim, activation='softmax'))

    return model


def conv_lstm_net(output_dim=11, mel_dim=(32, 26, 1)):
    model = tf.keras.Sequential()

    # Add a LSTM layer with 128 internal units.
    model.add(layers.Conv2D(filters=50, kernel_size=(5, 1), activation='relu', padding='same', input_shape=mel_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Dropout(rate=1 - 0.7))

    model.add(layers.Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=1 - 0.7))

    model.add(layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=1 - 0.7))

    model.add(layers.Reshape((-1, 80)))

    model.add(layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True)))

    model.add(layers.Bidirectional(layers.CuDNNLSTM(32)))

    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dense(20, activation='relu'))

    model.add(layers.Dropout(rate=1 - 0.7))

    # Add a Dense layer with 10 units and softmax activation.
    model.add(layers.Dense(output_dim, activation='softmax'))

    return model
