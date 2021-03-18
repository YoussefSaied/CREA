import tensorflow as tf

from tensorflow.keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    RepeatVector,
    Reshape,
    TimeDistributed,
    RNN,
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras import activations


def encoder_decoder2(
    latent_dim1, n_outputs, n_timesteps, n_features, l1, l2,
):
    model = tf.keras.Sequential()

    model.add(
   
            GRU(
                latent_dim1,
                activation="relu",
                return_sequences=True,
                recurrent_regularizer=l1_l2(0, 0),
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
                dropout=0.0,
                recurrent_dropout=0.0,            input_shape=(n_timesteps, n_features),

            ),
        
    )
    model.add(Reshape((n_timesteps * latent_dim1 ,)))
    model.add(RepeatVector(n_outputs))
    model.add(Reshape((n_outputs, n_timesteps, latent_dim1 )))
    model.add(Dropout(0.1))
    model.add(
        TimeDistributed(
                GRU(
                    latent_dim1,
                    activation="relu",
                    # return_sequences=True,
                    kernel_regularizer=l1_l2(l1=l1, l2=l2),
                    dropout=0.0,
                    recurrent_dropout=0.0,
                )
        
        )
    )
    model.add(TimeDistributed(Dense(10, activation="relu", kernel_regularizer=l1_l2(l1=0, l2=0))))

    # model.add(
    #     Dense(
    #         10,
    #         activation="relu",
    #         kernel_regularizer=l1_l2(l1=l1, l2=l2),
    #     )
    # )
    # model.add(
    #     Dense(
    #         10,
    #         activation="relu",
    #         kernel_regularizer=l1_l2(l1=l1, l2=l2),
    #     )
    # )
    model.add(TimeDistributed(Dense(1,)))
    return model


def encoder_decoder3(
    latent_dim1, n_outputs, n_timesteps, n_features, l1, l2,
):
    model = tf.keras.Sequential()

    model.add(
        LSTM(
            latent_dim1,
            activation="relu",
            return_sequences=False,
            recurrent_regularizer=l1_l2(0, 0),
            kernel_regularizer=l1_l2(l1=l1, l2=l2),
            dropout=0.0,
            recurrent_dropout=0,
            input_shape=(n_timesteps, n_features),
        ),
    )

    model.add(Dropout(0.0))

    # model.add(
    #     Dense(
    #         10,
    #         activation="relu",
    #         kernel_regularizer=l1_l2(l1=l1, l2=l2),
    #     )
    # )
    # model.add(
    #     Dense(
    #         10,
    #         activation="relu",
    #         kernel_regularizer=l1_l2(l1=l1, l2=l2),
    #     )
    # )
    model.add(Dense(1,))
    return model


def encoder_decoder1(
    latent_dim1, n_outputs, n_timesteps, n_features, l1, l2,
):
    # encoder
    encoder_input = keras.Input(
        shape=(n_timesteps, n_features), name="time_series"
    )
    peephole_lstm_cells_encoder = tf.keras.layers.StackedRNNCells(
        [PeepholeLSTMCell(size) for size in [latent_dim1] * 2]
    )
    encoder_layer = RNN(peephole_lstm_cells_encoder, return_state=True)
    encoded = encoder_layer(encoder_input)
    encoder_output = encoded[0]
    encoder_state = encoded[1:]
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")
    encoder_output = layers.Reshape((latent_dim1, 1))(encoder_output)

    # decoder
    # decoder_input = keras.Input(
    #     shape=(latent_dim1), name="encoded_time_series"
    # )
    peephole_lstm_cells_decoder = tf.keras.layers.StackedRNNCells(
        [PeepholeLSTMCell(size) for size in [latent_dim1] * 2]
    )
    decoder_layer = RNN(peephole_lstm_cells_decoder)

    decoder_intermediate_output1 = decoder_layer(
        inputs=encoder_output, initial_state=encoder_state
    )
    decoder_intermediate_output2 = tf.keras.layers.Dropout(0.5)(
        decoder_intermediate_output1
    )
    decoder_output = layers.Dense(
        units=n_outputs,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
    )(decoder_intermediate_output2)
    # decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # encoded_series = encoder(autoencoder_input)
    # decoded_img = decoder(encoded_series)
    autoencoder = keras.Model(
        encoder_input, decoder_output, name="autoencoder"
    )
    return autoencoder


def FNN_model(
    n_outputs, n_timesteps, n_features,
):
    FNN_input = keras.Input(
        shape=(n_timesteps, n_features), name="time_series"
    )

    def fc_stack(intermediate_input, reps=3, size=100):
        origin = intermediate_input
        origin = layers.Dense(units=size, activation=None)(intermediate_input)
        for j in range(reps):
            intermediate_input = layers.Dropout(0.1)(intermediate_input)
            intermediate_input = layers.Dense(units=size)(intermediate_input)
        intermediate_output = origin + intermediate_input
        return intermediate_output

    intermediate_input = layers.Flatten()(FNN_input)
    intermediate_input = fc_stack(intermediate_input, 3, 256)
    intermediate_input = fc_stack(intermediate_input, 3, 128)
    intermediate_input = fc_stack(intermediate_input, 3, 64)

    final_output = layers.Dense(units=1)(intermediate_input)
    FNN = keras.Model(FNN_input, final_output, name="FNN")

    return FNN


# def simpleModel1(
#     n_outputs, n_timesteps, n_features,
# ):
#     model = Sequential()
#     model.add(Input(shape=(n_timesteps, n_features)))
#     model.add(Dense(1))
#     return model
