import tensorflow
from tensorflow import keras


def TextCnn(feature_size, embedding_size, vocab_size, classes_num, filter_num, filter_list: str, drop_out_ratio):
    x = keras.Input((feature_size,), name="chat_content")
    embeded = keras.layers.Embedding(vocab_size, embedding_size, name="embedding")(x)
    reshaped = keras.layers.Reshape((feature_size, embedding_size, 1), name="add_channel")(embeded)
    print(reshaped)
    pool_ls = []
    for filter_size in list(map(int, filter_list.split(","))):
        filter_shape = (filter_size, embedding_size)
        conv_i = keras.layers.Conv2D(filter_num, filter_shape, padding="valid", strides=(1, 1),
                                     activation=keras.activations.relu,
                                     bias_initializer=keras.initializers.constant(0.1),
                                     data_format="channels_last", name="conv_%s" % filter_size)(reshaped)
        pool_size = (feature_size - filter_size + 1, 1)
        pool_i = keras.layers.MaxPool2D(pool_size=pool_size, padding="valid", strides=(1, 1),
                                        data_format="channels_last", name="pool_%s" % filter_size)(conv_i)
        pool_ls.append(pool_i)
    output = keras.layers.Concatenate(axis=-1, name="concat")(pool_ls)
    output = keras.layers.Flatten(name="flatten")(output)
    output = keras.layers.Dropout(drop_out_ratio, name='dropout')(output)
    output = keras.layers.Dense(classes_num, activation=keras.activations.softmax,
                                bias_initializer=keras.initializers.constant(0.1),
                                name="dense")(output)
    model = keras.Model(inputs={"chat_content": x}, outputs=output)
    return model
