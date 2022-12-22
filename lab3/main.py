import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import lstm_utilits


def studying(lstm, set, steps):
    for _ in range(steps):
        lstm.fit(set, validation_data=None)


def testing(lstm, succession, length):
    iteration = 0
    output = list()
    while True:
        if iteration >= length:
            break
        else:
            output.append(succession[iteration][0])
            iteration = iteration + 1

    start = [succession[:length]]
    iteration = len(succession) - length

    for _ in tqdm(range(iteration)):
        tempory = lstm.predict(start, verbose=0)[0][0]
        finish = float(tempory)
        start[0] = start[0][1:]
        start[0].append([finish])
        output.append(finish)
    return output


if __name__ == '__main__':
    koef = 0.8

    settings = lstm_utilits.LSTM_utils(koef)

    my_lstm = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(settings.length, 1)),
        tf.keras.layers.LSTM(10, activation="linear", return_sequences=False),
        tf.keras.layers.Dense(units=1)
    ])

    my_lstm.compile(loss=tf.losses.Huber(),
                    optimizer=tf.optimizers.Adam(learning_rate=0.01),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    my_lstm.build()
    my_lstm.summary()

    studying(my_lstm, settings.train_list, 10)
    prediction = testing(my_lstm, settings.set_test, settings.length)

    plt.plot(settings.set_test_general, settings.set_test)
    plt.plot(settings.set_test_general, prediction)
    plt.show()
