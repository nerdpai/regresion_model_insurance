import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def getDataAndLables(dataFrame: pd.DataFrame, name: str = "target"):
    numericColumns = (
        dataFrame.drop(name, axis=1)
        .select_dtypes(include=[int, float])
        .columns.to_list()
    )
    nonNumericColumns = (
        dataFrame.drop(name, axis=1)
        .select_dtypes(exclude=[int, float])
        .columns.to_list()
    )

    columnTransformer = make_column_transformer(
        (MinMaxScaler(), numericColumns),
        (OneHotEncoder(handle_unknown="ignore"), nonNumericColumns),
    )

    x = dataFrame.drop(name, axis=1)
    y = dataFrame[name]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    columnTransformer.fit(x_train)

    x_trainNormal = columnTransformer.transform(x_train)
    x_testNormal = columnTransformer.transform(x_test)

    x_trainNormal = tf.convert_to_tensor(x_trainNormal)
    x_trainNormal = tf.cast(x_trainNormal, tf.float32)

    x_testNormal = tf.convert_to_tensor(x_testNormal)
    x_testNormal = tf.cast(x_testNormal, tf.float32)

    return (x_trainNormal, x_testNormal, y_train, y_test)


def main():
    insurance = pd.read_csv(
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    )

    training_x, test_x, training_y, test_y = getDataAndLables(insurance, "charges")

    activation = tf.keras.activations.gelu

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(100, activation=activation),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(80, activation=activation),
            tf.keras.layers.Dense(40, activation=activation),
            tf.keras.layers.Dense(10, activation=activation),
            tf.keras.layers.Dense(1),
        ]
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss=tf.keras.losses.huber,
        metrics=["mae"],
    )

    model.fit(
        batch_size=50,
        x=training_x,
        y=training_y,
        verbose=1,
        epochs=500,
        callbacks=callback,
    )

    prediction = model.predict(test_x)
    prediction = np.array(prediction).squeeze()

    plt.style.use("dark_background")

    x = np.arange(len(test_y))
    width = 0.2

    plt.bar(x - 0.2, test_y, width)
    plt.bar(x, prediction, width)

    plt.legend(
        [
            "Test",
            "Prediction",
        ]
    )
    print(tf.keras.losses.mae(test_y, prediction))
    plt.show()


if __name__ == "__main__":
    main()
