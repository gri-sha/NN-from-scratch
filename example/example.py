import tensorflow as tf
from tensorflow import keras
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model_path = "my_model.h5"
retrain = True

if os.path.exists(model_path) and not retrain:
    model = keras.models.load_model(model_path)
    print("Loaded existing model.")
else:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=8)
    model.save(model_path)
    print("Trained and saved new model.")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nAccuracy: {test_acc:.3%}")
