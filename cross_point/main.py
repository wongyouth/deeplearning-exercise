import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GaussianNoise
from tensorflow.keras.callbacks import TensorBoard
import seed
import numpy as np
import time

input_shape=(28, 28, 1)
size = 28 * 28

model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=input_shape, padding='same'),
    MaxPooling2D(),
    GaussianNoise(.2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    # Conv2D(128, (3, 3), activation='relu'),
    # MaxPooling2D(),
    # Conv2D(256, (3, 3), activation='relu'),
    # MaxPooling2D(),
    # Dropout(0.2),
    Flatten(),
    Dense(size, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())


x_train, y_train, x_test, y_test = seed.load_data('data.npz')
# x_train, y_train = seed.create_data(5000)
# x_test, y_test = seed.create_data(1000)

x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32')
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32')

x_train /= 255.0
x_test /= 255.0

y_train = keras.utils.to_categorical(y_train, size)
y_test = keras.utils.to_categorical(y_test, size)

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()))
model.fit(x_train, y_train, batch_size=200, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard])

# Save weights to a HDF5 file
# model.save_weights('my_model.h5', foramt='dhf5')
model.save('my_model.h5')