import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GaussianNoise
from tensorflow.keras.callbacks import TensorBoard
import dataset
import numpy as np
import time

input_shape=(28, 28, 1)
size = 28 * 28
num_class = 26

model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=input_shape, padding='same'),
    # MaxPooling2D(),
    # GaussianNoise(.2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    # Conv2D(128, (3, 3), activation='relu'),
    # MaxPooling2D(),
    # Conv2D(256, (3, 3), activation='relu'),
    # MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_class, activation='softmax'),
])

adam = keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print("\nNetwork:")
print(model.summary())

print("\nLoading data...")
x_train, y_train, x_test, y_test = dataset.load_data()
# x_train, y_train = dataset.create_data(50000)
# x_test, y_test = dataset.create_data(10000)

print("\nData example:")
print(x_train[0])

x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32')
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32')

x_train /= 255.0
x_test /= 255.0

print(x_train.shape, x_train.ndim, x_train.dtype)
print(y_train.shape, y_train.ndim, y_train.dtype)

y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)

print("\nStart training...")
# tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()))
# model.fit(x_train, y_train, batch_size=200, epochs=3, validation_data=(x_test, y_test), callbacks=[tensorboard])
model.fit(x_train, y_train, batch_size=200, epochs=15, validation_data=(x_test, y_test), verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save weights to a HDF5 file
# model.save_weights('my_model.h5', foramt='dhf5')
model.save('letter.h5')