import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import np_utils

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# plt.figure()
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy', 
				optimizer='adam', 
				metrics=['accuracy'])

model.fit(np.array(x_train).reshape([-1, 28, 28, 1]), np.array(y_train), batch_size=64, epochs=20)

test_loss, test_acc = model.evaluate(np.array(x_test).reshape([-1, 28, 28, 1]), np.array(y_test))

print('Test accuracy:', test_acc)