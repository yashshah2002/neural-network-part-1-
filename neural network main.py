from functools import partial
import keras
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import callbacks
from tensorflow.python.keras.layers.core import Dense

# Loading data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Reshaping data-Adding number of channels as 1 (Grayscale images)
train_images = train_images.reshape((train_images.shape[0],
                                     train_images.shape[1],
                                     train_images.shape[2], 1))

test_images = test_images.reshape((test_images.shape[0],
                                   test_images.shape[1],
                                   test_images.shape[2], 1))

# scaling down pixel values
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

# Encoding lables to a binary class martrix
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu",
                        input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary
model.compile(optimizer ="rmsprop", loss ="categorical_crossentropy",metrics =['accuracy'])
val_images = train_images[:10000]
partial_images = train_images[10000:]
val_labels = y_train[:10000]
partial_labels = y_train[10000:]
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                       mode ="min", patience = 5,
                                       restore_best_weights = True)
history = model.fit(partial_images, partial_labels, batch_size = 128,
                    epochs = 25, validation_data =(val_images, val_labels),
                    callbacks =[earlystopping]) 
