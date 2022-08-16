
import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.datasets as datasets
import tensorflow.keras as keras

keras.backend.set_floatx('float64')


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = np.expand_dims(x_train, -1) / np.float32(255.0)
y_train = np.expand_dims(y_train, -1)
x_test = np.expand_dims(x_test, -1) / np.float32(255.0)
y_test = np.expand_dims(y_test, -1)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

np.random.seed(0)
weights = []
for weight in model.get_weights():
    weights.append(np.random.randn(*weight.shape)/np.prod(weight.shape))
model.set_weights(weights)

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 256

model.fit(x_train, y_train, shuffle=False, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10, callbacks=[callbacks.ModelCheckpoint('test_no_optimizer.h5', save_best_only=True, monitor='val_accuracy')])

weights_with_fit = model.get_weights()

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.set_weights(weights)

model.compile(optimizer=BatchOptimizer(optimizers.RMSprop(learning_rate=0.001), 16, 256, x_train.shape[0], model), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, shuffle=False, validation_data=(x_test, y_test), batch_size=16, epochs=10, callbacks=[callbacks.ModelCheckpoint('test_optimizer.h5', save_best_only=True, monitor='val_accuracy')])

weights_with_batch_model = model.get_weights()

for index in range(len(weights_with_fit)):
    print(np.mean(np.abs(weights_with_fit[index] - weights_with_batch_model[index])))
