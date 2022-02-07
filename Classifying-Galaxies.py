# Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
import app


input_data, labels = load_galaxy_data()

# spliting data
x_train, x_test, y_train, y_test = train_test_split(input_data, labels, test_size = 0.2, random_state = 222, stratify = labels, shuffle = True)

# Data Preprocessing
data_generator = ImageDataGenerator(rescale = 1./255)
data_generator_iterator = data_generator.flow(x_train, y_train, batch_size = 5)
test_generator_iterator = data_generator.flow(x_test, y_test, batch_size = 5)

# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (128, 128, 3)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(4, activation = 'softmax'))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy()])
model.summary()

# Fit and evaluate model
model.fit(data_generator_iterator, steps_per_epoch = len(x_train) / 5, validation_data = test_generator_iterator, validation_steps = len(x_test) / 5, epochs = 8)

from visualize import visualize_activations
visualize_activations(model,test_generator_iterator)
