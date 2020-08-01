#cria modelo
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ht = cnn.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=500,
                        validation_data=test_set,
                        validation_steps=2000)

cnn.save('model')

#testa uma predição simples

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/up/up061.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'up'
else:
    prediction = 'down'
print(prediction)

#verifica acurácia e loss

from matplotlib import pyplot

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(ht.history['loss'], label='train')
pyplot.plot(ht.history['val_loss'], label='test')
pyplot.legend()

pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(ht.history['acc'], label='train')
pyplot.plot(ht.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
