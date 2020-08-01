#Bem vindo ao classificador!
#
#Primeiramente, é necessário carregar o modelo, os dados e a função - uma única vez
#Para isso, selecione as linhas de 11 a 40 e selecione a opção "Executar Seleção Atual";
# Para classificar uma imagem, basta colocar o caminho para ela na função testImage(/coloque aqui o caminho/), selecionar a linha e 
#"Executar Seleção Atual". No console será dito se é um thumbs up ou thumbs down
#As imagens de demonstração já estão prontas para você nas linhas 44 até 63

#carrega modelo treinado e categorias

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

cnn = keras.models.load_model('model50epoc');

#função de teste
def testImage(imagem):
    test_image = image.load_img(imagem, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'thumbs up'
    else:
        prediction = 'thumbs down'
    print(prediction)


#classifica imagens
testImage('dataset/demo_set/demo01.jpg');
testImage('dataset/demo_set/demo02.jpg');  
testImage('dataset/demo_set/demo03.jpg');  
testImage('dataset/demo_set/demo04.jpg');  
testImage('dataset/demo_set/demo05.jpg');  
testImage('dataset/demo_set/demo06.jpg');  
testImage('dataset/demo_set/demo07.jpg');  
testImage('dataset/demo_set/demo08.jpg');  
testImage('dataset/demo_set/demo09.jpg');  
testImage('dataset/demo_set/demo10.jpg');
testImage('dataset/demo_set/demo11.jpg');
testImage('dataset/demo_set/demo12.jpg');  
testImage('dataset/demo_set/demo13.jpg');  
testImage('dataset/demo_set/demo14.jpg');  
testImage('dataset/demo_set/demo15.jpg');  
testImage('dataset/demo_set/demo16.jpg');  
testImage('dataset/demo_set/demo17.jpg');  
testImage('dataset/demo_set/demo18.jpg');  
testImage('dataset/demo_set/demo19.jpg');  
testImage('dataset/demo_set/demo20.jpg');    
  
