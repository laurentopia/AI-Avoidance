import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D

from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
# from keras.applications import "E:/AI/test1 avoidance/test 2/mobilenet.py"
# from "E:/AI/test1 avoidance/test 2/mobilenet.py" import preprocess_input

import numpy as np
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt

DATASET_DIR = 'E:/AI/test1 avoidance/test 2'

print ('new top for mobilenet.....................')
base_model=keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha = 0.50,depth_multiplier = 1, dropout = 0.001, pooling='avg',include_top = False, weights = "imagenet")#was classes=1000
x=base_model.output
x = Dropout(0.001)(x)
x=Dense(256,activation='relu')(x) #funnel down with dense layer so the model can learn better
x = Dropout(0.001)(x)
x=Dense(64,activation='relu')(x)
x = Dropout(0.001)(x)
x=Dense(16,activation='relu')(x)
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)

# for i,layer in enumerate(model.layers):
#     print(i,layer.name)

for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[86:]:
    layer.trainable=True

print ('preparing training....................... ')
paralleled_model=model #multi_gpu_model(model, gpus=2)
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory(DATASET_DIR+"/training", target_size=(224,224), color_mode='rgb',batch_size=300, class_mode='categorical', shuffle=True)
paralleled_model.summary()
paralleled_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# checkpoint
class MyCbk(keras.callbacks.Callback):
    def __init__(self, model):
        self.model_to_save = model
    def on_epoch_end(self, epoch, logs=None):
        if epoch%100==0:
            self.model_to_save.save(DATASET_DIR+'/model_at_epoch_%d.h5' %(epoch+1))
checkpoint = MyCbk(model)
callbacks_list = [checkpoint]

# 50step/epoch, 20epoch, about 4 hours in dual-1080Ti, accuray ~60%
print('training..............')
step_size_train=train_generator.n//train_generator.batch_size/8
paralleled_model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train, epochs = 100,callbacks=callbacks_list)
print('done!! ...............')

model.save(DATASET_DIR+'/avoidance.h5',overwrite=True)

# testing
print ('testing ........')
def load_image(img_path, show=False):
    print("---" + img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor
model.load_weights(DATASET_DIR+'/avoidance.h5')
import random, os
random.seed()
random_filename = random.choice([
    x for x in os.listdir(DATASET_DIR+"/testing")
    if os.path.isfile(os.path.join(DATASET_DIR+"/testing", x)) and x.endswith('.jpg')
])
print(random_filename)
preprocessed_image = load_image(os.path.join(DATASET_DIR+"/testing", random_filename),True)
predictions = model.predict(preprocessed_image)
print(predictions)





