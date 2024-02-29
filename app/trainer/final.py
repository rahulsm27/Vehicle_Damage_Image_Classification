#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing libraries
# !pip3 install split-folders
# !pip3 install keras-efficientnet
# !pip3 install -U efficientnet
#!pip3 install --upgrade tensorflow


# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np

import os
import shutil

# import cv2
# from PIL import Image

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# from IPython.display import display

colab = False
kaggle = False

# In[ ]:


# import shutil

# shutil.rmtree('/kaggle/working/dataset_final')

# shutil.rmtree('/kaggle/working/dataset')


# In[ ]:


if colab :
  from google.colab import drive

  # Mount Google Drive
  drive.mount('/content/drive')


# In[ ]:


if colab:
    train_csv = pd.read_csv("/content/drive/MyDrive/hackathon/train/train.csv")
    test_csv =  pd.read_csv("/content/drive/MyDrive/hackathon/test/test.csv")
elif kaggle:
    train_csv = pd.read_csv("/kaggle/input/dataset/train/train.csv")
    test_csv =  pd.read_csv("/kaggle/input/dataset/test/test.csv")
else:
    train_csv = pd.read_csv("/gcs/vertex_r/train/train.csv")
    test_csv =  pd.read_csv("/gcs/vertex_r/test/test.csv")

labels = {1:'crack',2:'scratch',3:'tire flat',4 :'dent', 5: 'glass shatter', 6: 'lamp broken'}


# In[ ]:


# Define image folder path
if colab:
    path_to_train_images = "/content/drive/MyDrive/hackathon/train/images/"
    path_to_test_images = "/content/drive/MyDrive/hackathon/test/images/"
elif kaggle:
    path_to_train_images = "/kaggle/input/dataset/train/images/"
    path_to_test_images = "/kaggle/input/dataset/test/images/"
else:
    path_to_train_images = "/gcs/vertex_r/train/images/"
    path_to_test_images = "/gcs/vertex_r/test/images/"

# Listing all images
train_image_files = os.listdir(path_to_train_images)
#test_image_files = os.listdir(path_to_test_images)

train_image_files = [t for t in train_image_files if t.endswith(".jpg")]
train_image_files=[t for t in train_image_files if '(' not in t]


# In[ ]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Assuming y_train is your target variable
class_labels = np.unique(np.array(train_csv['label']))

# Calculate class weights
class_weights = compute_class_weight('balanced',classes = np.unique(train_csv['label']),y=train_csv['label'])

# Create a dictionary of class weights
class_weights_dict = dict(zip(class_labels, class_weights))

print(class_weights_dict)


# In[ ]:


labels = {1:'crack',2:'scratch',3:'tire flat',4 :'dent', 5: 'glass shatter', 6: 'lamp broken'}
folder = '/gcs/vertex_r/dataset/train'
train_image_files=[t for t in train_image_files if '(' not in t]

#making folder for each classes
for values in labels:
    path = os.path.join(folder,str(values))
    os.makedirs(path,exist_ok=True)

# coping our raw data to each classes

for image in train_image_files:
 #   print(image)
    label = int((train_csv[train_csv['filename'] == image]['label']).iloc[0])
  #  print(label)
  #  path = os.path.join(folder,label,'/')
    from_path = os.path.join(path_to_train_images,image)
    to_path = os.path.join(folder,str(label))
    if not os.path.exists(os.path.join(to_path,image)):
      shutil.copy(from_path,to_path)


# In[ ]:


import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("/gcs/vertex_r/dataset/train", output="/gcs/vertex_r/dataset_final",
    seed=1337, ratio=(.8, .2, ), group_prefix=None, move=True) # default values


# In[ ]:


from tensorflow.keras.applications import EfficientNetB7


# In[ ]:


import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

import keras
from keras import layers


IMG_SIZE = 600
batch = 32


# In[ ]:


# Add our data-augmentation parameters to ImageDataGenerator

train_dir = '/gcs/vertex_r/dataset_final/train'
validation_dir = '/gcs/vertex_r/dataset_final/val'


train_datagen = ImageDataGenerator()#rotation_range = 40, zoom_range = 0.2, horizontal_flip = True)

valid_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(train_dir, batch_size = batch, class_mode = 'categorical', target_size = (IMG_SIZE , IMG_SIZE ))

validation_generator = valid_datagen.flow_from_directory( validation_dir, batch_size = batch, class_mode = 'categorical', target_size = (IMG_SIZE, IMG_SIZE))



# In[ ]:


class_weights_final = {}
for i in range(1,7):
    #pass
    class_weights_final[train_generator.class_indices[str(i)]] = class_weights_dict[i]

print(class_weights_final)
print(train_generator.class_indices)
print(class_weights_dict)


# In[ ]:


import keras
from keras import layers


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# In[ ]:


NUM_CLASSES = 6
model = build_model(num_classes=NUM_CLASSES)



# In[ ]:


def unfreeze_model(model):
    # We unfreeze the top 5 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy","categorical_accuracy"]
    )


unfreeze_model(model)



# In[ ]:


epochs = 30
#hist = model.fit(train_generator, epochs=epochs, validation_data=validation_generator,callbacks=[model_checkpoint_callback])
hist = model.fit(train_generator, epochs=epochs, validation_data=validation_generator,class_weight = class_weights_final)#,callbacks=[model_checkpoint_callback])


# In[ ]:


model.save('my_model5.keras')


# In[ ]:


if colab:
    test_dir = '/content/drive/MyDrive/hackathon/test'
elif kaggle:
    test_dir = "/kaggle/input/dataset/test/"
else:
    test_dir = '/gcs/vertex_r/test/'
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory( test_dir, batch_size = 20, class_mode = None, target_size = (IMG_SIZE, IMG_SIZE),shuffle= False)

test_generator.reset()


# In[ ]:


predictions = model.predict(test_generator)

predicted_class_indices=np.argmax(predictions,axis=1)
labels_map = (train_generator.class_indices)
labels_map = dict((v,k) for k,v in labels_map.items())
predictions = [labels_map[i] for i in predicted_class_indices]


# In[ ]:


# Preparing submitssion file

filenames=test_generator.filenames
filenames = [f[7:-4] for f in filenames]
results=pd.DataFrame({"image_id":filenames,
                      "label":predictions})
results.to_csv("sub.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## SUBMITING OUR PREDICTIONS

# In[ ]:





# In[ ]:




