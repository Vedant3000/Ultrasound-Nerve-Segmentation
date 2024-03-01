import numpy as np
import skimage
from skimage.transform import resize
from scipy.ndimage import rotate
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
import tensorflow_io as tfio
import keras
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate,Dropout
from tensorflow.keras.layers import Multiply, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow
import keras
# import cv
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import segmentation_models as sm
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
focal_loss = sm.losses.cce_dice_loss
import random
import segmentation_models as sm
from segmentation_models import Unet
# sm.set_framework('tf.keras')
tf.keras.backend.set_image_data_format('channels_last')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from PIL import Image, ImageFilter
img_size=256
def preprocess_image(img_name, img_size=256):
    # Step 1: Load the image in grayscale
    img = tf.keras.preprocessing.image.load_img(img_name, color_mode="grayscale")

    # Step 2: Convert the image to a NumPy array
    in_img = tf.keras.preprocessing.image.img_to_array(img)

    # Step 3: Resize the image
    in_img = skimage.transform.resize(in_img, (img_size, img_size, 1), mode='constant', preserve_range=True, anti_aliasing=False)
    in_img = np.expand_dims(in_img, axis=0) 

    # Step 4: Normalize the pixel values to the range [0, 1]
    in_img = in_img / 255.0

    return in_img

# def preprocess_image(file_storage, img_size=256):
#     # Step 1: Load the image in grayscale
#     img = Image.open(file_storage)
#     img = img.convert("L")  # Convert to grayscale

#     # Step 2: Convert the image to a NumPy array
#     in_img = np.array(img)

#     # Step 3: Resize the image
#     in_img = resize(in_img, (img_size, img_size, 1), mode='constant', preserve_range=True, anti_aliasing=False)
#     in_img = np.expand_dims(in_img, axis=0)

#     # Step 4: Normalize the pixel values to the range [0, 1]
#     in_img = in_img / 255.0

#     return in_img




import tensorflow as tf

def Conv2D_Block(input_tensor, n_filters):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def U_Net_Generator(img_shape, n_filters=16):
    img_input = tf.keras.layers.Input(shape=img_shape)  # Input shape for noise vector
    conv1 = Conv2D_Block(img_input, n_filters * 1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(conv1)
    pool1 = tf.keras.layers.Dropout(0.01)(pool1)
    
    conv2 = Conv2D_Block(pool1, n_filters * 2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2),padding='valid')(conv2)
    pool2 = tf.keras.layers.Dropout(0.01)(pool2)
    
    conv3 = Conv2D_Block(pool2, n_filters * 4)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2),padding='valid')(conv3)
    pool3 = tf.keras.layers.Dropout(0.01)(pool3)
    
    conv4 = Conv2D_Block(pool3, n_filters * 8)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(0.01)(pool4)
    
    conv5 = Conv2D_Block(pool4, n_filters * 16)
    
    up6 = tf.keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), (2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv4])
    up6 = tf.keras.layers.Dropout(0.01)(up6)
    conv6 = Conv2D_Block(up6, n_filters * 8)
    
    up7 = tf.keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), (2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv3])
    up7 = tf.keras.layers.Dropout(0.01)(up7)
    conv7 = Conv2D_Block(up7, n_filters * 4)
    
    up8 = tf.keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), (2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([up8, conv2])
    up8 = tf.keras.layers.Dropout(0.01)(up8)
    conv8 = Conv2D_Block(up8, n_filters * 2)
    
    up9 = tf.keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), (2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([up9, conv1])
    up9 = tf.keras.layers.Dropout(0.01)(up9)
    conv9 = Conv2D_Block(up9, n_filters * 1)
    
    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv9)  # Sigmoid for binary segmentation
    
    generator = tf.keras.Model(inputs=img_input, outputs=output)
    
    return generator

# Example usage:
input_path = "train/1_1.tif"  # Replace with the path to your input image


image_shape = (img_size, img_size, 1)  # Adjust the input shape according to your dataset
model = U_Net_Generator(image_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss='binary_crossentropy', metrics=['accuracy', sm.losses.DiceLoss()])
model.load_weights('model_nerveAdam2.h5')
processed_image = preprocess_image(input_path)
predicted_mask=model.predict(processed_image)




img_arr = np.array(processed_image)
image_mask = np.array(predicted_mask)
img_arr = img_arr[0]
image_mask=image_mask[0]


fig, ax = plt.subplots(1, 3, figsize=(16, 12))

ax[0].imshow(img_arr, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(image_mask, cmap='gray')
ax[1].set_title('Mask')

ax[2].imshow(img_arr, cmap='gray', interpolation='none')
ax[2].imshow(image_mask, interpolation='none', alpha=0.7)
ax[2].set_title('Mask overlay')


# Save the figure
output_path = "results/result.png"  # Adjust the output path as needed
plt.savefig(output_path)


# plt.show()
