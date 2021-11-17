from tensorflow import keras
from tensorflow.keras import layers

img_height = 256
img_width = 256

metal_class_names =  []

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)