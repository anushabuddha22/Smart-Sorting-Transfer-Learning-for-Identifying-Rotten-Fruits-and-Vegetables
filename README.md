import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
# Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetables
# A deep learning project using transfer learning to classify fresh and rotten fruits and vegetables. Implemented using TensorFlow and Keras in Google Colab.
