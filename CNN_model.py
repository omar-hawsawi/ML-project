import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tensorflow.keras.models import load_model

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 
train_csv = pd.read_csv(os.path.join(DATA_DIR,"sign_mnist_train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR,"sign_mnist_test.csv"))

X_train = train_csv.drop("label", axis=1).values.reshape(-1,28,28,1)
y_train = train_csv["label"].values

X_test = test_csv.drop("label",axis=1).values.reshape(-1,28,28,1)
y_test = test_csv["label"].values

y_train_cat = to_categorical(y_train, num_classes=25)


# normlizatoin
X_train=X_train.astype("float32")/255.0
X_test=X_test.astype("float32")/255.0




X_train_split , X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_cat, test_size=0.2, random_state=42
)

flip = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True
)


flip.fit(X_train_split)
model = Sequential([
#layer1
Conv2D(32, (2,2) , activation="relu",padding = "same", input_shape=(28, 28, 1)),
BatchNormalization(),
#pooling part
Conv2D(32,(2,2), activation="relu",padding = "same"),
MaxPooling2D(pool_size = (2,2)),

#layer2
Conv2D(64,(2,2),activation="relu",padding = "same"),
BatchNormalization(),
#pooling part
Conv2D(64,(2,2), activation="relu",padding = "same"),
MaxPooling2D(pool_size = (2,2)),

#layer3
Conv2D(128,(2,2),activation="relu",padding = "same"),
BatchNormalization(),
#pooling part
Conv2D(128,(2,2), activation="relu",padding = "same"),
MaxPooling2D(pool_size = (2,2)),






Flatten(),
Dense(128, activation="relu"),
Dropout(0.65),
Dense(25, activation='softmax')


])
model.compile(tf.keras.optimizers.Adam(learning_rate=0.002), 
loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train_split, y_train_split, 
epochs=10,
batch_size=32, 
validation_data=(X_val_split, y_val_split),
verbose=1)

perdiction = model.predict(X_test)
pred_Labels = np.argmax(perdiction, axis=1)

# saving the model
save = model.save(os.path.join(DATA_DIR,"CNN folder/sign_language_model.h5"))