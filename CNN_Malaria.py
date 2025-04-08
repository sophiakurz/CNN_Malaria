import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
image_dir = "/jet/home/skurz/HW2/dataset/train_images"

image_paths = []
labels = []
train_csv = pd.read_csv("/jet/home/skurz/HW2/dataset/train_data.csv")

for img_name, label in zip(train_csv['img_name'], train_csv['label']):
    img_path = os.path.join(image_dir, img_name)
    if os.path.exists(img_path):
        image_paths.append(img_path)
        labels.append(label)
image_paths = np.array(image_paths)
labels = np.array(labels)
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
def load_images(paths, labels, datagen):
    images = []
    for path in paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = datagen.standardize(img)
        images.append(img)
    return np.array(images), np.array(labels)
X_train, y_train = load_images(train_paths, train_labels, train_datagen)
X_val, y_val = load_images(val_paths, val_labels, val_datagen)

X_train.shape, X_val.shape, y_train.shape, y_val.shape

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)  

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.2f}")
val_predictions = model.predict(X_val)
val_pred_labels = (val_predictions > 0.5).astype(int).flatten()
f1 = f1_score(y_val, val_pred_labels)
print("F1 Score on Validation set:", f1)
test_dir = "/jet/home/skurz/HW2/dataset/test_images"
test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith('.png')]

X_test = []
test_filenames = []

for img_path in test_images:
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    X_test.append(img)
    test_filenames.append(os.path.basename(img_path))

X_test = np.array(X_test)

predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int).flatten()
submission_df = pd.DataFrame({'img_name': test_filenames, 'label': predicted_labels})
submission_df.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")