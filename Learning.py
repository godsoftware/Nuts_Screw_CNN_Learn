# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:07:03 2024

@author: ozkal
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image
import shutil

# Görüntüleri yüklemek ve yeniden boyutlandırmak için fonksiyon
def load_image(file_path, size=(64, 64)):
    image = Image.open(file_path)
    image = image.resize(size)
    image = np.array(image)
    if len(image.shape) == 2:  # grayscale to RGB
        image = np.stack((image,)*3, axis=-1)
    return image

# Somun ve cıvata görüntü dosya yollarını ve etiketlerini hazırlayın
images = []
labels = []

# Somun ve cıvata klasörlerini tanımlayın
nut_dir = r'C:\python\InternshipTask\nutscrew1\cropnut'
screw_dir = r'C:\python\InternshipTask\nutscrew1\cropscrew'

# Somun ve cıvata görüntülerini yükleyin
for filename in os.listdir(nut_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(nut_dir, filename)
        images.append(load_image(file_path))
        labels.append(0)  # somun için etiket

for filename in os.listdir(screw_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(screw_dir, filename)
        images.append(load_image(file_path))
        labels.append(1)  # cıvata için etiket

# Verileri numpy dizilerine dönüştürün
images = np.array(images)
labels = np.array(labels)

# Verileri eğitim ve test olarak ayırın (80% eğitim, 20% test)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Etiketleri one-hot encoding ile dönüştürün
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# CNN Modelini Oluşturma
model = models.Sequential([
    Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2 sınıf (somun ve cıvata)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitin
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Modeli Değerlendirme
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Modeli Kullanarak Tahmin Yapma
predictions = model.predict(test_images)
print(np.argmax(predictions, axis=1))  # Tahmin edilen etiketler

# Test setindeki görüntüleri kopyalayacağınız klasör
test_output_dir = r'C:\python\InternshipTask\nutscrew1\test_set'

# Eğer klasör yoksa oluşturun
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Test setindeki görüntüleri kopyalayın
for i, test_image in enumerate(test_images):
    # Orijinal görüntü dosya adı
    if np.argmax(test_labels[i]) == 0:  # somun ise
        original_filename = os.listdir(nut_dir)[i]
        original_path = os.path.join(nut_dir, original_filename)
    else:  # cıvata ise
        original_filename = os.listdir(screw_dir)[i]
        original_path = os.path.join(screw_dir, original_filename)

    # Yeni dosya adı ve yolu
    new_filename = f'test_image_{i}_{original_filename}'
    new_path = os.path.join(test_output_dir, new_filename)

    # Görüntüyü yeni konuma kopyalayın
    shutil.copyfile(original_path, new_path)

print(f'Test setindeki görüntüler {test_output_dir} klasörüne kopyalandı.')

# Yeni test setindeki görüntüleri yükleyip tahmin yapma
new_test_dir = r'C:\python\InternshipTask\nutscrew1\newtest_set'
new_test_nut_dir = r'C:\python\InternshipTask\nutscrew1\newtest_nut'
new_test_screw_dir = r'C:\python\InternshipTask\nutscrew1\newtest_screw'

# Eğer klasörler yoksa oluşturun
if not os.path.exists(new_test_nut_dir):
    os.makedirs(new_test_nut_dir)
if not os.path.exists(new_test_screw_dir):
    os.makedirs(new_test_screw_dir)

new_test_images = []
new_test_filenames = []

# Yeni test setindeki görüntüleri yükleyin
for filename in os.listdir(new_test_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(new_test_dir, filename)
        new_test_images.append(load_image(file_path))
        new_test_filenames.append(filename)

new_test_images = np.array(new_test_images)

# Yeni test setindeki her görüntü için tahmin yapın ve uygun klasöre kopyalayın
for i, new_test_image in enumerate(new_test_images):
    prediction = model.predict(np.expand_dims(new_test_image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    if predicted_class == 0:
        print(f'Görüntü {new_test_filenames[i]} somun olarak sınıflandırıldı.')
        shutil.copyfile(os.path.join(new_test_dir, new_test_filenames[i]),
                        os.path.join(new_test_nut_dir, new_test_filenames[i]))
    else:
        print(f'Görüntü {new_test_filenames[i]} cıvata olarak sınıflandırıldı.')
        shutil.copyfile(os.path.join(new_test_dir, new_test_filenames[i]),
                        os.path.join(new_test_screw_dir, new_test_filenames[i]))

print(f'Yeni test setindeki görüntüler sınıflandırılıp ilgili klasörlere kopyalandı.')
