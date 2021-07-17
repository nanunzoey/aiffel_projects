import os
import glob
import random from PIL import Image
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt


# 이미지 크기 28*28로 변경
def resize_images(img_path):
    images = glob.glob(img_path + "/*.jpg")
    print(len(images)
    target_size=(28, 28)

    for img in images:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size, Image.ANTIALIAS)
        new_img.save(img, "JPEG")

        print(len(images)


    image_dir_path=os.getenv("HOME") + "/aiffel/rock_scissor_paper"

    resize_images(image_dir_path + "/rock")
    resize_images(image_dir_path + "/scissor")
    resize_images(image_dir_path + "/paper")


    def load_data(img_path, number_of_data=1200):
        img_size=28
        color=3

        imgs=np.zeros(number_of_data*img_size*img_size*color,
                      dtype=np.int32).reshape(number_of_data, img_size, img_size, color)
        labels=np.zeros(number_of_data, dtype=np.int32)

        idx=0
        for file in glob.iglob(img_path + '/scissor/*.jpg'):
            img=np.array(Image.open(file), dtype=np.int32)
            imgs[idx, :, :, :]=img
            labels[idx]=0
            idx += 1

        for file in glob.iglob(img_path + '/rock/*.jpg'):
           img=np.array(Image.open(file), dtype=np.int32)
            imgs[idx, :, :, :]=img
            labels[idx]=1
            idx += 1

        for file in glob.iglob(img_path + '/paper/*.jpg'):
            img=np.array(Image.open(file), dtype=np.int32)
            imgs[idx, :, :, :]=img
            labels[idx]=2
            idx += 1

        print(\"이미지 개수: \", idx)
       return imgs, labels



image_dir_path=os.getenv("HOME") + "/aiffel/rock_scissor_paper"
    (x_train, y_train)=load_data(image_dir_path)
    x_train_norm=x_train / 255.0
    print(x_train_norm.shape, y_train.shape)



model=keras.models.Sequential()
model.add(keras.layers.Conv2D(
    16, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()


plot_model(model, show_shapes=True, dpi=70)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_norm, y_train, epochs=30)


# test
test_path=os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"

resize_images(test_path + "/rock")
resize_images(test_path + "/scissor")
resize_images(test_path + "/paper")

(x_test, y_test)=load_data(test_path)
x_test_norm=x_test / 255.0

loss: 1.0422 - accuracy: 0.8367

test_loss, test_accuracy=model.evaluate(x_test_norm, y_test, verbose=2)


predicted_result=model.predict(x_test_norm)
predicted_labels=np.argmax(predicted_result, axis=1)


wrong_predict_list=[]

for i, _ in enumerate(predicted_labels):
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

samples=random.choices(population=wrong_predict_list, k=5)

for n in samples:
    print("예측 확률 븐포: ", str(predicted_result[n]))
    print("라벨: ", str(y_test[n]))
    print("예측 결과: ", str(predicted_labels[n]))

    plt.imshow(x_test_norm[n])
    plt.show()
