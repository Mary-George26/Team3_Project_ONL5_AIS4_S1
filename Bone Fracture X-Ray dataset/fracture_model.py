
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

datadir = "."

folders = ["train", "val", "test"]
categories = ["fractured", "not fractured"]

for folder in folders:
    print("\n", folder)

    for category in categories:
        path = os.path.join(datadir, folder, category)

        images = os.listdir(path)

        print(category, ":", len(images))


training_data = []
testing_data = []
validation_data = []

img_size = 60

# training data
for category in categories:
    path = os.path.join(datadir, "train", category)
    class_num = categories.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            if img_array is None:
                continue

            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])

        except Exception as e:
            print(e)

# testing data
for category in categories:
    path = os.path.join(datadir, "test", category)
    class_num = categories.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            if img_array is None:
                continue

            new_array = cv2.resize(img_array, (img_size, img_size))
            testing_data.append([new_array, class_num])

        except Exception as e:
            print(e)

# validation data
for category in categories:
    path = os.path.join(datadir, "val", category)
    class_num = categories.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            if img_array is None:
                continue

            new_array = cv2.resize(img_array, (img_size, img_size))
            validation_data.append([new_array, class_num])

        except Exception as e:
            print(e)

print("training images:", len(training_data))
print("testing images:", len(testing_data))
print("validation images:", len(validation_data))

len(training_data)

# training data
X_train = np.array([i[0] for i in training_data])
y_train = np.array([i[1] for i in training_data])

# testing data
X_test = np.array([i[0] for i in testing_data])
y_test = np.array([i[1] for i in testing_data])

# validation data
X_val = np.array([i[0] for i in validation_data])
y_val = np.array([i[1] for i in validation_data])

X_train = X_train / 255.0
X_test = X_test / 255.0
X_val = X_val / 255.0

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)


# testing the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("test loss:", test_loss)
print("test accuracy:", test_accuracy)
