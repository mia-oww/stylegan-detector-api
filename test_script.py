import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from skimage.feature import greycomatrix

def getCoMatrices(img):
    b, g, r = cv2.split(img)
    distance = 1
    angle = 0
    rcomatrix = greycomatrix(r, [distance], [angle])
    gcomatrix = greycomatrix(g, [distance], [angle])
    bcomatrix = greycomatrix(b, [distance], [angle])
    tensor = tf.constant([rcomatrix[:, :, 0, 0], gcomatrix[:, :, 0, 0], bcomatrix[:, :, 0, 0]])
    tensor = tf.reshape(tensor, [256, 256, 3])
    return tensor

def genDs(fakeFolder, realFolder):
    trainingDs = []
    trainingLabels = []
    valDs = []
    valLabels = []
    split = 0.8  # 80% train, 20% val
    fakeLabel = 0
    realLabel = 1
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    fakeFiles = [f for f in os.listdir(fakeFolder) if os.path.splitext(f)[1].lower() in valid_exts]
    realFiles = [f for f in os.listdir(realFolder) if os.path.splitext(f)[1].lower() in valid_exts]

    imageToProcess = min(len(fakeFiles), len(realFiles))
    maxTrainFake = int(imageToProcess * split)
    maxTrainReal = int(imageToProcess * split)

    # for fake images
    count = 0
    for image in fakeFiles:
        if count >= imageToProcess:
            break
        img_path = os.path.join(fakeFolder, image)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Skipping unreadable fake image {img_path}")
            continue
        if count < maxTrainFake:
            trainingDs.append(getCoMatrices(img))
            trainingLabels.append(fakeLabel)
        else:
            valDs.append(getCoMatrices(img))
            valLabels.append(fakeLabel)
        count += 1

    # for real images
    count = 0
    for image in realFiles:
        if count >= imageToProcess:
            break
        img_path = os.path.join(realFolder, image)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Skipping unreadable real image {img_path}")
            continue
        if count < maxTrainReal:
            trainingDs.append(getCoMatrices(img))
            trainingLabels.append(realLabel)
        else:
            valDs.append(getCoMatrices(img))
            valLabels.append(realLabel)
        count += 1

    # convert to numpy/tensors
    trainingDs = tf.stack(trainingDs)
    valDs = tf.stack(valDs)
    trainingLabels = np.array(trainingLabels)
    valLabels = np.array(valLabels)

    print(f"Training data shape: {trainingDs.shape}, labels shape: {trainingLabels.shape}")
    print(f"Validation data shape: {valDs.shape}, labels shape: {valLabels.shape}")

    return trainingDs, trainingLabels, valDs, valLabels

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 3)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),

        layers.Dense(2, activation='softmax')  # 2 classes: fake, real
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    fake_folder = "fake_images"
    real_folder = "real_images"

    # generates dataset
    tds, tlbl, vds, vlbl = genDs(fake_folder, real_folder)
    if tds.shape[0] == 0:
        print("Error: No valid images loaded for training.")
        return

    # for building and training model
    model = build_model()
    model.summary()

    model.fit(tds, tlbl, epochs=5, batch_size=40, validation_data=(vds, vlbl))

    # saving model
    os.makedirs('models/png_model', exist_ok=True)
    os.makedirs('models/jpg_model', exist_ok=True)
    model.save('models/png_model/ImageDetectmodel.h5')
    model.save('models/jpg_model/ImageDetectmodel.h5')
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
