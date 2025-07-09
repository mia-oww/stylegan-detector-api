import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix 

###### Helper functions ######

def getCoMatrices(img):
    b, g, r = cv2.split(img)
    distance = 1
    angle = 0
    rcomatrix = greycomatrix(r, [distance], [angle])
    gcomatrix = greycomatrix(g, [distance], [angle])
    bcomatrix = greycomatrix(b, [distance], [angle])
    tensor = tf.constant([rcomatrix[:,:,0,0], gcomatrix[:,:,0,0], bcomatrix[:,:,0,0]])
    tensor = tf.reshape(tensor, [256, 256, 3])
    return tensor

def genDs(augmentedFakeFolder, augmentedRealFolder):
    trainingDs = []
    trainingLabels = []
    valDs = []
    valLabels = []
    split = 0.8
    fakeLabel = 0
    realLabel = 1
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    dirsFake = os.listdir(augmentedFakeFolder)
    dirsReal = os.listdir(augmentedRealFolder)
    imageToProcess = min(len(dirsFake), len(dirsReal))
    maxFake = int(imageToProcess * split)
    maxReal = int(imageToProcess * split)

    count = 0
    for image in dirsFake:
        ext = os.path.splitext(image)[1].lower()
        if ext not in valid_exts:
            print(f"Skipping non-image file in fake folder: {image}")
            continue
        if count >= imageToProcess:
            break
        img_path = os.path.join(augmentedFakeFolder, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read fake image file: {img_path}")
            continue
        img = cv2.resize(img, (256, 256))
        if count >= maxFake:
            valDs.append(getCoMatrices(img))
            valLabels.append([fakeLabel])
        else:
            trainingDs.append(getCoMatrices(img))
            trainingLabels.append([fakeLabel])
        count += 1

    count = 0
    for image in dirsReal:
        ext = os.path.splitext(image)[1].lower()
        if ext not in valid_exts:
            print(f"Skipping non-image file in real folder: {image}")
            continue
        if count >= imageToProcess:
            break
        img_path = os.path.join(augmentedRealFolder, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read real image file: {img_path}")
            continue
        img = cv2.resize(img, (256, 256))
        if count >= maxReal:
            valDs.append(getCoMatrices(img))
            valLabels.append([realLabel])
        else:
            trainingDs.append(getCoMatrices(img))
            trainingLabels.append([realLabel])
        count += 1

    if len(trainingDs) == 0 or len(valDs) == 0:
        print("Error: No valid images loaded for training or validation!")
        sys.exit(1)

    trainingLabels = np.asarray(trainingLabels)
    valLabels = np.asarray(valLabels)
    trainingDs = tf.stack(trainingDs)
    valDs = tf.stack(valDs)

    print(f"Training data shape: {trainingDs.shape}, labels shape: {trainingLabels.shape}")
    print(f"Validation data shape: {valDs.shape}, labels shape: {valLabels.shape}")

    return trainingDs, trainingLabels, valDs, valLabels

def trainModel():
    model = models.Sequential() 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2))

    model.compile(optimizer=Adam(learning_rate=0.000075),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def plotAccuracy(model, train_matrices, train_labels, test_matrices, test_labels):
    data = model.fit(train_matrices, train_labels, epochs=5, batch_size=40,
                      validation_data=(test_matrices, test_labels))

    plt.plot(data.history['loss'])
    plt.plot(data.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()

    plt.plot(data.history['accuracy'], label='accuracy')
    plt.plot(data.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss, test_acc = model.evaluate(test_matrices,  test_labels, verbose=2)
    plt.show()
    print('Accuracy is ' + str(test_acc * 100) + '%')

###### Main Script ######

if len(sys.argv) != 3:
    print("Usage: python train_png_model.py [fake_img_dir] [real_img_dir]")
    sys.exit(1)

# Generate dataset from images
tds, tlbl, vds, vlbl = genDs(sys.argv[1], sys.argv[2])

# training model
modelCNN = trainModel()
modelCNN.summary()
plotAccuracy(modelCNN, tds, tlbl, vds, vlbl)

# ensuring model directory exists
os.makedirs('models/png_model', exist_ok=True)

# saving full model (architecture + weights)
modelCNN.save('models/png_model/ImageDetectmodel.h5')

print("Model saved successfully.")
