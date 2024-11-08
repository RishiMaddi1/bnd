# train_model.py
import numpy as np 
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load and preprocess the data
image_size = 150
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_data():
    print("Loading data...")
    X_train = []
    Y_train = []
    
    # Load the training data
    for label in labels:
        folderPath = os.path.join('dataset/Training', label)
        for image_file in os.listdir(folderPath):
            img = cv2.imread(os.path.join(folderPath, image_file))
            img = cv2.resize(img, (image_size, image_size))
            X_train.append(img)
            Y_train.append(label)

    # Convert to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Shuffle the data
    X_train, Y_train = shuffle(X_train, Y_train, random_state=101)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=101)

    # Convert labels to categorical format
    y_train = tf.keras.utils.to_categorical([labels.index(i) for i in y_train])
    y_test = tf.keras.utils.to_categorical([labels.index(i) for i in y_test])

    return X_train, y_train, X_test, y_test

def train_model():
    X_train, y_train, X_test, y_test = load_data()
    print("Data loaded. Starting training...")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow(X_train, y_train, batch_size=32)

    # Load ResNet50 model for transfer learning
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(len(labels), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_gen, epochs=2, validation_data=(X_test, y_test))
    print("Training complete.")

    # Save the model
    model.save('brain_tumor_model.h5')

if __name__ == "__main__":
    train_model()  # Run this to train and save the model