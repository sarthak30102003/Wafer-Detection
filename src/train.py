"""
train.py - Training script for Wafer Defect Classification

Author: Sarthak Aggarwal
Date: 30-08-2025
Description:
    - Loads wafer defect dataset
    - Preprocesses images (cropping + resizing)
    - Trains a CNN model
    - Saves trained model to /models
"""

import os
import cv2
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence, to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore


# ---- Custom Data Generator ---- #
class WaferDataGenerator(Sequence):
    """Custom generator for wafer defect dataset."""

    def __init__(self, image_paths, labels, batch_size, img_size, num_classes, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = []

        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = self.crop_axes(img)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype("float32") / 255.0
            batch_images.append(img)

        batch_images = np.array(batch_images)
        batch_labels = to_categorical(batch_labels[:len(batch_images)], num_classes=self.num_classes)
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.image_paths, self.labels))
            random.shuffle(temp)
            self.image_paths, self.labels = zip(*temp)

    @staticmethod
    def crop_axes(image):
        """Crop black axes around wafer images."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        return image[y:y+h, x:x+w]


def build_model(img_size, num_classes):
    """Define CNN model architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main(args):
    # Classes (adjust if dataset changes)
    classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc',
               'Near-full', 'none', 'Random', 'Scratch']

    # Collect image paths
    print("Collecting image paths...")
    image_paths, image_labels = [], []
    for idx, defect_class in enumerate(classes):
        class_dir = os.path.join(args.dataset_dir, defect_class)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, fname))
            image_labels.append(idx)

    # Train-validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, image_labels, test_size=0.2,
        stratify=image_labels, random_state=42
    )

    # Generators
    train_gen = WaferDataGenerator(train_paths, train_labels,
                                   args.batch_size, args.img_size, len(classes))
    val_gen = WaferDataGenerator(val_paths, val_labels,
                                 args.batch_size, args.img_size, len(classes))

    # Model
    model = build_model(args.img_size, len(classes))

    # Train
    print("Starting training...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, verbose=1)

    # Save
    os.makedirs("models", exist_ok=True)
    model.save("models/wafer_defect_model_generator.h5")
    print("✅ Model saved at models/wafer_defect_model_generator.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wafer defect detection model")
    parser.add_argument("--dataset_dir", type=str, default="data/train",
                        help="Path to training dataset directory")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image size for resizing")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    args = parser.parse_args()
    main(args)
