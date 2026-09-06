"""
train_model.py
---------------
Trains a breed-classification model for Indian Cattle & Buffalo breeds
using Transfer Learning (MobileNetV2 backbone, pretrained on ImageNet).

Matches the "Methodologies" pipeline from the project report:
    Image Upload -> Preprocessing -> Feature Extraction ->
    Breed Classification -> Confidence & Breed Information -> UI

Expected folder layout (create with prepare_dataset.py):

    dataset/
        train/<breed_name>/*.jpg
        val/<breed_name>/*.jpg
        test/<breed_name>/*.jpg      (optional, used only for final evaluation)

Run:
    python train_model.py --epochs 25 --batch_size 32
    python train_model.py --epochs 10 --fine_tune            # 2nd stage fine-tuning

Outputs (saved into ./model/):
    model/breed_model.h5         -> the trained Keras model (loaded by app.py)
    model/class_indices.json     -> {class_name: index} mapping used at inference
    model/training_history.png   -> accuracy / loss curves
    model/confusion_matrix.png   -> evaluation on the test set (if present)
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
MODEL_DIR = "model"
DATASET_DIR = "dataset"


def build_generators(batch_size: int):
    """Image preprocessing + augmentation (per the report's Data Preprocessing step)."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


def build_model(num_classes: int, fine_tune: bool = False):
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = fine_tune
    if fine_tune:
        # Only unfreeze the last ~30 layers for gentle fine-tuning
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    lr = 1e-5 if fine_tune else 1e-3
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
    return model


def plot_history(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune", action="store_true",
                         help="Continue training an existing model with the backbone unfrozen (stage 2).")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    train_gen, val_gen = build_generators(args.batch_size)
    num_classes = train_gen.num_classes
    print(f"Found {num_classes} breed classes: {list(train_gen.class_indices.keys())}")

    # Save the class index mapping immediately - app.py depends on this file.
    class_indices_path = os.path.join(MODEL_DIR, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Saved class index mapping to {class_indices_path}")

    model_path = os.path.join(MODEL_DIR, "breed_model.h5")

    if args.fine_tune and os.path.exists(model_path):
        print("Loading existing model for fine-tuning stage...")
        model = tf.keras.models.load_model(model_path)
        base_model = model.layers[1]
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                       loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model = build_model(num_classes, fine_tune=False)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=3, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    # Optional final evaluation on a held-out test set + confusion matrix
    test_dir = os.path.join(DATASET_DIR, "test")
    if os.path.isdir(test_dir):
        evaluate_on_test(model, train_gen.class_indices, test_dir)


def evaluate_on_test(model, class_indices, test_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    idx_to_class = {v: k for k, v in class_indices.items()}
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=32,
        class_mode="categorical", shuffle=False,
    )

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("\nClassification report (test set):")
    print(classification_report(y_true, y_pred,
                                 target_names=[idx_to_class[i] for i in range(len(idx_to_class))]))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(idx_to_class)))
    ax.set_yticks(range(len(idx_to_class)))
    ax.set_xticklabels([idx_to_class[i] for i in range(len(idx_to_class))], rotation=90)
    ax.set_yticklabels([idx_to_class[i] for i in range(len(idx_to_class))])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Test Set")
    fig.colorbar(im)
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), bbox_inches="tight")
    print("Saved confusion matrix to model/confusion_matrix.png")


if __name__ == "__main__":
    main()
