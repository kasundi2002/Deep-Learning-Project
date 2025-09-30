import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Paths & Config
# ===========================
DATA_DIR = "data"
RUNS_DIR = "runs/densenet121"
os.makedirs(RUNS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# ===========================
# Data Generators
# ===========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

# ===========================
# Build DenseNet121
# ===========================
base_model = DenseNet121(weights="imagenet", include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze first

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ===========================
# Class Weights (to fix imbalance)
# ===========================
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# ===========================
# Callbacks
# ===========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(RUNS_DIR, "best_model.h5"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
]

# ===========================
# Training Phase 1 (frozen base)
# ===========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ===========================
# Fine-tuning (unfreeze last 30 layers)
# ===========================
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ===========================
# Evaluation
# ===========================
model.load_weights(os.path.join(RUNS_DIR, "best_model.h5"))

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Save classification report
report = classification_report(
    y_true, y_pred_classes,
    target_names=list(test_generator.class_indices.keys())
)
with open(os.path.join(RUNS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print(report)

# Save confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DenseNet121 Confusion Matrix")
plt.savefig(os.path.join(RUNS_DIR, "confusion_matrix.png"))
plt.close()

# Save training curve
plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Training/Validation Accuracy")
plt.savefig(os.path.join(RUNS_DIR, "training_curve.png"))
plt.close()
