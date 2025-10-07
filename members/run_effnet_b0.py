import os, json, time, argparse
import numpy as np
import tensorflow as tf
import os, shutil, random
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

base_dir = "/content/ham10000_raw"
output_dir = "/content/ham10000_split"
os.makedirs(output_dir, exist_ok=True)

splits = ["train", "val", "test"]
for s in splits:
    for c in os.listdir(base_dir):
        os.makedirs(os.path.join(output_dir, s, c), exist_ok=True)

for c in os.listdir(base_dir):
    if not os.path.isdir(os.path.join(base_dir, c)): continue
    images = os.listdir(os.path.join(base_dir, c))
    random.shuffle(images)
    n = len(images)
    train_end, val_end = int(0.8*n), int(0.9*n)
    for i, img in enumerate(images):
        if i < train_end: split = "train"
        elif i < val_end: split = "val"
        else: split = "test"
        shutil.copy(os.path.join(base_dir, c, img), os.path.join(output_dir, split, c))

def get_datasets(data_dir, img_size=(224,224), batch=32, seed=42, binary=False):
    def loader(split, shuffle):
        return keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_dir, split), image_size=img_size,
            batch_size=batch, seed=seed, shuffle=shuffle
        )
    ds_train = loader("train", True)
    ds_val   = loader("val",   False)
    ds_test  = loader("test",  False)

    class_names = ds_train.class_names
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    ds_train = ds_train.map(lambda x,y: (aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    autotune = tf.data.AUTOTUNE
    ds_train = ds_train.prefetch(autotune)
    ds_val   = ds_val.prefetch(autotune)
    ds_test  = ds_test.prefetch(autotune)
    return ds_train, ds_val, ds_test, class_names, binary

def compute_class_weights(ds, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for _, y in ds.unbatch():
        counts[int(y.numpy())]+=1
    total = counts.sum()
    return {i: total/(num_classes*max(counts[i],1)) for i in range(num_classes)}, counts.tolist()

def build_model(num_classes, img_size=(224,224), binary=False):
    inp = keras.Input(shape=(*img_size,3))
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x) if binary \
          else layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    return model, base

def compile_and_fit(model, train_ds, val_ds, *, loss, lr, epochs, class_weight, ckpt_path):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    cbs = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, mode="max")
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=cbs)
    with open(os.path.join(os.path.dirname(ckpt_path), "history.json"), "w") as f:
        json.dump({k:[float(v) for v in vals] for k,vals in hist.history.items()}, f, indent=2)

def evaluate_and_save(model, ds_test, class_names, out_dir, binary=False):
    y_true, y_pred = [], []
    for x, y in ds_test:
        p = model.predict(x, verbose=0)
        if binary:
            yp = (p.reshape(-1) >= 0.5).astype(int)
        else:
            yp = p.argmax(axis=1)
        y_true.extend(y.numpy().tolist()); y_pred.extend(yp.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average=("binary" if binary else "macro")))
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred).tolist()
    json.dump({"accuracy":acc,"macro_f1":macro_f1,"confusion_matrix":cm}, open(os.path.join(out_dir,"metrics.json"),"w"), indent=2)
    open(os.path.join(out_dir,"classification_report.txt"),"w").write(report)
    json.dump(class_names, open(os.path.join(out_dir,"classes.json"),"w"), indent=2)
    print(report); print("Saved to:", out_dir)

def main(args):
    tf.random.set_seed(42)
    run = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("runs","effnet",run); os.makedirs(out_dir, exist_ok=True)

    ds_tr, ds_va, ds_te, class_names, binary = get_datasets(args.data, (args.size,args.size), args.batch, binary=bool(args.binary))
    num_classes = 2 if binary else len(class_names)
    cw, counts = compute_class_weights(ds_tr, num_classes)
    print("Class counts:", counts); print("Class weights:", cw)

    model, base = build_model(num_classes, (args.size,args.size), binary)
    loss = "binary_crossentropy" if binary else "sparse_categorical_crossentropy"

    # Stage 1: warmup (freeze base)
    base.trainable = False
    compile_and_fit(model, ds_tr, ds_va, loss=loss, lr=args.base_lr, epochs=max(3,args.warmup), class_weight=cw, ckpt_path=os.path.join(out_dir,"best.keras"))

    # Stage 2: simple fine-tune (unfreeze last blocks)
    base.trainable = True
    if args.unfreeze > 0 and len(base.layers) > args.unfreeze:
        for l in base.layers[:-args.unfreeze]:
            l.trainable = False
    compile_and_fit(model, ds_tr, ds_va, loss=loss, lr=1e-5, epochs=args.epochs, class_weight=cw, ckpt_path=os.path.join(out_dir,"best.keras"))

    evaluate_and_save(model, ds_te, class_names, out_dir, binary)

class Args:
    data = "/content/ham10000_split"
    epochs = 10
    warmup = 3
    unfreeze = 20
    size = 128
    batch = 16
    base_lr = 1e-4
    binary = 0
    run_name = ""

args = Args()
main(args)