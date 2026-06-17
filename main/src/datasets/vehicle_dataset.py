import tensorflow as tf
import os


def _build_aug(task: str) -> "tf.keras.Sequential":
    """Build a task-specific augmentation pipeline.

    Color classification is sensitive to hue/saturation shifts that would
    corrupt the class signal, so only geometry + mild brightness/contrast are
    used.  Brand classification depends on shape/logo features and tolerates
    stronger augmentation.

    Args:
        task: One of ``"color"`` or ``"brand"``.  Unknown values fall back to
            the brand recipe.

    Returns:
        A ``tf.keras.Sequential`` augmentation model (no Rescaling included).
    """
    layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ]
    if task == "color":
        # Color: geometry + mild brightness/contrast ONLY.  Hue/saturation
        # jitter is intentionally EXCLUDED — it corrupts the class signal.
        layers += [
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
            tf.keras.layers.RandomContrast(0.15),
        ]
    else:  # brand: shape/logo features tolerate stronger contrast jitter
        layers += [
            tf.keras.layers.RandomContrast(0.3),
        ]
    return tf.keras.Sequential(layers)


def load_split_dataset(base_dir, batch_size=32, img_height=224, img_width=224,
                       seed=42, task: str = "color"):
    """Load a pre-split dataset laid out as ``base_dir/{train,val,test}/<class>``.

    Unlike :func:`load_classification_dataset` (which carves a validation slice
    out of one folder and exposes no test set), this consumes the physical
    train/val/test split produced by ``scripts/split_dataset.py`` so the held-out
    test set is stable and never seen during training.

    Images are normalised to ``[0, 1]``; the model's own input Rescaling layer
    then maps to whatever range its backbone expects. Augmentation is applied to
    the training split only.

    Args:
        base_dir: Root directory containing ``train/``, ``val/``, ``test/``
            sub-directories each with one sub-folder per class.
        batch_size: Number of images per batch.
        img_height: Resize height in pixels.
        img_width: Resize width in pixels.
        seed: Random seed for reproducibility.
        task: ``"color"`` (geometry + mild brightness/contrast, NO hue jitter)
            or ``"brand"`` (stronger contrast augmentation).  Controls the
            augmentation pipeline applied to the training split.

    Returns:
        ``(train_ds, val_ds, test_ds, class_names)``.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Split dataset directory not found at: {base_dir}")

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation_model = _build_aug(task)

    def _make(split):
        ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(base_dir, split),
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=(split == "train"),
        )
        class_names = ds.class_names
        if split == "train":
            ds = ds.map(lambda x, y: (augmentation_model(normalization_layer(x)), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (normalization_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE), class_names

    train_ds, class_names = _make("train")
    val_ds, _ = _make("val")
    test_ds, _ = _make("test")
    return train_ds, val_ds, test_ds, class_names


def load_classification_dataset(data_dir, batch_size=32, img_height=224, img_width=224, validation_split=0.2, subset="both", seed=42):
    """
    Loads and splits a classification dataset from a directory of subfolders.
    
    Args:
        data_dir (str): Directory containing class subfolders.
        batch_size (int): Batch size.
        img_height (int): Target height.
        img_width (int): Target width.
        validation_split (float): Fraction of data for validation.
        subset (str): One of "training", "validation", or "both".
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple or tf.data.Dataset: (train_ds, val_ds) or single dataset.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")

    # Standard preprocessing layers
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    augmentation_model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    def preprocess(image, label, augment=False):
        # Normalize to [0, 1]
        image = normalization_layer(image)
        if augment:
            image = augmentation_model(image)
        return image, label

    if subset == "both":
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode="categorical"
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode="categorical"
        )
        
        # Apply preprocessing
        train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: preprocess(x, y, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
        
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    else:
        ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset=subset,
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode="categorical"
        )
        augment = (subset == "training")
        ds = ds.map(lambda x, y: preprocess(x, y, augment=augment), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
