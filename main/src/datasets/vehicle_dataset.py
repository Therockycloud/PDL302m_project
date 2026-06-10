import tensorflow as tf
import os

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
