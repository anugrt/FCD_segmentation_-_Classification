
import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib
from scipy import ndimage



def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    volume = read_nifti_file(path)
    
    # Resize volume to a common shape
    target_shape = (60, 64, 64)
    volume = resize_volume(volume, target_shape)
    
    volume = normalize(volume)
    return volume

def resize_volume(img, target_shape):
    current_shape = img.shape
    depth = current_shape[0] / target_shape[0]
    width = current_shape[1] / target_shape[1]
    height = current_shape[2] / target_shape[2]
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.rotate(img, 50, reshape=False)
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    
    return img


# Paths to your dataset
normal_scan_paths = [
    os.path.join(os.getcwd(), "C://Users//HP//Desktop//CCCNN//SORTED_WITHOUT_FCD_FLAIR", x)
    for x in os.listdir("C://Users//HP//Desktop//CCCNN//SORTED_WITHOUT_FCD_FLAIR")
]

abnormal_scan_paths = [
    os.path.join(os.getcwd(), "C://Users//HP//Desktop//CCCNN//SORTED_WITH_FCD_FLAIR", x)
    for x in os.listdir("C://Users//HP//Desktop//CCCNN//SORTED_WITH_FCD_FLAIR")
]



# Read and process the scans.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

# Labels
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data for training and validation
split_index = int(0.7 * min(len(abnormal_scans), len(normal_scans)))
x_train = np.concatenate((abnormal_scans[:split_index], normal_scans[:split_index]), axis=0)
y_train = np.concatenate((abnormal_labels[:split_index], normal_labels[:split_index]), axis=0)
x_val = np.concatenate((abnormal_scans[split_index:], normal_scans[split_index:]), axis=0)
y_val = np.concatenate((abnormal_labels[split_index:], normal_labels[split_index:]), axis=0)

# Data loaders
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
batch_size = 1
train_dataset = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(2)
)

validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .batch(batch_size)
    .prefetch(2)
)



print(len(list(train_dataset)))


print(len(list(validation_dataset)))



# Define spatial attention
def spatial_attention(x):
    squeeze = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(x)
    return layers.multiply([x, squeeze])



def get_model_with_attention_v2(width=64, height=64, depth=60):
    inputs = keras.Input((depth, width, height, 1))  
    
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", kernel_initializer='he_normal')(x)
    x = spatial_attention(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn_with_attention_v2")
    return model



# Build model with spatial attention.
model_with_attention = get_model_with_attention_v2(width=64, height=64, depth=60)
model_with_attention.summary()



# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model_with_attention.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)

# Train the model, doing validation at the end of each epoch
epochs = 25
model_with_attention.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb],
)


from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
y_pred_prob = model_with_attention.predict(validation_dataset).squeeze()
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import f1_score

# Assuming you have the true labels (y_true) and predicted labels (y_pred) from your validation set
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
y_pred_prob = model_with_attention.predict(validation_dataset).squeeze()
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_true, y_pred)

print(f'F1 Score: {f1:.4f}')


from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Function to plot the training/validation history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Train the model and collect history
history = model_with_attention.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb],
)

# Plot training/validation history
plot_history(history)

# Evaluate the model on the validation set
val_loss, val_acc = model_with_attention.evaluate(validation_dataset, verbose=0)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Predictions on validation set
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
y_pred_prob = model_with_attention.predict(validation_dataset).squeeze()
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
print('Classification Report:')
print(classification_report(y_true, y_pred))

# F1 Score
f1 = f1_score(y_true, y_pred)
print(f'F1 Score: {f1:.4f}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming you have the true labels (y_true) and predicted labels (y_pred) from your validation set
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
y_pred_prob = model_with_attention.predict(validation_dataset).squeeze()
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
