import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib
from scipy import ndimage
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Assuming you have datasets for segmentation: fcd_dataset_paths and roi_dataset_paths

import os

# Assuming you have datasets for segmentation: fcd_dataset_paths and roi_dataset_paths

# Load Segmentation Datasets
fcd_dataset_paths = [
    os.path.join(os.getcwd(), "/kaggle/input/flairrr/flair/SORTED_WITH_FCD_FLAIR", x)
    for x in os.listdir("/kaggle/input/flairrr/flair/SORTED_WITH_FCD_FLAIR")
    if x.endswith(".nii")  # Filter only NIfTI files
]

roi_dataset_paths = [
    os.path.join(os.getcwd(), "/kaggle/input/flairrr/roi/ROI", x)
    for x in os.listdir("/kaggle/input/flairrr/roi/ROI")
    if x.endswith(".nii")  # Filter only NIfTI files
]

def process_segmentation_data(path, target_shape):
    segmentation_data = read_nifti_file(path)
    segmentation_data = resize_volume(segmentation_data, target_shape)
    segmentation_data = normalize(segmentation_data)
    return segmentation_data

def process_roi_data(path, target_shape, intensity_threshold=0.5):
    roi_data = read_nifti_file(path)
    roi_data = resize_volume(roi_data, target_shape)
    # Assuming intensity_threshold is the value to differentiate the colored part
    roi_mask = (roi_data > intensity_threshold).astype(np.float32)
    return roi_mask

def read_nifti_file(file_path):
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    return nifti_data

def resize_volume(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = [target / current for target, current in zip(target_shape, current_shape)]
    resized_volume = ndimage.zoom(volume, zoom_factors, order=1)
    return resized_volume

def normalize(volume):
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    normalized_volume = (volume - mean_val) / std_val
    return normalized_volume

# Assuming you have segmentation labels (masks) corresponding to FCD and ROI datasets
fcd_labels = np.ones((len(fcd_dataset_paths), 64, 64, 128, 1), dtype=np.float32)
roi_labels = np.zeros((len(roi_dataset_paths), 64, 64, 128, 1), dtype=np.float32)




target_shape = (64, 64, 128)

# Read and process the segmentation data
fcd_segmentation_data = np.array([process_segmentation_data(path, target_shape) for path in fcd_dataset_paths])
roi_segmentation_data = np.array([process_segmentation_data(path, target_shape) for path in roi_dataset_paths])







# Split data for training and validation
split_index_segmentation = int(0.7 * min(len(fcd_dataset_paths), len(roi_dataset_paths)))
x_train_segmentation = np.concatenate((fcd_segmentation_data[:split_index_segmentation], roi_segmentation_data[:split_index_segmentation]), axis=0)
y_train_segmentation = np.concatenate((fcd_labels[:split_index_segmentation], roi_labels[:split_index_segmentation]), axis=0)
x_val_segmentation = np.concatenate((fcd_segmentation_data[split_index_segmentation:], roi_segmentation_data[split_index_segmentation:]), axis=0)
y_val_segmentation = np.concatenate((fcd_labels[split_index_segmentation:], roi_labels[split_index_segmentation:]), axis=0)

# Data loaders for segmentation
segmentation_train_loader = tf.data.Dataset.from_tensor_slices((x_train_segmentation, y_train_segmentation))
segmentation_validation_loader = tf.data.Dataset.from_tensor_slices((x_val_segmentation, y_val_segmentation))
batch_size_segmentation = 1
segmentation_train_dataset = (
    segmentation_train_loader.shuffle(len(x_train_segmentation))
    .batch(batch_size_segmentation)
    .prefetch(2)
)

segmentation_validation_dataset = (
    segmentation_validation_loader.shuffle(len(x_val_segmentation))
    .batch(batch_size_segmentation)
    .prefetch(2)
)



def get_unet_model(input_shape=(64, 64, 128, 1)):  # Updated input shape
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv3D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    # Decoder
    conv2 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv2)
    up1 = layers.UpSampling3D(size=(2, 2, 2))(conv2)

    # Output
    output_segmentation = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(up1)

    model = models.Model(inputs, output_segmentation, name="3d_unet_segmentation")
    return model

# Create U-Net model for segmentation
seg_model = get_unet_model(input_shape=(64, 64, 128, 1))
seg_model.summary()

# Compile the segmentation model
seg_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from scipy import ndimage
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Define the Dice coefficient, sensitivity, and accuracy functions
def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + 1e-7) / (union + 1e-7)

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * y_pred)
    actual_positives = tf.reduce_sum(y_true)
    return true_positives / (actual_positives + 1e-7)

# Define custom callback to calculate metrics during training
class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_preds = self.model.predict(self.validation_data[0])
        val_dice = dice_coefficient(self.validation_data[1], val_preds)
        val_sensitivity = sensitivity(self.validation_data[1], val_preds)
        val_accuracy = logs['val_accuracy']  # Access accuracy from logs
        print(f'Validation Dice coefficient: {val_dice}, Sensitivity: {val_sensitivity}, Accuracy: {val_accuracy}')

# Assuming 'seg_checkpoint_cb' is your ModelCheckpoint callback
seg_checkpoint_cb = keras.callbacks.ModelCheckpoint("3d_segmentation_model.keras", save_best_only=True)

# Train the segmentation model with custom callback
seg_model.fit(
    segmentation_train_dataset,
    validation_data=segmentation_validation_dataset,
    epochs=30,
    shuffle=True,
    verbose=2,
    callbacks=[seg_checkpoint_cb, MetricsCallback((x_val_segmentation, y_val_segmentation))]
)
# Assuming you have a test dataset for segmentation
# Load and process the test segmentation data similarly to how you processed training and validation data

# Assuming you have a function to preprocess test data similar to what you've done before
def preprocess_test_data(test_dataset_paths, target_shape):
    # Process the test data similarly to how you processed training and validation data
    test_segmentation_data = np.array([process_segmentation_data(path, target_shape) for path in test_dataset_paths])
    return test_segmentation_data

# Assuming you have test dataset paths
test_dataset_paths = [
    os.path.join(os.getcwd(), "/kaggle/input/flairrr/roi/ROI/sub-00006_acq-tse3dvfl_FLAIR_roi.nii")
]

# Preprocess test data
x_test_segmentation = preprocess_test_data(test_dataset_paths, target_shape)

# Predict segmentation masks
predicted_masks = seg_model.predict(x_test_segmentation, batch_size=batch_size_segmentation)

# Assuming you want to visualize the prediction for the only test sample
sample_index = 0  # Change this index according to your preference

# Modified visualization function to handle 4D mask
def visualize_segmentation_mask(mask):
    # Remove the last dimension
    mask = np.squeeze(mask, axis=-1)
    # Visualize the segmentation mask
    plt.imshow(mask[..., 83], cmap='gray')
    plt.title("Segmentation Mask")
    plt.colorbar()
    plt.show()

# Visualize the predicted segmentation mask for the selected sample
visualize_segmentation_mask(predicted_masks[sample_index])
# Modified visualization function with larger images and three columns
def visualize_segmentation_mask_present_slices_large(mask):
    # Remove the last dimension
    mask = np.squeeze(mask, axis=-1)
    # Get the indices of slices where mask is present
    present_slices_indices = np.where(np.any(mask != 0, axis=(0, 1)))[0]
    
    # Plot only the slices where mask is present
    num_present_slices = len(present_slices_indices)
    if num_present_slices > 0:
        num_columns = 3  # Adjust the number of columns as needed
        num_rows = (num_present_slices + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 6*num_rows))  # Adjust figsize as needed
        for i, slice_index in enumerate(present_slices_indices):
            row_index = i // num_columns
            col_index = i % num_columns
            axes[row_index, col_index].imshow(mask[..., slice_index], cmap='gray')
            axes[row_index, col_index].set_title(f"Slice {slice_index}")
            axes[row_index, col_index].axis('off')
        plt.show()
    else:
        print("No slices with mask present")

# Visualize slices where the predicted segmentation mask is present for the selected sample (larger images, three columns)
visualize_segmentation_mask_present_slices_large(predicted_masks[sample_index])
# Step 1: Read the NIfTI file
file_path = "/kaggle/input/flairrr/roi/ROI/sub-00006_acq-tse3dvfl_FLAIR_roi.nii"
segmentation_data = read_nifti_file(file_path)

# Step 2: Process the segmentation data
processed_segmentation_data = process_segmentation_data(file_path, target_shape)

# Step 3: Display the 83rd slice
slice_index = 83
plt.imshow(processed_segmentation_data[..., slice_index], cmap='gray')
plt.title(f"Slice {slice_index}")
plt.colorbar()
plt.show()
