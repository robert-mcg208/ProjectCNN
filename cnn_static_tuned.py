import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_score
import matplotlib.pyplot as plt
import os

# Random seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
train_dir = r"C:\Users\rober\Documents\Uni\Year4\Honours\Project\dataset\data\binary\train"  
test_dir = r"C:\Users\rober\Documents\Uni\Year4\Honours\Project\dataset\data\binary\test"    
save_dir = r"C:\Users\rober\Documents\Uni\Year4\Honours\Project\CNN\FINISHED\results\binary21"  
os.makedirs(save_dir, exist_ok=True)  

# Image parameters
img_height = 300
img_width = 300
batch_size = 8  

# Scale and load images 
train_datagen = ImageDataGenerator(rescale=1./255) 
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_dataset = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

early_stopping = EarlyStopping(
    monitor='val_loss',       
    patience=5,                
    restore_best_weights=True  
)

# Build CNN 
my_model = Sequential([
    Conv2D(32, (3,3), activation='relu', strides = (2,2), input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(3,3)),
    
    Conv2D(64, (3,3), activation='relu', strides = (2,2)),
    MaxPooling2D(pool_size=(3,3)),

    Conv2D(128, (3,3), activation='relu', strides = (2,2)),
    MaxPooling2D(pool_size=(3,3)),

    Flatten(), 
    Dense(64, activation='relu'),
    Dropout(0.25),  
    Dense(32, activation='relu'),
    Dropout(0.20),  
    Dense(1, activation='sigmoid')  
])

# Compile CNN
my_model.compile(optimizer=Adam(learning_rate = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN  
history = my_model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=[early_stopping] 
)

# Evaluate CNN
test_loss, test_acc = my_model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.3f}")

# True labels and predictions
y_true = test_dataset.classes
y_pred_prob = my_model.predict(test_dataset)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix 
plt.figure(figsize=(5, 5))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xticks([0, 1], ['Benign', 'Malicious'])
plt.yticks([0, 1], ['Benign', 'Malicious'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black', fontsize=12)

conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

roc_curve_path = os.path.join(save_dir, "roc_curve.png")
plt.savefig(roc_curve_path)
plt.show()

# Learning Curves
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
learning_curves_path = os.path.join(save_dir, "learning_curves.png")
plt.savefig(learning_curves_path)
plt.show()

# Save model
my_model_save_path = os.path.join(save_dir, "static_model.keras")
my_model.save(my_model_save_path)