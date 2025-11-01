
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ✅ Dataset paths
train_dir = r"E:\Project\Skin_Disease_Prediction\archive (4)\Split_smol\train"
test_dir = r"E:\Project\Skin_Disease_Prediction\archive (4)\Split_smol\test"

# ✅ Data augmentation (helps a lot for small datasets)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# ✅ Load VGG16 and fine-tune last few layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)

# ✅ Extract CNN features
print("Extracting CNN features (this may take a few minutes)...")
X_train = model.predict(train_gen, verbose=1)
X_test = model.predict(test_gen, verbose=1)

# ✅ Flatten features
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

y_train = train_gen.classes
y_test = test_gen.classes

# ✅ Random Forest with tuning
print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_flat, y_train)

# ✅ Evaluate
y_pred = rf.predict(X_test_flat)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Improved Model Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=list(train_gen.class_indices.keys())))

# ✅ Save model
model_path = r"E:\Project\Skin_Disease_Prediction\hybrid_rf_vgg16_tuned.pkl"
joblib.dump(rf, model_path)
print(f"\nModel saved as: {model_path}")
