from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Preprocess images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='training'6
)

val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Step 2: Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output: 1 = open, 0 = closed
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Step 5: Save the trained model
model.save('eye_state_model.h5')

print("Model training complete and saved as 'eye_state_model.h5'")
train_model.py