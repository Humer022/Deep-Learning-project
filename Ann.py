# -*- coding: utf-8 -*-
"""ANN.ipynb


Original file is located at
    https://colab.research.google.com/drive/1arHc0TLUJe2-7tMy44QEYUhRdIU3WMAc
"""

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Step 2: Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Step 3: Preprocess the data (normalize and one-hot encode)
scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)

y_encoded = to_categorical(y)

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=30)

from tensorflow.keras.layers import Dropout

model = Sequential()

# Hidden Layer 1
model.add(Dense(80, input_shape=(4,), activation='relu'))
model.add(Dropout(0.2))  # 20% dropout

# Hidden Layer 2
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))  # Another dropout

# Output Layer
model.add(Dense(3, activation='softmax'))

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#6: Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, verbose=1)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy       : {accuracy:.4f}")
print(f"Test Loss           : {loss:.4f}")

# Extracting training history
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"\nFinal Train Accuracy: {train_accuracy:.4f}")
print(f"Final Val Accuracy  : {val_accuracy:.4f}")
print(f"Final Train Loss    : {train_loss:.4f}")
print(f"Final Val Loss      : {val_loss:.4f}")

# Step 8: Visualize loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Make Predictions on New Flower
# Example input (sepal & petal length/width)
new_sample = np.array([[6.0, 2.9, 4.5, 1.5]])  # Likely Setosa

# Scale it using same scaler
new_sample_scaled = scaler.transform(new_sample)

# Predict
prediction = model.predict(new_sample_scaled)
predicted_class = np.argmax(prediction)

# Class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

print(f"Predicted Class: {class_names[predicted_class]}")