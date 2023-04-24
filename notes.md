Yes, TensorFlow, combined with the Keras API, provides a convenient way to collect and visualize loss (and other metrics) during training using TensorBoard. Here's an example of how to train a simple neural network and visualize the loss vs epoch using TensorFlow and TensorBoard:

 

1. Install TensorFlow and TensorBoard:

 

```bash
pip install tensorflow
pip install tensorboard
```

 

2. Import the necessary modules and prepare your dataset:

 

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

 

# Assuming X is your feature matrix and y is your target variable
X = ...
y = ...

 

# Scale the data (if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

 

3. Create and compile the model:

 

```python
# Create a simple neural network model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

 

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")
```

 

4. Set up TensorBoard:

 

```python
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

 

# Create a logs directory for TensorBoard
logs_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(logs_dir, exist_ok=True)

 

# Set up the TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)
```

 

5. Train the model with the TensorBoard callback:

 

```python
# Train the model and collect the history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])
```

 

6. Start TensorBoard:

 

```bash
tensorboard --logdir logs/fit
```

 

This command will start TensorBoard, and you can access it at `http://localhost:6006/` in your browser. There, you can visualize the loss (and other metrics) during training for each epoch.

 

You can also plot the loss vs epoch directly using the `history` object returned by the `fit` method:

 

```python
import matplotlib.pyplot as plt

 

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.show()
```