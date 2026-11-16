
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Q1. Install and verify TensorFlow version
print("TensorFlow Version:", tf.__version__)


# Q2. Simple Linear Regression
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='sgd', loss='mse')
model.fit(x, y, epochs=200, verbose=0)
print("\nQ2 Prediction:", model.predict(np.array([[6.0]])))


# Q3. Element-wise Matrix Multiplication
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
print("\nQ3 Result:\n", tf.multiply(a, b).numpy())


# Q4. Compute Gradients using GradientTape
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    y = x ** 2 + 3 * x + 2
gradient = tape.gradient(y, x)
print("\nQ4 Gradient:", gradient.numpy())


# Q5. Custom Loss Function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='adam', loss=custom_loss)
model.fit([1, 2, 3, 4], [2, 4, 6, 8], epochs=10, verbose=0)
print("\nQ5 Custom loss model trained")


# Q6. ReLU Activation Function
x = tf.constant([-3.0, -2.0, 0.0, 2.0, 3.0])
print("\nQ6 ReLU:", tf.nn.relu(x).numpy())


# Q7. Neural Network to Approximate Sine Function
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, verbose=0)
print("\nQ7 Prediction:", model.predict(np.array([[1.0], [2.0], [3.0]])))


# Q8. Binary Classification using Sequential API
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=100, verbose=0)
print("\nQ8 Predictions:\n", model.predict(x))


# Q9. Visualize Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png',
                          show_shapes=True)
print("\nQ9 Model architecture saved")


# Q10. Save and Load Keras Model
model.compile(optimizer='adam', loss='mse')
model.fit([[1, 2], [2, 3], [3, 4]], [3, 5, 7],
          epochs=10, verbose=0)
model.save('my_model.keras')
loaded_model = tf.keras.models.load_model('my_model.keras')
print("\nQ10 Loaded Model:", loaded_model.predict([[4, 5]]))


# Q11. EarlyStopping and ModelCheckpoint
x = np.random.rand(100, 2)
y = np.sum(x, axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras',
                                       save_best_only=True)
]
model.fit(x, y, epochs=50, validation_split=0.2,
          callbacks=callbacks, verbose=0)
print("\nQ11 Training completed")


# Q12. Dropout Network
x = np.random.rand(200, 10)
y = np.random.randint(0, 2, 200)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=20, validation_split=0.2, verbose=0)
print("\nQ12 Dropout training completed")


# Q13. Functional API with Multiple Inputs and Outputs
input1 = tf.keras.Input(shape=(3,))
input2 = tf.keras.Input(shape=(3,))
combined = tf.keras.layers.Concatenate()([input1, input2])
hidden = tf.keras.layers.Dense(8, activation='relu')(combined)
output1 = tf.keras.layers.Dense(1)(hidden)
output2 = tf.keras.layers.Dense(1)(hidden)

model = tf.keras.Model(inputs=[input1, input2],
                       outputs=[output1, output2])
model.compile(optimizer='adam', loss='mse')

x1 = np.random.rand(100, 3)
x2 = np.random.rand(100, 3)
y1 = np.random.rand(100, 1)
y2 = np.random.rand(100, 1)

model.fit([x1, x2], [y1, y2], epochs=5, verbose=0)
print("\nQ13 Multiple input-output model trained")


# Q14. Hyperparameter Tuning using Keras Tuner
import keras_tuner as kt

x = np.random.rand(200, 4)
y = np.sum(x, axis=1)

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            hp.Int('units', 16, 64, step=16),
            activation='relu',
            input_shape=(4,)
        ),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        loss='mse'
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=3,
    directory='tuner_dir',
    project_name='tuning'
)
tuner.search(x, y, epochs=5, validation_split=0.2)
print("\nQ14 Best Parameters:",
      tuner.get_best_hyperparameters()[0].values)


# Q15. Plot Training and Validation Curves
x = np.random.rand(200, 4)
y = np.random.randint(0, 2, 200)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x, y, epochs=20, validation_split=0.2,
                    verbose=0)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()
