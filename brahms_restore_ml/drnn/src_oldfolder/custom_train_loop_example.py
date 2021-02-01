# Example from (starting at):
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#using_the_gradienttape_a_first_end-to-end_example

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Model
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)


# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


# # Training loop - Bare-bones example
# epochs = 2
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

#         # Open a GradientTape to record the operations run
#         # during the forward pass, which enables auto-differentiation.
#         with tf.GradientTape() as tape:

#             # Run the forward pass of the layer.
#             # The operations that the layer applies
#             # to its inputs are going to be recorded
#             # on the GradientTape.
#             logits = model(x_batch_train, training=True)  # Logits for this minibatch

#             # Compute the loss value for this minibatch.
#             loss_value = loss_fn(y_batch_train, logits)

#         # Use the gradient tape to automatically retrieve
#         # the gradients of the trainable variables with respect to the loss.
#         grads = tape.gradient(loss_value, model.trainable_weights)

#         # Run one step of gradient descent by updating
#         # the value of the variables to minimize the loss.
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         # Log every 200 batches.
#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %s samples" % ((step + 1) * 64))



# Spped up execution w/ tf.function - not for debugging (eager execution)
# I CHANGED: USE LOSS AS THE MAIN METRIC
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    # val_acc_metric.update_state(y, val_logits)
    return loss_value

# Example w/ tf.functions
# import time
epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    # start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d training samples" % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    # train_acc = train_acc_metric.result()
    # print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    # train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        loss_value = test_step(x_batch_val, y_batch_val)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Validation loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d val samples" % ((step + 1) * 64))

    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    # print("Time taken: %.2fs" % (time.time() - start_time))