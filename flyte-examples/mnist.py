import flytekit
from flytekit import task, workflow, Resources

from typing import List, Tuple
import numpy as np
import os

# Set the MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = "http://d3x-controller.d3x.svc.cluster.local:5000"

# Define Flyte tasks
@task(requests=Resources(cpu="2", mem="1Gi"))
# Import TensorFlow and other necessary libraries
def download_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    # Download the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    # Normalize and preprocess the data
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels, test_images, test_labels)

@task(requests=Resources(cpu="2", mem="1Gi"))
# Import TensorFlow and other necessary libraries
def train_model(train_images: np.ndarray, train_labels: np.ndarray,test_images: np.ndarray,test_labels: np.ndarray) -> str:
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    import mlflow
    import mlflow.tensorflow
    # Create and train a simple convolutional neural network (CNN)
    train_images = train_images
    train_labels = train_labels
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    # Serialize and save the model to a file
    #model.save('mnist_model.h5')
    #mlflow.tensorflow.log_model(model,"model")
    # Load the trained model
    test_images = test_images
    test_labels = test_labels
    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    # Log accuracy and precision
    accuracy = f'Accuracy: {test_acc}'
    test_losss = f'test_loss: {test_loss}'
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model,"mnist_model")
        mlflow.log_metric("accuracy",float(test_acc))
        mlflow.log_metric("loss",float(test_loss))
    return f'{accuracy}\n{test_loss}'

@workflow
def mnist_workflow() -> str:
    mnist_data=download_mnist_data()
    accuracy= train_model(train_images=mnist_data[0],train_labels=mnist_data[1],test_images=mnist_data[2],test_labels=mnist_data[3])
    return accuracy
