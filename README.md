# Dogs vs. Cats Classification

This project is a binary image classification model for distinguishing between images of dogs and cats. The model is built using TensorFlow and Keras, and the dataset used is the "Dogs vs. Cats" dataset from Kaggle.

## Overview

- **Dataset**: The project uses the "Dogs vs. Cats" dataset available on [Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats).
- **Model**: A Convolutional Neural Network (CNN) is implemented to classify images into two categories: dogs and cats.
- **Improvements**: Techniques such as dropout and batch normalization are applied to mitigate overfitting and improve the model's generalization.

## Features

- **Data Download**: Automated downloading and extraction of the dataset.
- **Data Preprocessing**: Normalization and resizing of images.
- **Model Building**: Two CNN architectures are built and trained—one baseline and one with dropout and batch normalization to reduce overfitting.
- **Model Evaluation**: Plots of training and validation accuracy and loss to evaluate model performance.
- **Testing**: Predictions on new images to test the model's performance.

## Setup

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Kaggle API key (for downloading the dataset)

### Installation

Install the required Python packages:

```bash
pip install tensorflow keras opencv-python matplotlib
```

### Kaggle API Key

To download the dataset from Kaggle, you need a Kaggle API key. Place the `kaggle.json` file containing your API credentials in the same directory as your script or in the `~/.kaggle/` directory.

## Usage

1. **Download the Dataset**: The dataset is downloaded and extracted using the Kaggle API. Run the following commands in a Colab notebook or a script:

    ```python
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !kaggle datasets download -d salader/dogs-vs-cats
    ```

2. **Extract the Dataset**:

    ```python
    import zipfile
    zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
    zip_ref.extractall('/content')
    zip_ref.close()
    ```

3. **Data Preparation**:

    ```python
    import tensorflow as tf
    from tensorflow import keras

    train_ds = keras.utils.image_dataset_from_directory(
        directory='/content/train',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 256)
    )

    validation_ds = keras.utils.image_dataset_from_directory(
        directory='/content/test',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 256)
    )

    def process(image, label):
        image = tf.cast(image / 255., tf.float32)
        return image, label

    train_ds = train_ds.map(process)
    validation_ds = validation_ds.map(process)
    ```

4. **Train the Model**:

    - **Baseline Model**:

        ```python
        from keras import Sequential
        from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
        ```

    - **Improved Model with Dropout and Batch Normalization**:

        ```python
        from keras.layers import BatchNormalization, Dropout

        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
        ```

5. **Evaluate the Model**:

    ```python
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], color='red', label='train')
    plt.plot(history.history['val_accuracy'], color='blue', label='validation')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], color='red', label='train')
    plt.plot(history.history['val_loss'], color='blue', label='validation')
    plt.legend()
    plt.show()
    ```

6. **Test the Model**:

    ```python
    import cv2

    test_img = cv2.imread('/content/cat.jpeg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (256, 256))
    test_input = test_img.reshape((1, 256, 256, 3))

    prediction = model.predict(test_input)
    print("Prediction:", "Cat" if prediction[0][0] < 0.5 else "Dog")
    ```

## Results

The project includes training and validation accuracy and loss plots to evaluate the model's performance. The improved model with dropout and batch normalization shows better generalization and reduced overfitting compared to the baseline model.

## Future Work

- **Data Augmentation**: Implementing data augmentation to further enhance model performance and reduce overfitting.
- **Hyperparameter Tuning**: Experimenting with different hyperparameters and model architectures.
- **Deployment**: Creating a web application or mobile app to deploy the model for real-time predictions.
