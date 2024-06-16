# Facial Recognition Neural Network

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
<!-- - [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results) -->
<!-- - [Contributing](#contributing) -->
<!-- - [License](#license) -->

## Introduction
This project implements a facial recognition system using a convolutional neural network. The system can identify and verify individuals, specifically children, from images or video streams by comparing their facial features to a database of known faces. The application of this CNN will be in the education sector to enhance school security be maintaining a student attendance record levaraging existent CCTV systems in schools.

## Features
- **Face Detection**: Locates faces within an image.
- **Face Recognition**: Matches extracted features to known faces in the database.
- **Real-Time Recognition**: Recognizes faces in real-time from a video feed.

## Installation
To run this project, you'll need to have Python installed along with `poetry` for dependency management. Follow these steps to set up your environment:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/chiaraalcantara/secured
    cd secured
    ```

2. **Install `poetry`**:
    Install `poetry` using the official installation script:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    Alternatively, you can install it via pip:
    ```sh
    pip install poetry
    ```

3. **Install the dependencies**:
    Run the following command to install all the dependencies defined in the `pyproject.toml` file:
    ```sh
    poetry install
    ```

4. **Activate the virtual environment**:
    Use the following command to activate the virtual environment managed by `poetry`:
    ```sh
    poetry shell
    ```

5. **Verify the installation**:
    Ensure that all dependencies are installed correctly by checking the installed packages:
    ```sh
    poetry show
    ```


<!-- ## Usage
To use the facial recognition system, follow these steps:

1. **Prepare the dataset**: Place your training images in the `data/train` directory and your testing images in the `data/test` directory.

2. **Train the model**:
    ```sh
    python train.py
    ```

3. **Evaluate the model**:
    ```sh
    python evaluate.py
    ```

4. **Run real-time recognition**:
    ```sh
    python recognize.py
    ```

## Dataset
The dataset should be organized as follows:
```
data/
  ├── train/
  │   ├── person1/
  │   │   ├── image1.jpg
  │   │   ├── image2.jpg
  │   ├── person2/
  │   │   ├── image1.jpg
  │   │   ├── image2.jpg
  ├── test/
      ├── person1/
      │   ├── image1.jpg
      │   ├── image2.jpg
      ├── person2/
      │   ├── image1.jpg
      │   ├── image2.jpg
```

## Training
To train the neural network, use the `train.py` script. This script will load the training images, preprocess them, and train the neural network model.

```sh
python train.py
```

## Evaluation
To evaluate the trained model, use the `evaluate.py` script. This will test the model on the test dataset and print the accuracy and other metrics.

```sh
python evaluate.py
```

## Results
After training and evaluation, you can find the results and metrics in the `results` directory. This will include accuracy, loss plots, and other relevant data. -->

<!-- ## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->

