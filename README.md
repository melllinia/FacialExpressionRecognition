# Facial Emotion Recognition

![License](https://img.shields.io/github/license/melllinia/FacialExpressionRecognition)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![Issues](https://img.shields.io/github/issues/melllinia/FacialExpressionRecognition)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Introduction

Facial Emotion Recognition is a project aimed at identifying human emotions from facial expressions in images or video sequences using deep learning techniques. The emotions recognized include happiness, sadness, surprise, anger, fear, disgust and neutral.

## Features

- Pre-trained models for quick setup
- Easy to train on custom datasets
- REST API for easy integration and testing
- Supports images
- Detailed results visualization

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/melllinia/FacialExpressionRecognition.git
    cd FacialExpressionRecognition/source
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip3 install -r requirements.txt
    ```

## Usage

### Running the REST API

To start the REST API server, run:

```sh
uvicorn server.controllers::app
```

The APIs will be available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
The Swagger UI will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### API Endpoints

**Emotion Recognition from Image with Probabilities Response**

- **Endpoint:** `/model/detect-emotion/`
- **Method:** POST
- **Response:** The list of probabilities of emotions with corresponding face coordinates 

**Emotion Recognition from Image with Image Response**

- **Endpoint:** `/model/detect-emotion/image`
- **Method:** POST
- **Response:** Annotated image



## Dataset

The model can be trained on various facial emotion datasets such as FER2013, CK+, etc. Make sure to download and place the dataset in the appropriate directory and update the path in the configuration file.

## Model Architecture

The model uses a convolutional neural network (CNN) based architecture for feature extraction and emotion classification. The architecture can be customized by modifying the `model/net.py` file.

## Results

Our pre-trained model achieves the following accuracy on the FER2013 dataset:
- Accuracy: 55%

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch-name`.
5. Open a pull request.

Please make sure your code follows the project's coding standards and includes proper documentation.

## Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

## Contact

For any queries, please contact [meline.mkrtchyan1@gmail.com](mailto:meline.mkrtchyan1@gmail.com) or [ghovhannes19@gmail.com](mailto:ghovhannes19@gmail.com).

---

*Keep innovating!* 💡🚀
