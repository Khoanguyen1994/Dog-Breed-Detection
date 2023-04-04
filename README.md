# Dog Breed Classification
This is the capstone project for the Udacity Data Science Nanodegree course. The project aims to identify the breed of a dog in an image using computer vision techniques. It also provides an estimate of the person that most resembles the image if a person is detected.

## Content:
- [Project Description](#project-description)
- [Files](#files)
- [Installation Instruction](#installation-instruction)
- [Usage Instruction](#usage-instruction)
- [License](#license)

## Project Description

### Project Overview
This project uses a convolutional neural network (CNN) to identify the breed of a dog in an image with the highest possible accuracy. The project evaluates different models such as VGG16, ResNet50, and a custom-built model. The trained model is used to predict the breed of a dog in any user-supplied image.

### Problem Statement
Identifying dog breeds is a challenging task, especially when dealing with images of dogs with similar features. The project aims to train and evaluate a CNN model that takes an RGB image of a dog as input and predicts the breed with the highest possible accuracy.

### Metrics
The evaluation metric for this project is accuracy since it is a classification problem. The model is evaluated on the test set, and the result is reported in percentage.


## Files
The repository contains the following files:

```
Dog_breed_detection
│   README.md
│   requirements.txt // required packages
|   LICENSE.txt
│
└───app
│   │   dog_app.py
│   │   run.py
│   │
│   └───upload_folder // contains images uploaded from users
│   │
│   └───templates // contains HTML template files for the web app.
│       │   index.html
│   
└───bottleneck_features
│   │   DogVGG16Data.npz
│   │   DogResnet50Data.npz
│
└───haarcascades
│   │   haarcascade_frontalface_alt.xml // model to detect human faces
│
└───notebook
│   ├───images
│   ├───dog_app.ipynb
│   └───extract_bottleneck_features.py
│
└───saved_models // contains saved data for dog breed identification model developed in the notebook
│   │   weights.best.from_scratch.hdf5
│   │   weights.best.Resnet50.hdf5
│   │   weights.best.VGG16.hdf5
```

## Installation Instruction
1. Clone or download the repository.
2. Create a new environment with Python version 3.6 or higher.
3. Install the required packages using `pip install -r requirements.txt`

## Usage Instruction
1. After installing the packages, navigate to the app directory.
2. Run python run.py.
3. Open a web browser and go to `http://127.0.0.1:8080/`
4. Upload an image of a dog or a person.
5. The application will predict the breed of the dog or the person that most resembles the image.

## License
This project is licensed under the MIT License. See the LICENSE.txt file for details.
