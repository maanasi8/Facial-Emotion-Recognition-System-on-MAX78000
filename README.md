# FACIAL EMOTION RECOGNITION
# PHASE 1
## Introduction
Facial emotion recognition(FER) is the process of detecting human emotions from facial expressions using computer vision and machine learning/ deep learning algorithms. Understanding emotions can help improve human-machine interactions, personalized digital marketing, and mental health diagnosis. Facial recognition systems use features such as eye movements, facial muscle movements, and lip position to classify emotions. Recognition of human emotions is a vital phase, which is involved in several applications such as augmented and virtual reality, advanced driver assistance systems, human-computer interaction, and security systems. Due to the growing demand for Facial Emotion Recognition (FER) in recent times, the current study deals with the identification of human emotions.
## Problem Statement
Develop a synthesizable AI model to perform image classification on facial emotions using a deep neural network.
## Objectives
1.  Preprocessing of the images.
2.  Building and training the classification model to classify the facial emotions of humans.
3.  Testing or performance analysis of the model.
4. Comparing the synthesizable model's accuracy and efficiency with state-of-the-art techniques for facial emotion recognition.
## Scope & Constraint
1. The dataset is constrained to 7 emotion classes.
2. The dataset consists of grey-scale images only. Hence, it inherently demands less computational power and is more energy-efficient compared to datasets involving color images.
## Software Engineering Requirements
### Functional requirements:
1. The user shall be able to input the grayscale facial images for emotion recognition.
2. The system shall classify a range of emotions such as happiness, sadness, anger, fear, etc., from the input images.
3. The system shall predict the emotions and generate scores/accuracies for all the emotion classes.
4. The user shall be able to view the performance analysis.
### Non-Functional requirements:
1. The system should achieve a response time of less than 2 seconds for emotion recognition processing, ensuring minimal delay in emotion recognition to enhance user experience.
2. The emotion recognition system should achieve a minimum accuracy of 70% on standardized emotion recognition benchmarks.
## Dataset description
### Facial emotion recognition
https://www.kaggle.com/datasets/chiragsoni/ferdata<br />
Dataset size: 56.51MB<br />
The dataset contains 35,914 grayscale images of faces.<br />
Testing images – 20.06% [7,205 images]<br />
Training images – 79.93% [28,709 images]<br />
The dataset consists of 7 classes. They are- Happy, sad, disgust, angry, fear, neutral, and surprise.<br /><br />
The number of images for each feature in train dataset is as follows- <br />
angry – 4953 images<br />
disgust - 547 images<br />
fear - 5121 images<br />
Happy - 8989 images<br />
neutral - 6198 images<br />
sad - 6077 images<br />
surprise - 4002 images<br /><br />
The number of images for each feature in test dataset is as follows- <br />
angry – 958 images<br />
disgust - 111 images<br />
fear - 1024 images<br />
Happy - 1774 images<br />
neutral - 1233 images<br />
sad - 1247 images<br />
surprise - 831 images<br /><br />
The images in the dataset are approximately of the size 2KB each.<br />
## Intermediate Results:
![Figure_1](https://github.com/maanasi8/Mini-Project/assets/126388400/9963e409-ae8c-46d0-9c70-a9cc3f5d6b1d)

## Results:
![1](https://github.com/maanasi8/Mini-Project/assets/126388400/438cbde1-6ba7-4d06-9cb3-3e1e118d7d23)

<img width="1000" alt="image" src="https://github.com/maanasi8/Mini-Project/assets/126388400/0e915a74-c0b4-4606-8333-47cdcfa42874">

# PHASE 2
## Introduction
* Integrating AI and deep learning into embedded systems has enabled advancements in Facial Emotion Recognition (FER) for various applications.
* Practical deployment of FER systems faces challenges, particularly in resource-constrained environments like edge devices.
* Developing deployable FER models optimized for ultra-low power embedded systems is essential.
* Successful deployment of this model will unlock new possibilities for human-computer interaction, mental health monitoring, and consumer behavior analysis.

## Motivation
Facial Emotion Recognition(FER) plays a major role in:
* Improving human-computer interaction, allowing dynamic system responses based on user emotions.
* Monitoring and assessing mental health conditions through facial emotion analysis.
* Analyze consumer reactions for targeted and effective advertising for marketing strategies.
* The increasing demand, interest, and adoption of facial emotion recognition systems was the motivation to delve into this subject for our study.

## Problem Statement
**Develop a deployable Facial Emotion Recognition model on an ultra-low power embedded system.**
## Objectives
* Design of a Deep Neural network-based energy-efficient Facial Emotion Recognition Model.
* Development of the embedded system for the efficient deployment of the Facial Emotion Recognition Model.

## Approach
1. Model Development: Design the model with PyTorch or TensorFlow-Keras.
2. Training: Train with floating-point weights, then quantize for MAX78000 deployment.
3. Model Evaluation: Assess quantized model accuracy using an evaluation dataset.
4. Synthesis Process: Use the MAX78000 Synthesizer tool to generate optimized C code from ONNX files, YAML model description, and input data. The tool generates C code for loading weights, performing inference, and validating results.

## Flow diagram for Facial Emotion Recognition System
![image](https://github.com/maanasi8/Mini-Project/assets/126388400/b16e3363-eb8d-475a-bb57-c37d8cf62676)

## Development Board
### MAXIM78000 FTHR BOARD:
![image](https://github.com/maanasi8/Mini-Project/assets/126388400/380cfd32-0422-4b82-b206-d0bdb0068ec1)

* Dual-Core Ultra-Low-Power Microcontroller
* Power Management Maximizes Operating Time for Battery Applications
* 12-bit Parallel Camera Interface
* The CNN accelerator also has 512KB of data memory
* Input Image Size up to 1024 x 1024 pixels
* 52 General-Purpose I/O Pins
* 512KB Flash and 128KB SRAM
