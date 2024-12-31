# Emotion Recognition Telegram Bot

This project provides a Telegram bot that predicts emotions from uploaded photos of faces.  
You can try the bot here: [Bot link](https://t.me/emotionscanbot)

## Contents

1. [Description](#description)
2. [Results](#results)
3. [Features](#features)
4. [Project Structure](#project-structure)

## Description

The Telegram bot accepts images with faces and determines the emotion, returning the result as a text message. The emotion prediction model is based on the ResNet50 architecture, pre-trained on VGGFACE2 and further fine-tuned on FER2013+.

## Results

The model achieved an **accuracy of 84%** on the test set of FER2013+. 

## Features

- Accepts images from the user
- Recognizes emotions on faces in the images
- Sends the predicted emotion back as a text message

## Project Structure

The project consists of the following files:

- `predict.py`: Contains the function for emotion prediction from the image.
- `main.py`: Runs the Telegram bot and handles user requests.
- `Facial_expression_recognition.ipynb`: Notebook with model training, where experiments were conducted.
- `requirements.txt`: List of project dependencies.





