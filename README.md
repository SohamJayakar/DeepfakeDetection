# Deepfake Detection

This repository contains code for deepfake detection using machine learning models.

## Dataset

To train the deepfake detection model, you need to download the dataset. We have used the Deepfake Detection Challenge dataset from Kaggle. You can find the dataset [here](https://www.kaggle.com/c/deepfake-detection-challenge/data).

## Preprocessing

Before training the model, it is necessary to preprocess the videos. You can preprocess the dataset using the `preprocessing.ipynb` notebook provided in this repository.

## Model Training

After preprocessing the dataset, you can train the deepfake detection model using the `model_train.ipynb` notebook.

## Prediction

Once the model is trained, you can use it to predict the authenticity of videos. You can either use the `predict.ipynb` notebook for prediction or use the Streamlit web app `app.py` provided in this repository.

## Streamlit Web App

To run the Streamlit web app, execute the following command:

```bash
streamlit run app.py
