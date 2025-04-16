#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize mediapipe
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    """
    Process an image through MediaPipe Holistic model.
    Args:
        image: Input image in BGR format
        model: MediaPipe Holistic model instance
    Returns:
        image: Processed image
        results: MediaPipe detection results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe detection results.
    Args:
        results: MediaPipe detection results
    Returns:
        np.array: Flattened array of keypoints
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Define actions/classes
actions = np.array(['hello', 'namaste', 'bye', 'india', 'thanks', 'sorry', 'good', 'yes', 'no', '_'])

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,1662))) # 30 frames, 1662 keypoints
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('model.h5')

