'''
Team Id : 1714
Author List : Shreyas, Srujan, Neeraj
Filename: image_processing.py
Theme: Geo-Guide
Functions: detect_ArUco_details, classify_with_threshold, predict_events, contour
Global Variables: prediction_threshold, transform
'''
import cv2
import numpy as np
import csv
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision import transforms
import os
from file_operations import write_dict
# prediction_threshold: Confidence level threshold used for predicting the output of the CNN's output layer.
prediction_threshold = 0.7

# transform: Composition of image transformations including conversion to PyTorch tensor and normalization.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class CNN(nn.Module):
    '''
    Purpose:
    Defines a Convolutional Neural Network (CNN) class for image classification tasks.

    Input Arguments:
    num_classes : [int]
    Number of classes for classification.

    Returns:
    model

    Example instantiation:
    model = CNN(num_classes= 6)
    '''
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear( 20736, 128)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 20736)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
'''
Purpose:
Classifies an image tensor using the provided model with a specified confidence threshold.

Input Arguments:
model : [torch.nn.Module]
The trained model used for image classification.

img_tensor : [torch.Tensor]
The input image tensor to be classified.

threshold : [float]
The confidence threshold for classification.

Returns:
predicted_label : [int or None]
The predicted label index if the maximum probability exceeds the threshold, otherwise None.

max_probability : [float]
The maximum probability value among the predicted classes.

Example call:
label, probability = classify_with_threshold(model, input_image_tensor, 0.8)
'''
def classify_with_threshold(model, img_tensor, prediction_threshold):
    outputs = model(img_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)

    if max_prob.item() >= prediction_threshold:
        return predicted.item(), max_prob.item()
    else:
        return None, max_prob.item()

'''
Purpose:
Predicts events.

Input Arguments: None

Returns: [dict]
identified_labels: Dictonary of identified events.

Example call: identified_labels = predict_events()
'''
def predict_events():
    loaded_model = CNN(num_classes=6)
    loaded_model.load_state_dict(torch.load('fine_tuned_model_epoch_20.pth', map_location=torch.device('cpu')))
    loaded_model.eval()
    detected_events={}
    printing_format={}
    lable_name = ["Blank","combat", "destroyed_buildings", "fire", "humanitarian_aid", "military_vehicles"]
    class_names = ["","Combat", "Destroyed buildings", "Fire", "Humanitarian Aid and rehabilitation", "Military Vehicles"]
    for filename in os.listdir('./events'):
        image_path = os.path.join('./events', filename)
        key = filename.split('.')[0]
        image = cv2.imread(image_path)
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
        resize_transform = transforms.Resize((75, 75), interpolation=Image.BILINEAR)
        pil_img = resize_transform(pil_img)
        img_tensor = transform(pil_img).unsqueeze(0)
        lable, model_confidence = classify_with_threshold(loaded_model, img_tensor,prediction_threshold)
        if model_confidence >= prediction_threshold:
            if lable !=0:
                detected_events[key] = lable_name[lable]
                printing_format[key] = class_names[lable]
    write_dict('detected_events',detected_events)
    print(printing_format)
    return detected_events

'''

Purpose:
Detects and extracts white frames from the input image using contour detection.

Input:
image : [numpy.ndarray]
    The input image containing white frames to be detected.

Output:
white_frames : [dict]
    Dictionary containing coordinates of detected white frames.

Logic:
This function first converts the input image to grayscale and applies Gaussian blur to reduce noise.
It then thresholds the blurred image to create a binary image where white areas represent the frames.
Morphological operations (erosion and dilation) are applied to further refine the binary image.
Connected component analysis is performed to detect individual white frames.
Frames with areas between 5000 and 6400 pixels are considered valid and stored in a dictionary.
The coordinates of the detected frames are saved to a file named 'white_frames'.

Example Call:
white_frames = contour(image)
'''

def contour(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    threshold_value = 150
    _, thresholded_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    num_iterations = 2
    for i in range(num_iterations):
        thresholded_image = cv2.erode(thresholded_image, kernel, iterations=1)
        thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image, connectivity=4)
    white_frames={}
    event_alphabet=['E','D','C','B','A']
    i=0
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if 5000 < area < 6400:
            white_frames[event_alphabet[i]]=[x,y,w,h]
            i+=1
    write_dict('white_frames',white_frames)

    return white_frames

