'''
Team Id : 1714
Author List : Shreyas, Aparna
Filename: aruco_processing.py
Theme: Geo-Guide
Functions: detect_ArUco_details
Global Variables: None
'''

from cv2 import aruco
import cv2
import numpy as np

'''
Purpose:
Detects ArUco markers in the provided image and retrieves their details.

Input Arguments:
image : [numpy.ndarray]
The input image containing ArUco markers.

Returns:
ArUco_details_dict : [dict]
A dictionary mapping ArUco marker IDs to their center coordinates.

ArUco_corners : [dict]
A dictionary mapping ArUco marker IDs to their corner coordinates.

Example call:
details, corners = detect_ArUco_details(input_image)
'''
def detect_ArUco_details(image):
    ArUco_details_dict1 = {}
    for i in range(0,101):
        ArUco_details_dict1[i]=[-1,-1]
    ArUco_corners = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000) 
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary = aruco_dict,detectorParams = parameters)
    a,b,_ = detector.detectMarkers(gray)
    if np.all(b is not None):  
        for i in range(len(b)):
            c = a[i][0]
            center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
            center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
            center = [center_x, center_y]
            id =int(b[i][0])
            if id in ArUco_details_dict1:
                ArUco_details_dict1[id] = center
                ArUco_corners[id] = c
    return ArUco_details_dict1, ArUco_corners 


