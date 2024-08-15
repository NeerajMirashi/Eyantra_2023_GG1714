'''
Team Id : 1714
Author List : Shreyas
Filename: lat_lon_model_train.py
Theme: Geo-Guide
Functions: write_xy
Global Variables: 
Purpose: Get x,y coordinates of all arucos and train the model which will predict the lat lon for each pixel where for each aruco respective lat lon is already georeferenced once.
Note: This file is executed only when we change the camera setup to train our model for lat lon prediction.
'''
import csv
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from aruco_processing import detect_ArUco_details

'''
Purpose:
Writes ArUco marker center coordinates to a CSV file.

Input Arguments:
d : [dict]
    Dictionary containing ArUco marker IDs as keys and their respective center coordinates as values.

Returns:
None

Logic:
This function writes the ArUco marker center coordinates to a CSV file named 'x_y_aruco.csv'.
It first removes entries with invalid coordinates (containing -1) from the dictionary.
Then, it writes the ID and center coordinates (x, y) for each marker as rows in the CSV file.

Function Name: write_xy
'''
def write_xy(d):
    with open('x_y_aruco.csv', 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f)
        d = {key: values for key, values in d.items() if -1 not in values}
        csvwriter.writerows([key] + values for key, values in d.items())

# Load the input image
image1 = cv2.imread('./arena.png')

# Detect ArUco markers and write their center coordinates to a CSV file
d, _ = detect_ArUco_details(image1)
write_xy(d)

# Draw circles at the detected ArUco marker positions on the image
for key, val in d.items():
    cv2.circle(image1, (val[0], val[1]), 3, (0, 255, 0), -1)

# Read ArUco marker center coordinates from the CSV file
df1 = pd.read_csv('x_y_aruco.csv', header=None, names=['id', 'x', 'y'])

# Read the latitude and longitude CSV file containing ArUco marker details
df2 = pd.read_csv('lat_lon_aruco.csv', header=None, names=['id', 'lat', 'lon'])

# Merge the dataframes based on the marker ID
merged_df = pd.merge(df1, df2, on='id')

# Split the dataset into features (X) and target variables (y)
X = merged_df[['x', 'y']]
y = merged_df[['lat', 'lon']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'latlonModel1.pkl')

# Display the merged dataframe and the image with detected ArUco markers
print(merged_df)
cv2.imshow('image', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
