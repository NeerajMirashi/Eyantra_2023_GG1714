'''
Team Id : 1714
Author List : Shreyas, Srujan, Neeraj
Filename: main.py
Theme: Geo-Guide
Functions: save_crop_coordinates, get_croped_frame, mark_eventzones, capture_events, get_vangaurd_position, update_vangaurd_lat_lon_position, is_inside_rectangle, is_outside_after_inside, sort_events, encode_and_send_event_list, monitor_vangaurd, get_client_socket, show, init
Global Variables: model, client_socket, sorted_events, stop_roi, event_area
'''
import cv2
from aruco_processing import *
import time
from image_processing import *
from file_operations import *
import joblib
import numpy as np
import socket

#model: Logistic regression model to calculate the lat lon for corresponding x y coordinates.
global model
#client_socket: socket.socket
global client_socket
#sorted_events: stored event keys
global sorted_events
#stop_roi: stores coordinate of end area
global stop_roi
#event_area: the
global event_area

# camera settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

'''
Purpose:
Detects corner ArUco markers in the input frame and defines a square frame around the arena based on these markers. 
Calculates coordinates of the top, bottom, left, and right edges of the frame with additional margins, and saves these coordinates to a file.

Input:
frame : [numpy.ndarray]
    The input frame containing ArUco markers.

Output:
None

Logic:
This function detects the corner ArUco markers in the input frame and calculates a square frame around the arena based on these markers. 
It extracts the coordinates of the top-left, bottom-left, top-right, and bottom-right corners of the arena. 
Margin values are added to these coordinates to ensure the entire arena is captured within the frame. 
The resulting crop coordinates are stored in a dictionary. 
Finally, the crop coordinates are written to a file named 'arena_frame_crop_coordinates.txt'.

Example Call:
save_crop_coordinates(frame)
'''
def save_crop_coordinates(frame):
    # Detect ArUco markers in the frame
    ArUco_details_dict, _ = detect_ArUco_details(frame)
    
    # Extract coordinates of ArUco markers for the corners of the arena
    top_left_x, top_left_y = ArUco_details_dict[7]  # ArUco marker 7 represents top-left corner
    bottom_left_x, bottom_left_y = ArUco_details_dict[6]  # ArUco marker 6 represents bottom-left corner
    top_right_x, top_right_y = ArUco_details_dict[5]  # ArUco marker 5 represents top-right corner
    bottom_right_x, bottom_right_y = ArUco_details_dict[4]  # ArUco marker 4 represents bottom-right corner
    
    # Define crop margins around the detected markers to capture the entire arena
    crop_top = min(top_left_y, top_right_y) - 30
    crop_bottom = max(bottom_left_y, bottom_right_y) + 30
    crop_left = min(top_left_x, bottom_left_x) - 20
    crop_right = max(bottom_right_x, top_right_x) + 40
    
    # Store crop coordinates in a dictionary
    crop_coordinates = {
        "top": crop_top,
        "bottom": crop_bottom,
        "left": crop_left,
        "right": crop_right
    }
    
    # Write crop coordinates to a file for further processing
    write_dict('arena_frame_crop_coordinates.txt', crop_coordinates)

'''
Purpose:
Retrieves a cropped and processed frame of the arena from the input frame based on pre-defined crop coordinates.

Input:
frame : [numpy.ndarray]
    The input frame containing the arena.

Output:
resized_frame : [numpy.ndarray]
    The cropped, rotated, and resized frame of the arena.

Logic:
This function reads pre-defined crop coordinates from a file named 'arena_frame_crop_coordinates.txt'.
It then extracts the relevant portion of the input frame using these coordinates to obtain a cropped frame of the arena.
The cropped frame is rotated 90 degrees counterclockwise using OpenCV's 'cv2.rotate' function.
Finally, the rotated frame is resized to a specified dimension (970x970) using OpenCV's 'cv2.resize' function.
The resulting resized frame is returned as the output.

Example Call:
cropped_frame = get_croped_frame(frame)
'''
def get_croped_frame(frame):
    crop_coordinates = read_dict('arena_frame_crop_coordinates.txt')
    cropped_frame = frame[crop_coordinates['top']:crop_coordinates['bottom'], crop_coordinates['left']:crop_coordinates['right']]
    rotated_frame = cv2.rotate(cropped_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resized_frame = cv2.resize(rotated_frame, (970, 970))
    return resized_frame

'''
Purpose:
Marks event zones on the input frame based on pre-defined coordinates and event labels.

Input:
frame : [numpy.ndarray]
    The input frame on which event zones are to be marked.

Output:
None

Logic:
This function reads pre-defined event coordinates and labels from a file named 'white_frames'.
It iterates over each detected event zone and its corresponding label.
For each event zone, it draws a rectangle around it using OpenCV's 'cv2.rectangle' function with green color and thickness of 2.
Additionally, it adds text displaying the event label above each rectangle using OpenCV's 'cv2.putText' function.
The marked frame is updated in-place, and no explicit output is returned.

Example Call:
mark_eventzones(frame)
'''
def mark_eventzones(frame):
    event_coordinate = read_dict('white_frames')
    for key,value in detected_events.items():
            x,y,w,h=event_coordinate[key]
            event_lable = value
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, f"{event_lable}", (x , y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

'''

Purpose:
Captures event areas from the input frame and saves them as individual images for prediction.

Input:
frame : [numpy.ndarray]
    The input frame containing event areas.

Output:
None

Logic:
This function reads event areas (coordinates) from a file named 'white_frames'.
It iterates over each event area and its corresponding key.
For each event area, it extracts the corresponding portion of the input frame using the provided coordinates.
The extracted area is then saved as an individual image file in the './events/' directory using OpenCV's 'cv2.imwrite' function.
No explicit output is returned.

Example Call:
capture_events(frame)
'''

def capture_events(frame):
    event_area = read_dict('white_frames')
    for key,val in event_area.items():
        x,y,w,h=val
        image=frame[y:y+h,x:x+w]
        cv2.imwrite(f'./events/{key}.png',image)

'''

Purpose:
Retrieves the position of the vanguard ArUco marker from the input frame.

Input:
frame : [numpy.ndarray]
    The input frame containing ArUco markers.

Output:
vanguard_position : [tuple]
    The position (x, y) of the vanguard ArUco marker.

Logic:
This function detects ArUco markers in the input frame using the 'detect_ArUco_details' function.
It then retrieves the position of the vanguard ArUco marker (marker ID 100) from the detected markers dictionary.
The position (x, y) of the vanguard ArUco marker is returned as output.

Example Call:
vanguard_position = get_vangaurd_position(frame)
'''

def get_vangaurd_position(frame):
    ArUco_details_dict,_=detect_ArUco_details(frame)
    return ArUco_details_dict[100]


'''

Purpose:
Updates the latitude and longitude position of the vanguard based on its current position.

Input:
vangaurd_position : [tuple]
    The current position (x, y) of the vanguard ArUco marker.

Output:
None

Logic:
This function extracts the x and y coordinates from the input vanguard position tuple.
If the position is (-1, -1), indicating that the marker is not detected, the function returns without further processing.
Otherwise, it creates a numpy array containing the source coordinates (x, y).
It then uses a logistic regression model to predict the latitude and longitude values based on the source coordinates.
The predicted latitude and longitude values are written to a CSV file named 'live_location.csv'.

Example Call:
update_vangaurd_lat_lon_position(vangaurd_position)
'''

def update_vangaurd_lat_lon_position(vangaurd_position):
    x,y=vangaurd_position
    if x==-1 and y==-1:
        return
    source_xy=np.array([[x,y]])
    lon,lat=model.predict(source_xy)[0]
    write_csv(lat,lon,'live_location.csv')

'''
Purpose:
Checks if a given point is inside a rectangle defined by its four corner coordinates.

Input Arguments:
center : [tuple]
The coordinates of the point to be checked.

rectangle : [tuple]
The coordinates of the top_left corner of the rectangle and width and height.

Returns:
True if the point is inside the rectangle, False otherwise.

Example call:
is_inside = is_inside_rectangle((x, y), (rect_top_left_x,rect_top_left_y,w,h))
'''
def is_inside_rectangle(center, rectangle):
    x, y = center
    if x==-1 and y==-1:
        return False
    rect_top_left_x,rect_top_left_y,w,h=rectangle
    return rect_top_left_x <= x <= rect_top_left_x+w and rect_top_left_y <= y <= rect_top_left_y+h

'''
Purpose:
Checks if a given point is outside a rectangle after being inside it, defined by its four corner coordinates.

Input Arguments:
center : [tuple]
The coordinates of the point to be checked.

rectangle : [tuple]
The coordinates of the top_left corner of the rectangle and width and height.

Returns:
True if the point is outside the rectangle after being inside it, False otherwise.

Example call:
is_outside = is_outside_after_inside((x, y), (rect_top_left_x,rect_top_left_y,w,h))
'''
def is_outside_after_inside(center,rectangle):
    x, y = center
    if x==-1 and y==-1:
        return False
    rect_top_left_x,rect_top_left_y,w,h=rectangle
    return rect_top_left_x > x or x > rect_top_left_x + w

'''
Purpose:
Sorts the identified events based on a predefined priority order.

Input Arguments:
None

Returns:
sorted_keys : [list]
A list of event keys sorted according to their priority order.

Example call:
sorted_events_keys = sort_events()
'''
def sort_events():
    priority_order = ["fire", "destroyed_buildings", "humanitarian_aid", "military_vehicles", "combat"]
    sorted_keys = sorted(detected_events.keys(), key=lambda x: (priority_order.index(detected_events[x]), -ord(x[0])))
    return sorted_keys

'''
Purpose:
Encodes and sends the sorted event list to the server via the client socket.

Input Arguments:
sorted_keys : [list]
A list of event keys sorted according to their priority order.

Returns:
None

Example call:
encode_and_send_event_list(sorted_keys)
'''

def encode_and_send_event_list(sorted_keys):
    encoding_dict = {'A': 'b', 'B': 'f', 'C': 'k', 'D': 'i', 'E': 'p'}
    encoded_string = 'z' + chr(ord('a') + len(sorted_keys) - 1)
    for key in sorted_keys:
        if key in encoding_dict:
            encoded_string += encoding_dict[key]
    client_socket.sendall(str.encode(encoded_string))

'''
Purpose:
Monitors the movement of the vanguard based on its position and predefined event zones.

Input:
vangaurd_position : [tuple]
    The current position (x, y) of the vanguard ArUco marker.

sorted_events : [list]
    A list of sorted event keys representing the sequence of events to be triggered.

Output:
None

Logic:
This function iterates over predefined event stop areas and checks if the vanguard is inside any of them.
If the vanguard is inside an event stop area and the signal flag is True, it updates the last frame and sets the signal flag to False.
If the sorted events list is not empty and the first event matches the current event, it sets the stop flag to True, removes the event from the sorted events list, and sends a corresponding signal to the client socket.
Otherwise, it sends a NODE signal to the client socket.
If the vanguard moves outside the last frame and the signal flag is False, it sets the signal flag to True.
If the stop flag is True and the vanguard is inside the stop ROI, it sets the stop flag to False and sends a STOP signal to the client socket.

Example Call:
monitor_vangaurd(vangaurd_position, sorted_events)
'''


def monitor_vangaurd(vangaurd_position, sorted_events):
    global signal_flag
    global stop_flag
    global last_frame
    for key,value in event_stop_area.items():
        if(is_inside_rectangle(vangaurd_position,value) and signal_flag):
            last_frame = value
            signal_flag = False
            if sorted_events and sorted_events[0] == key:
                stop_flag = True
                sorted_events.pop(0)
                client_socket.sendall(str.encode("ZONE"))
            else:
                client_socket.sendall(str.encode("NODE"))
    if last_frame:
        if(is_outside_after_inside(vangaurd_position,last_frame) and signal_flag==False):
            signal_flag = True
    if stop_flag:
        if(is_inside_rectangle(vangaurd_position,stop_roi)):
            stop_flag = False
            client_socket.sendall(str.encode("STOP"))

'''
Purpose:
Establishes a client socket connection to communicate with a server.

Returns:
client_socket : [socket.socket]
The client socket object for communication with the server.

Example call:
client_socket = get_client_socket()
'''

def get_client_socket():
    host = '192.168.231.108'  
    port = 65534
    # Create a TCP/IP socket
    client_socket = socket.socket()
    # Connect to the server (simulated ESP32)
    server_address = (host, port)
    client_socket.connect(server_address)
    return client_socket

'''
Purpose:
Displays the camera feed with marked event zones and monitors the movement of the vanguard.

Input:
None

Output:
None

Logic:
This function continuously reads frames from the camera feed and processes them.
It first captures the cropped frame using the 'get_croped_frame' function and marks event zones using the 'mark_eventzones' function.
The position of the vanguard is obtained using the 'get_vangaurd_position' function, and its latitude and longitude position are updated using the 'update_vangaurd_lat_lon_position' function.
Event areas are captured again using the 'capture_events' function, and the vanguard's movement is monitored using the 'monitor_vangaurd' function.
The processed frame is displayed in a window named 'GG_1714 Camera Feed'.
The loop continues until the user presses the 'q' key to exit.
Once the loop is exited, the camera feed is released, and all OpenCV windows are closed.

Example Call:
show()
'''

def show():
    window='GG_1714 Camera Feed'
    cv2.namedWindow(window)
    cv2.moveWindow(window,0,0)
    global last_frame
    last_frame=None
    global signal_flag
    signal_flag = True
    global stop_flag
    stop_flag = False
    while True:
        _,frame=cap.read()
        croped_frame=get_croped_frame(frame)
        mark_eventzones(croped_frame)
        vangaurd_position = get_vangaurd_position(croped_frame)
        update_vangaurd_lat_lon_position(vangaurd_position)
        capture_events(croped_frame)
        monitor_vangaurd(vangaurd_position,sorted_events)
        cv2.imshow(window,croped_frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

'''
Purpose:
Initializes global variables, loads necessary models, captures initial event data, and prepares for event monitoring.

Input:
None

Output:
None

Logic:
This function is called once at the beginning of the program execution.
It initializes global variables including the machine learning model, client socket, sorted events list, stop ROI, event areas, detected events, and event stop areas.
The machine learning model is loaded from the 'lat_lon.pkl' file using joblib.
The stop ROI is defined as a list containing the coordinates of the stop region of interest.
The function reads the current frame from the camera and saves crop coordinates using the 'save_crop_coordinates' function.
It then captures the cropped frame using the 'get_croped_frame' function and extracts event areas using the 'capture_events' function.
Detected events are predicted using the 'predict_events' function and event stop areas are read from the 'event_stop_zone' file.
The function then adjusts the event stop areas to create larger zones for better detection and writes them back to the file.
Events are sorted using the 'sort_events' function and a client socket is obtaiget_client_socketned using the '' function.
Finally, the sorted events list is encoded and sent to the client socket using the 'encode_and_send_event_list' function.

Example Call:
init()
'''

def init():
    global model
    global client_socket
    global sorted_events
    global stop_roi
    global event_area
    global detected_events
    global event_stop_area
    model = joblib.load('lat_lon.pkl')
    stop_roi = [80,820,70,70]
    _,frame=cap.read()
    time.sleep(1)
    _,frame=cap.read()
    save_crop_coordinates(frame)
    croped_image = get_croped_frame(frame)
    capture_events(croped_image)
    detected_events = predict_events()
    event_stop_area = read_dict('event_stop_zone')
    event_area = read_dict('white_frames')
    event_stop_zone={}
    for key,val in event_area.items():
        x,y,w,h = val
        event_stop_zone[key] = [x-65,y-120,w+160,h+30]
    write_dict('event_stop_zone',event_stop_zone)
    sorted_events = sort_events()
    client_socket = get_client_socket()
    encode_and_send_event_list(sorted_events)
    # contour(croped_image)

if __name__=="__main__":
    init()
    show()
  