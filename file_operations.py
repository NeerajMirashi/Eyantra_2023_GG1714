'''
Team Id : 1714
Author List : Shreyas, Aparna
Filename: file_operations.py
Theme: Geo-Guide
Functions: read_dict, write_dict, write_dict_to_csv, write_csv
Global Variables: None
'''
import csv

'''
Purpose:
Reads a dictionary from a text file.

Input:
filename : [str]
    The name of the text file (without extension) containing the dictionary.

Output:
data : [dict]
    The dictionary read from the file.

Logic:
This function reads lines from the specified text file and evaluates them as Python dictionaries.
It then returns the resulting dictionary.

Example Call:
data = read_dict('filename')
'''

def read_dict(filename):
    with open(f'{filename}.txt', 'r') as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        data = eval(line)
    return data

'''
Purpose:
Writes a dictionary to a text file.

Input:
filename : [str]
    The name of the text file (without extension) to write the dictionary.

dictionary : [dict]
    The dictionary to be written to the file.

Output:
None

Logic:
This function opens the specified text file in write mode and writes the string representation of the dictionary to the file.

Example Call:
write_dict('filename', dictionary)
'''

def write_dict(filename, dictionary):
    with open(f'{filename}.txt', 'w', newline='', encoding='utf-8') as f:
        print(dictionary, file=f)

'''
Purpose:
Writes a dictionary to a CSV file.

Input:
filename : [str]
    The name of the CSV file (without extension) to write the dictionary.

dictionary : [dict]
    The dictionary to be written to the CSV file.

Output:
None

Logic:
This function opens the specified CSV file in write mode and writes the dictionary keys and values to the file using the csv.writer.

Example Call:
write_dict_to_csv('filename', dictionary)
'''

def write_dict_to_csv(filename, dictionary):
    with open(f'{filename}.csv', 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f)
        dictionary = {key: values for key, values in dictionary.items() if -1 not in values}
        csvwriter.writerows([key] + values for key, values in dictionary.items())

'''
Purpose:
Writes latitude and longitude values to a CSV file.

Input Arguments:
lat : [float]
    The latitude value to be written to the CSV file.
lon : [float]
    The longitude value to be written to the CSV file.
csv_name : [str]
    The name of the CSV file (including extension) where the data will be written.

Returns:
None

Logic:
This function opens the specified CSV file in write mode and writes the latitude and longitude values to it.
It first writes the header row with field names 'lat' and 'lon', then writes a single row containing the provided latitude and longitude values.

Function Name: write_csv
'''
def write_csv(lat, lon, csv_name):
    with open(csv_name, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['lat', 'lon'])
        writer.writeheader()
        writer.writerow({'lat': lon, 'lon': lat})
