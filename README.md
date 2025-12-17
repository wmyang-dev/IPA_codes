# Introduction

This project is an implementation of various image processing techniques in Python. 
It focuses on tasks such as filtering, template matching, and analyzing region properties. 

The main goal is to provide a set of scripts that can process images, detect features (like raisins in example datasets), and perform basic analysis on detected objects. 
It is useful for learning fundamental image processing workflows, testing different filtering methods, and experimenting with noise addition and object detection.

The project includes several scripts organised by functionality, and example outputs are provided in below for reference.

## Part 1: Filters

This section demonstrates fundamental image filtering techniques. 
Starting from a 512×512 grayscale image, exploring noise addition and edge detection using Sobel and Canny operators. 
The outputs illustrate how these filters respond to Gaussian noise and highlight key image features, providing a visual proof of the processing results.

The source code is [`part1.py`](part1.py).

### Output:
<img width="660" height="391" alt="Screenshot 2025-12-17 at 11 53 47" src="https://github.com/user-attachments/assets/d36cd2bf-99b4-4528-b37c-5639c086950f" />

## Part 2: Template matching

This section focuses on detecting patterns in images using template matching. 
Starting from a source image and a template ![template_tie](https://github.com/user-attachments/assets/98277481-2bdf-4db5-ba0a-3d3b20db75e3), exploring detecting and highlighting instances of the template in the source image. 
The outputs illustrate the accuracy of single-instance detection as well as full detection of all template occurrences.

The source code is [`part2a`](part2a_rotateAndNoise.py) and [`part2b`](part2b).




<img width="1107" height="665" alt="part2a" src="https://github.com/user-attachments/assets/1e1afba7-b0fb-4882-b953-69e1aeaaf02f" />
<img width="987" height="59" alt="0 92 and 0 91" src="https://github.com/user-attachments/assets/f2fc32bb-868c-4575-ab19-5a7f56a282da" />



## Part 3: Region Properties

This section focuses on analysing connected regions in an image to extract meaningful object-level properties. 
Using a colour image of raisins, exploring automated object segmentation while excluding boundary-touching properties. 
The analysis includes counting valid objects, identifying the smallest object, and extracting its its area and centroid.


<img width="900" height="593" alt="found object display" src="https://github.com/user-attachments/assets/0432f579-4848-42eb-b4a2-fbd3ff5066ce" />
<img width="976" height="296" alt="found object details" src="https://github.com/user-attachments/assets/c12d579e-b8ce-4e17-8441-83ddf7e54578" />

An additional experiment demonstrates the robustness of the approach under image rotation.
<img width="1795" height="862" alt="Screenshot 2024-10-27 at 22 41 01" src="https://github.com/user-attachments/assets/1fbb6d11-57b1-4413-baeb-fdc60b0418f8" />
