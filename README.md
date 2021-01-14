# Mini-project 3: 
### Paper https://afribary.com/works/human-activity-recognition-by-wearable-sensors/read

### Project Explanation
In the 3rd mini-project, we will look at the problem of human action recognition from wearable sensor data. We will analyse the labelled data se from a mobile phone containing an accelerometer and gyroscope.
In the mini-project, you will implement a method that is able to classify a person's action into six pre-defined classes and evaluate the performance of the method.

## Goal of the project
+ To familiarize with Human Activity Recognition  
+ Segment human activity from mobile phone sensor data
+ Learn machine learning tools and techniques used in activity recognition

## Data
+ 30 subjects within the age interval 19 - 48 y/o
+ Daily living activities: Walking, Walking-Stairs-Up, Walking-Stairs-Down, Sitting, Standing, Laying
+ Waist-mounted smartphones (Samsung Galaxy SII) with embedded inertial sensors.

### Data preprocessing
+ Noise filtering
+ Windowing with fixed size 2.56 s window each containing 128 samples with 50% overlap
+ The sensor acceleration is divided to two components: the gravity and body acceleration (Butterworth low-pass filter with cutoff at 0.3 Hz)
+ For each window, a vector of features was computed in both time and frequency
+ You have acccess to both the extracted features and the preprocessed, windowed acceleration and angular velocity signals.
+ time domain: X time, Y magnitude. Fourier domain: X frequency (Hz), Y magnitude. 
