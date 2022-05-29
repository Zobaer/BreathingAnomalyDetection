# ML based breathing anomaly detection
This is part of an NSF funded project on human respiration monitoring using infrared sensing. Here, anomaly detection was performed on the breathing data through machine learning based data classification. Breathing classes were:
1) Eupnea or normal breathing
2) Apnea or cessation of breathing
3) Tachypnea
4) Bradypnea
5) Hyperpnea
6) Hypopnea
7) Kussmaul's breathing
8) Faulty data



# Description of the files:

1) **Classification_using_DT_RF.py**: This is the main python file that did classification of breathing data into 8 different classes using decision tree and random forest algorithms, does hyperparameter tuning, 10-fold cross validation, calculate accuracies and cross-validation create confusion matrices.
2) **automated_data_gen.m**: This is MATLAB file that reads a list of 50 pairs (breathing depth, breathing rate) from a file and generates continuous breathing data (amplitude), each 60 seconds long, automatically based on given rate and depth values. It adds 5 seconds of buffer at the beginning and at the end of the whole data. This generated data can be loaded to the breathing robot and the robot will breathe following that pattern.
3) **split_file.m**: After collecting data for longer duration (50 mins), this MATLAB file was used to split it into individual data files (60 seconds segments). It does appropriate naming too to recognize them.
4) **data.mat**: This file has collected 1200 breathing data with class labels, in MATLAB compatible format.
5) **plot_one_data.m**: MATLAB file to plot one data, along with its amplitude spectrum for visualization.

![alt text](https://github.com/Zobaer/BreathingAnomalyDetection/blob/main/figs/Data_visualization.png)

7) **ML data collection plan.xlsx**: This is the plan for data collection at 0.5m, 1m and 1.5m distances, belonging to 8 different classes.
8) **create_csv_and_plot_half_meter.m**: MATLAB code to create csv files fromt he text files, calculate some handcrafted features and plot all data in separate folders, in different image files, for visualization and data cleaning
9) **Handcrafted_feature_extraction.m**: Data were read from csv or  .mat files and below four handcrafted features were extracted for classification using this MATLAB code:
  - Peak-to-peak amplitude (in V)
  - Breathing rate (in bpm)
  - Effective spectral spread (a custom feature to get an estimate on how the frequency components are spread in the signal, it differentiates between one clean peak vs a lot of peaks at different frequencies in the amplitude spectrum.)
  - Signal to noise ratio
