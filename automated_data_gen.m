%This code reads breathing depth and rate lists from a txt file and creates
%a single waveform with those data (with some padding only at the beginning
%of the file

clc; clear; close all;
filename = 'class_6_depth_rate_new';
filename2 = strcat('./Respiration_Waveforms/',filename,'.txt');
frame_duration = 60; %in seconds
f_s = 100;

zero_pad_len = 5; %in seconds
%end_zero_pad_len = 5; %in seconds
%First column contains depth information, while second column contains rate
%information
depth_rate_data = readmatrix(strcat('./depth_rate_lists/',filename,'.txt'));

num_of_frames = length(depth_rate_data(:,1));
total_duration = frame_duration*num_of_frames;
numPoints_frame = frame_duration*f_s;
numPoints_total = total_duration*f_s;
t_frame = linspace(0,frame_duration,numPoints_frame);


freq_bpm = depth_rate_data(:,2); %in bpm
freq_hz = freq_bpm/60;
omega = 2*pi*freq_hz;
phase = omega.*t_frame;

%each row contain waveforms with each row of depth-rate information
%columns are timestamps within a frame
waveform = (sin(phase/2)).^6;

depths = depth_rate_data(:,1);
waveform_modulated = (depths/100).*waveform;
%waveform_modulated = waveform;

%merge the rows together for plotting
flatten = [1,num_of_frames*numPoints_frame];
flattened_waveform = reshape(waveform_modulated.',flatten);

zero_padding = zeros([1,zero_pad_len*f_s]);
flat_wave_with_zero_pad = [zero_padding flattened_waveform zero_padding];

t_total = linspace(0,total_duration+2*zero_pad_len,numPoints_total+ ...
    2*zero_pad_len*f_s);

plot(t_total,flat_wave_with_zero_pad); grid on;
xlim([0 total_duration+2*zero_pad_len]);
%ylim([0 .35]);

%% Write File
fileID = fopen(filename2, 'wt');
fprintf(fileID,'%f\n',flattened_waveform);