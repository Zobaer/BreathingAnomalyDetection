clc; clear; close all;
class = "6"; %6_0.5m done
filename = "class_"+class+"_collected_new_1.5m";
filename2 = strcat('./collected_data/',filename,'.txt');

buffer = 0; %10s buffer overall, 5s at the beginning and 5 sec at the end
num_frames = 51;
frame_duration = 60; %60s
f_s = 100;
sigtime = num_frames*frame_duration;
num_points_sig = sigtime*f_s;
tottime = sigtime + buffer;
num_points_tot = tottime*f_s;
data = readmatrix(filename2);
time_list_sig = linspace(0,sigtime,num_points_sig);
time_list_tot = linspace(0,tottime,num_points_tot);

collected_time_tot = data(:,1);
collected_sig_tot = data(:,2);
subplot(711);
plot(collected_time_tot,collected_sig_tot); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");

split_start_point = 55;
[minval, startindex] = min(abs(collected_time_tot(:,1) - split_start_point));


perfile = f_s*frame_duration; %number of data per file
data_new = data(startindex:end,:);
s = size(data_new);
len = s(1);
numfile = floor(len/perfile); %number of file
cd splitted_data\;
for i =1:numfile
    onefile = data_new((i-1)*perfile+1:i*perfile,:);
    onefile = [onefile(:,1)-onefile(1) onefile(:,2)];
    writematrix(onefile,num2str(i+(8+str2double(class))*50)+"."+class+".txt"); %for 0.5m -> 16, 1.5m-> 8, 1m->0
end
%cd ..\

subplot(712);
d1 = readmatrix("701."+class+".txt");
plot(d1(:,1),d1(:,2)); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");
subplot(713);
d2 = readmatrix("702."+class+".txt");
plot(d2(:,1),d2(:,2)); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");
subplot(714);
d3 = readmatrix("703."+class+".txt");
plot(d3(:,1),d3(:,2)); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");
subplot(715);
d4 = readmatrix("704."+class+".txt");
plot(d4(:,1),d4(:,2)); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");
subplot(716);
d5 = readmatrix("749."+class+".txt");
plot(d5(:,1),d5(:,2)); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");
subplot(717);
d6 = readmatrix("750."+class+".txt");
plot(d6(:,1),d6(:,2)); grid on; xlabel("Time(s)"); ylabel("Voltage(V)");
