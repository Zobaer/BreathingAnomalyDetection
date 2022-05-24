clc; clear; close all;
fplotlim = 100; %xlimit of fft plot
tplotlim = 60; %xlimit of time domain plots
movavgnum = 50;
titlefontsize = 13;
linewidth = 1.64; %graph linewidth
axistitlefontsize = 11;
axisfontsize = 7;
fs = 100;

file = "1061.5.txt";
label = "0";
txt = "(Eupnea)";
m = readmatrix(file);
t = m(:,1);
d = m(:,2);
subplot(311);
plot(t,d,'color',[0 0.5 0]); grid on; hold on;
xlabel("Time"); ylabel("Voltage (V)");
title("Datafile "+file+", Class Label = "+label+" "+txt,'FontSize',titlefontsize);

d_mm = movmean(d,50);
plot(t,d_mm,'r',"linewidth",1.7); grid on;

hh = legend("Raw", "Moving averaged","fontsize",axistitlefontsize);
hh.Location = "southeast";

d_mm_mnsb = d_mm - mean(d_mm);
subplot(312);
plot(t,d_mm_mnsb,'b',"linewidth",1.7); grid on; hold on;
xlabel("Time"); ylabel("Voltage (V)");
%title("Mean subtracted data",'FontSize',titlefontsize);




d_mm_dtr = detrend(d_mm,5);
%d_mm_mnsb_dtr = bandpass(d_mm_mnsb,[1/60 50/60],fs);
%d_mm_mnsb_dtr2 = detrend(d_mm_mnsb_dtr);
%subplot(413);
%plot(t,d_mm_dtr,'g',"linewidth",1.3); %grid on; hold on;
%xlabel("Time"); ylabel("Voltage (V)");
%title("Data after detrending",'FontSize',titlefontsize);

% for i = 1:length(d_mm_dtr)
%     d_mm = 
% end
% d_mm_smoothed = smoothdata(d_mm_mnsb,"movmedian",10);
d_mm_smoothed = d_mm_mnsb - movmean(d_mm_mnsb,450);
plot(t,d_mm_dtr,'r',"linewidth",1.5);hold on;
gg = legend("Mean subtracted data","Detrended data");
gg.Location = "Southeast";



L1 = length(d_mm);     % Length of signal
f1 = fs*(0:(L1/2))/L1;

Y1 = fft(d_mm_mnsb);
P2_1 = abs(Y1/L1);
P1_1 = P2_1(1:L1/2+1);
P1_1(2:end-1) = 2*P1_1(2:end-1);
fstamp = f1(2:101)*60;
Sp_amp = P1_1(2:101);
subplot(313);
plot(fstamp,Sp_amp,'color',[0 0.45 0],"linewidth",2.1); grid on; hold on;
xlabel("Frequency (bpm)"); ylabel("Spectral Ampl.(V)");
title("Single-sided amplitude spectrum (FFT)",'FontSize',titlefontsize);

Y1 = fft(d_mm_smoothed);
P2_1 = abs(Y1/L1);
P1_1 = P2_1(1:L1/2+1);
P1_1(2:end-1) = 2*P1_1(2:end-1);
fstamp = f1(2:101)*60;
Sp_amp = P1_1(2:101);
%subplot(414);
plot(fstamp,Sp_amp,'m',"linewidth",1.7);
hh = legend("FFT of regular data", "FFT of smoothed data","fontsize",axistitlefontsize);
hh.Location = "northeast";

[snr n] = snr(d)