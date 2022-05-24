clc; clear;close all;

%This part is for creating .mat files from .csv files
%dat = readmatrix("dat_raw_0.5m_1m_1.5m.csv");
%tim = readmatrix("tstamp_0.5m_1m_1.5m.csv");
%save("t.mat","tim");
%save("dat.mat","dat");

dfile = importdata("dat.mat");
tfile = importdata("t.mat");
data = dfile(:,2:end-1);
time = tfile(:,2:end-1);
serial_num = dfile(:,1);
label = dfile(:,end);


fplotlim = 100; %xlimit of fft plot
tplotlim = 60; %xlimit of time domain plots
titlefontsize = 14;
linewidth = 1.48; %graph linewidth
axistitlefontsize = 12;
axisfontsize = 10;

fs = 100; %sampling frequency used while data collection
movavgnum = 50;
dtr_order = 5;
eff_ampl_threshold = .2;
data_count = length(label); %number of data
L = length(data(1,:)); %length of each data

noise_halfm = data(find(serial_num == 858),:);
noise_onem = data(find(serial_num == 58),:);
noise_oneandhalfm = data(find(serial_num == 452),:);

n_half = noise_halfm - mean(noise_halfm);
n_one = noise_onem - mean(noise_onem);
n_oneandhalf = noise_oneandhalfm - mean(noise_oneandhalfm);

%startloop = 1;
endloop = data_count;
startloop =1;
%endloop = 2;

for k = startloop:endloop
    raw_data = data(k,:);
    z = raw_data - mean(raw_data); %mean subtracted raw data
    t = time(k,:); %timestamps for plotting later
    l = label(k); %target values
    mvavg = movmean(raw_data,movavgnum); %take moving average of raw data
    mnsb = mvavg - mean(mvavg); %Moving averaged and mean subtracted data
    dtr = detrend(mvavg,dtr_order); %Moving averaged and detrended data
    
    %feature 1 - peak-to-peak amplitude
    amp = max(dtr)-min(dtr); %peak to peak amplitude, found on detrended data

    f1 = fs*(0:(L/2))/L;
    Y = fft(dtr);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    [pfreq, argmax] = max(P1(1:end));
    
    %feature 2 - breathing rate
    rate = f1(argmax)*60; %Breathing frequency, found on detrended data
    
    %FFT is done on moving averaged and mean-subtracted data too, only for
    %plotting
    Y_mnsb = fft(mnsb);
    P2_mnsb = abs(Y_mnsb/L);
    P1_mnsb = P2_mnsb(1:L/2+1);
    P1_mnsb(2:end-1) = 2*P1_mnsb(2:end-1);
    
    fthreshold = pfreq*eff_ampl_threshold;
    count = length(find(P1>=fthreshold));
    
    %feature 3 - effective spectral spread
    eff_amp = count/length(P1)*100;
    
    %feature 4 - signal to noise ratio
    s = serial_num(k);
    if (s>=1&&s<=400)
        y = n_one;
    elseif(s>=401&&s<=800)
        y = n_oneandhalf;
    else
        y = n_half;
    end

    yavg_all = (n_half+n_one+n_oneandhalf)/3;

    yavg_half_one = (n_half+n_one)/2;
    yavg_one_oneandhalf = (n_one+n_oneandhalf)/2;
    yavg_half_oneandhalf = (n_half+n_oneandhalf)/2;

    %snr1 = snr(z);
    %snr2 = mag2db( (rssq(z)/length(z)) / (rssq(y)/length(y))-1);
    temp = rssq(z)/rssq(y);
    if (temp>1)
        snr1 = mag2db(temp-1);
    else
        snr1 = -90;
    end


    tempavg_half_one = rssq(z)/rssq(yavg_half_one);
    if (tempavg_half_one>1)
        snravg_half_one = mag2db(tempavg_half_one-1);
    else
        snravg_half_one = -90;
    end

    tempavg_one_oneandhalf = rssq(z)/rssq(yavg_one_oneandhalf);
    if (tempavg_one_oneandhalf>1)
        snravg_one_oneandhalf = mag2db(tempavg_one_oneandhalf-1);
    else
        snravg_one_oneandhalf = -90;
    end

    tempavg_half_oneandhalf = rssq(z)/rssq(yavg_half_oneandhalf);
    if (tempavg_half_oneandhalf>1)
        snravg_half_oneandhalf = mag2db(tempavg_half_oneandhalf-1);
    else
        snravg_half_oneandhalf = -90;
    end



    tempavg = rssq(z)/rssq(yavg_all);
    if (tempavg>1)
        snravg_all = mag2db(tempavg-1);
    else
        snravg_all = -90;
    end


    %[a,b,yfit] = Fseries(t,mnsb,10);
    %writematrix([s amp rate eff_amp snr1 a' b' l],"handcrafted4.csv",'WriteMode','append');
    writematrix([s amp rate eff_amp snr1 l],"hf_n123.csv",'WriteMode','append'); %appropriate ones are merged
    writematrix([s amp rate eff_amp snravg_all l],"hf_navg.csv",'WriteMode','append'); %all three averaged

    if (s>=1&&s<=400)
         writematrix([s amp rate eff_amp snr1 l],"hf_one.csv",'WriteMode','append'); %individual with appropriate one

         writematrix([s amp rate eff_amp snravg_half_one l],"hf_half_one.csv",'WriteMode','append');
         writematrix([s amp rate eff_amp snravg_one_oneandhalf l],"hf_one_oneandhalf.csv",'WriteMode','append');
    elseif(s>=401&&s<=800)
        writematrix([s amp rate eff_amp snr1 l],"hf_oneandhalf.csv",'WriteMode','append'); %individual with appropriate one

        writematrix([s amp rate eff_amp snravg_half_oneandhalf l],"hf_half_oneandhalf.csv",'WriteMode','append');
        writematrix([s amp rate eff_amp snravg_one_oneandhalf l],"hf_one_oneandhalf.csv",'WriteMode','append');
    else
        writematrix([s amp rate eff_amp snr1 l],"hf_half.csv",'WriteMode','append'); %individual with appropriate one

        writematrix([s amp rate eff_amp snravg_half_one l],"hf_half_one.csv",'WriteMode','append');
        writematrix([s amp rate eff_amp snravg_half_oneandhalf l],"hf_half_oneandhalf.csv",'WriteMode','append');
    end
    
    %plotting code
%     figure(k);
%     subplot(311);
%     plot(t,raw_data,'color',[0 0.5 0],'LineWidth',linewidth-.2); hold on;
%     plot(t,mvavg,'r','LineWidth',linewidth); hold on;
%     grid on;xlim([0 tplotlim]);
%     set(gca,'FontSize',axisfontsize);
%     xl = xlabel("Time (s)","fontsize",axistitlefontsize);
%     %xl.Position(1) = 14;
%     ylabel("Voltage (V)","fontsize",axistitlefontsize);
%     title("Breathing Data, Class Label = "+l,'FontSize',titlefontsize);
%     hh = legend("Raw","Moving averaged","fontsize",axistitlefontsize-.55);
%     %set(hh,'Location','southeast','Orientation','vertical','Box','off');
%     rect = [0.6, 0.61, .25, .12];
%     set(hh, 'Position', rect,'Orientation','horizontal')
% 
% 
%     subplot(312);
%     plot(t,mnsb,'color',[0 0.5 0],'LineWidth',linewidth); hold on;
%     plot(t,dtr,'r','LineWidth',linewidth);
%     %plot(t,mavg_meansub_data-mavg_dtr_data,'k--','LineWidth',0.8);
%     grid on;xlim([0 tplotlim]);
%     set(gca,'FontSize',axisfontsize);
%     xl2 = xlabel("Time (s)","fontsize",axistitlefontsize);
%     %xl2.Position(1) = 14;
%     ylabel("Voltage (V)","fontsize",axistitlefontsize);
%     title("Mean-subtracted and detrended data",'FontSize',titlefontsize);
%     gg = legend("Mean-subtracted","Detrended","fontsize",axistitlefontsize-1);
%     %rect = [0.6, 0.61, .25, .12];
%     %set(hh, 'Position', rect,'Orientation','horizontal')
%     gg.Location = "Northeast";
% 
% 
%     subplot(313);
%     plot(f1*60,P1_mnsb,'color',[0 0.5 0],'LineWidth',linewidth); hold on;
%     plot(f1*60,P1,'r','LineWidth',linewidth);
%     xlim([0 fplotlim]);
%     set(gca,'FontSize',axisfontsize);
%     title('Single-sided Amplitude Spectrum (FFT)','FontSize',titlefontsize);
%     xlabel('Frequency (bpm)',"fontsize",axistitlefontsize);
%     ylabel('Spectral Ampl.(V)',"fontsize",axistitlefontsize-2);grid on;
%     %set(gca,'FontSize',axisfontsize);
%     kk = legend("FFT of mean-subtracted data","FFT of detrended data","fontsize",axistitlefontsize-.7);
%     kk.Location= "Northeast";
end