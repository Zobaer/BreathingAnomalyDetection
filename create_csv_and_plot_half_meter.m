clc; clear; close all;
txtfiledir = "splitted_data_0.5m";
csvfiledir = "csv_0.5m";
var = "dataplots_classwise_0.5m\";
figdir_eupnea = var+"0. Eupnea";
figdir_apnea =  var+"1. Apnea";
figdir_tachypnea =  var+"2. Tachypnea";
figdir_bradypnea =  var+"3. Bradypnea";
figdir_hyperpnea =  var+"4. Hyperpnea";
figdir_hypopnea =  var+"5. Hypopnea";
figdir_kussmaul =  var+"6. Kussmauls";
figdir_faulty =  var+"7. Faulty";

cd(txtfiledir)
files = dir('*.txt');
N = length(files);

fs = 100; %sampling frequency used while data collection
wsize = 60; %windows size (each data will be wsize seconds long)
perfile = fs*wsize; %how many data points per file.

cutdc = 0;
fplotlim = 100; %xlimit of fft plot
tplotlim = 60; %xlimit of time domain plots
movavgnum = 50;
titlefontsize = 10;
linewidth = 1.48; %graph linewidth
axistitlefontsize = 8;
axisfontsize = 7;

for j = 1:N
    thisfile = files(j).name;
    td = load(thisfile);
    s = size(td);
    len = s(1); %number of points in one data
    numfile = floor(len/perfile); %number of data generated from one file
    for i =1:numfile
        onefile = td((i-1)*perfile+1:i*perfile,:);
        onefile = [onefile(:,1)-(i-1)*wsize onefile(:,2)];
        onefile = transpose(onefile);
        %extract data serial number and label
        label = 0;
        %num =  [];
        for k = 1:length(thisfile)
            if(thisfile(k)=='.')
                L = thisfile(k+1);
                label=L-48; %convert string to number
                break;
            else
                num(k) = thisfile(k);
            end
        end
        %num
        serial_num = str2num(num(1:k-1));
        
        raw_dat = onefile(2,:);
        mvavg_dat = movmean(raw_dat,movavgnum);
        mvavg_mnsb_dat = mvavg_dat - mean(mvavg_dat);
        mvavg_mnsb_smth = mvavg_mnsb_dat - movmean(mvavg_mnsb_dat,450);
        mvavg_dtr_dat = detrend(mvavg_dat,5);
        mvavg_std_dat = (mvavg_dat - mean(mvavg_dat))/std(mvavg_dat);

        %This is the line to create data csv
        cd ..
        cd(csvfiledir);
%         writematrix([serial_num raw_dat label],"dat_raw_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num mvavg_dat label],"dat_mvavg_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num mvavg_mnsb_dat label],"dat_mvavg_mnsb_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num mvavg_dtr_dat label],"dat_mvavg_dtr_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num mvavg_std_dat label],"dat_mvavg_std_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num mvavg_mnsb_smth label],"dat_mvavg_smth_0.5m.csv",'WriteMode','append');
%         
%         writematrix([serial_num onefile(1,:) label],"tstamp_0.5m.csv",'WriteMode','append');
        
        %Find_fft
        %fn = fs/2; % Nyquist frequency
        L1 = length(mvavg_std_dat);     % Length of signal
        f1 = fs*(0:(L1/2))/L1;

        %moving averaged + standardized data
        Y1_mvavg_std = fft(mvavg_std_dat);
        P2_mvavg_std = abs(Y1_mvavg_std/L1);
        P1_mvavg_std = P2_mvavg_std(1:L1/2+1);
        P1_mvavg_std(2:end-1) = 2*P1_mvavg_std(2:end-1);


        %raw data
        Y1_raw = fft(raw_dat);
        P2_raw = abs(Y1_raw/L1);
        P1_raw = P2_raw(1:L1/2+1);
        P1_raw(2:end-1) = 2*P1_raw(2:end-1);

        %moving averaged + meansub data
        Y1_mnsb = fft(mvavg_mnsb_dat);
        P2_mnsb = abs(Y1_mnsb/L1);
        P1_mnsb = P2_mnsb(1:L1/2+1);
        P1_mnsb(2:end-1) = 2*P1_mnsb(2:end-1);

        %moving averaged +detrend
        Y1_dtr = fft(mvavg_dtr_dat);
        P2_dtr = abs(Y1_dtr/L1);
        P1_dtr = P2_dtr(1:L1/2+1);
        P1_dtr(2:end-1) = 2*P1_dtr(2:end-1);

        %meansub + smoothed
        Y1_smoothed = fft(mvavg_mnsb_smth);
        P2_smoothed = abs(Y1_smoothed/L1);
        P1_smoothed = P2_smoothed(1:L1/2+1);
        P1_smoothed(2:end-1) = 2*P1_smoothed(2:end-1);

%         writematrix([serial_num f1*60 label],"fstamp_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num P1_mvavg_std label],"fft_mvavg_std_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num P1_raw label],"fft_raw_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num P1_mnsb label],"fft_mvavg_mnsb_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num P1_dtr label],"fft_mvavg_dtr_0.5m.csv",'WriteMode','append');
%         writematrix([serial_num P1_smoothed label],"fft_mvavg_smth_0.5m.csv",'WriteMode','append');
        
        %Handcrafted_feature_extraction
        amp = max(mvavg_std_dat)-min(mvavg_std_dat);
        amp_raw = max(raw_dat)-min(raw_dat);
        amp_mnsb = max(mvavg_mnsb_dat)-min(mvavg_mnsb_dat);
        amp_dtr = max(mvavg_dtr_dat)-min(mvavg_dtr_dat);
        amp_smoothed = max(mvavg_mnsb_smth)-min(mvavg_mnsb_smth);
        
        %moving averaged + standardized data
        [pfreq, argmax] = max(P1_mvavg_std(cutdc+1:end));
        fthreshold = pfreq*.2;
        count = length(find(P1_mvavg_std>=fthreshold));
        bw = count; %/100*100;
        %pow = P1_1.^2;

        %raw data
        [pfreq_raw, argmax_raw] = max(P1_raw(cutdc+1:end));
        fthreshold_raw = pfreq_raw*.2;
        count_raw = length(find(P1_raw>=fthreshold_raw));
        bw_raw = count_raw; %/100*100;

        %moving averaged + meansub data
        [pfreq_mnsb, argmax_mnsb] = max(P1_mnsb(cutdc+1:end));
        fthreshold_mnsb = pfreq_mnsb*.2;
        count_mnsb = length(find(P1_mnsb>=fthreshold_mnsb));
        bw_mnsb = count_mnsb; %/100*100;

        %moving averaged + detrended data
        [pfreq_dtr, argmax_dtr] = max(P1_dtr(cutdc+1:end));
        fthreshold_dtr = pfreq_dtr*.2;
        count_dtr = length(find(P1_dtr>=fthreshold_dtr));
        bw_dtr = count_dtr; %/100*100;

        %_smoothed
        [pfreq_smoothed, argmax_smoothed] = max(P1_smoothed(cutdc+1:end));
        fthreshold_smoothed = pfreq_smoothed*.2;
        count_smoothed = length(find(P1_smoothed>=fthreshold_smoothed));
        bw_smoothed = count_smoothed; %/100*100;

        snr_val = snr(raw_dat);

        writematrix([serial_num amp f1(argmax+cutdc)*60 snr_val bw label],"handcrafted_mvavg_stdzed_0.5m_snr.csv",'WriteMode','append');
        writematrix([serial_num amp_raw f1(argmax_raw+cutdc)*60 snr_val bw_raw label],"handcrafted_raw_0.5m_snr.csv",'WriteMode','append');
        writematrix([serial_num amp_mnsb f1(argmax_mnsb+cutdc)*60 snr_val bw_mnsb label],"handcrafted_mvavg_mnsb_0.5m_snr.csv",'WriteMode','append');
        writematrix([serial_num amp_dtr f1(argmax_dtr+cutdc)*60 snr_val bw_dtr label],"handcrafted_mvavg_dtr_0.5m_snr.csv",'WriteMode','append');
        writematrix([serial_num amp_smoothed f1(argmax_smoothed+cutdc)*60 snr_val bw_smoothed label],"handcrafted_mvavg_smth_0.5m_snr.csv",'WriteMode','append');
%         
        cd ..
        %%%%%%%%%%%%%%%%%%%%%%%uncomment from next line for plot
        %plotting and saving the plot
        %cd(figdir); %go to the folder where the plots will be saved

        switch label
            case 0
                cd(figdir_eupnea);
                txt = "(Eupnea)";
            case 1
                cd(figdir_apnea);
                txt = "(Apnea)";
            case 2
                cd(figdir_tachypnea);
                txt = "(Tachypnea)";
            case 3
                cd(figdir_bradypnea);
                txt = "(Bradypnea)";
            case 4
                cd(figdir_hyperpnea);
                txt = "(Hyperpnea)";
            case 5
                cd(figdir_hypopnea);
                txt = "(Hypopnea)";
            case 6
                cd(figdir_kussmaul);
                txt = "(Kussmaul's breathing)";
            otherwise
                cd(figdir_faulty);
                txt = "(Faulty Data)";
        end

        f = figure('visible','off'); %don't display the plots in MATLAB figure window
        %figure(j);
        subplot(311);
        plot(onefile(1,:),raw_dat,'color',[0 0.5 0],'LineWidth',linewidth-.2); hold on;
        plot(onefile(1,:),mvavg_dat,'r','LineWidth',linewidth); hold on;
        grid on;xlim([0 tplotlim]);
        set(gca,'FontSize',axisfontsize);
        xl = xlabel("Time (s)","fontsize",axistitlefontsize);
        xl.Position(1) = 14;
        ylabel("Voltage (V)","fontsize",axistitlefontsize);
        title("Breathing Data, Class Label = "+label+" "+txt,'FontSize',titlefontsize);
        hh = legend("Raw","Moving averaged","fontsize",axistitlefontsize-.55);
        %set(hh,'Location','southeast','Orientation','vertical','Box','off');
        rect = [0.6, 0.61, .25, .12];
        set(hh, 'Position', rect,'Orientation','horizontal')
        

        subplot(312);
        plot(onefile(1,:),mvavg_mnsb_dat,'color',[0 0.5 0],'LineWidth',linewidth); hold on;
        plot(onefile(1,:),mvavg_dtr_dat,'r','LineWidth',linewidth);
        %plot(onefile(1,:),mavg_meansub_data-mavg_dtr_data,'k--','LineWidth',0.8);
        grid on;xlim([0 tplotlim]);
        set(gca,'FontSize',axisfontsize);
        xl2 = xlabel("Time (s)","fontsize",axistitlefontsize);
        %xl2.Position(1) = 14;
        ylabel("Voltage (V)","fontsize",axistitlefontsize);
        title("Mean-subtracted and detrended data",'FontSize',titlefontsize);
        gg = legend("Mean-subtracted","Detrended","fontsize",axistitlefontsize-1);
        %rect = [0.6, 0.61, .25, .12];
        %set(hh, 'Position', rect,'Orientation','horizontal')
        gg.Location = "Northeast";
        
        
        subplot(313);
        plot(f1*60,P1_mnsb,'color',[0 0.5 0],'LineWidth',linewidth); hold on;
        plot(f1*60,P1_dtr,'r','LineWidth',linewidth);
        xlim([0 fplotlim]);
        set(gca,'FontSize',axisfontsize);
        title('Single-sided Amplitude Spectrum (FFT)','FontSize',titlefontsize);
        xlabel('Frequency (bpm)',"fontsize",axistitlefontsize);
        ylabel('Spectral Ampl.(V)',"fontsize",axistitlefontsize-2);grid on;
        %set(gca,'FontSize',axisfontsize);
        kk = legend("FFT of mean-subtracted data","FFT of detrended data","fontsize",axistitlefontsize-.7);
        kk.Location= "Northeast";


        saveas(f,serial_num+"."+L+".png");
        cd ..\..
        cd(txtfiledir);
    end
end