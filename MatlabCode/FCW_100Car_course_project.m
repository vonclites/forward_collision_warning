% Copyright 2015 ; Yaser P. Fallah, West Virginia University, Dept. of
% Computer Science and Electrical Engineering

function [] = FCW_100Car_course_project_v7()
clear all; close all; clc;
timestep = 0.1; %sec
G = 9.8;
DEBUG = 0;
DO_EDCFS = 0;
filenumbers = 8296:1:9123;
global CONST_SPEED_MODEL;  % 1 means constat speed model, 0 means const. acceleration model.
CONST_SPEED_MODEL = 1;

%--------read from files, and process for alert
%PAPERS:   look at "Vehicle lateral and longitudinal velocity estimation based on Unscented Kalman Filter"
fnameprefix = '..\\100CarData\\100CarAllData_v1_3\\HundredCar_Public_%d.txt';
scenario = 0;
AnalysisResult_PB = zeros(1,7);
for filenum = filenumbers %8300:1:8400;
    FileNumber = filenum
    scenario = scenario + 1;
    fname = sprintf(fnameprefix,filenum);
    if ~exist(fname, 'file')
        continue
    end
    fid = fopen(fname);
    C = textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %*[^\n]', 'delimiter', ',');
    fclose(fid); tripid = C{1}; sync = C{2}; triptime = C{3}; gaspedpos=C{4}; speedvc=C{5};
    speedgpshrz=C{6}; yaw=C{7} ; gpsheading = C{8}; latAcc = C{9}; longAcc = C{10};
    if(DEBUG) figure();plot(speedvc);end;
    
    %initialize LV and FV
    N=size(triptime, 1);
    x_lv=zeros(1,N);v_lv=zeros(1,N);accel= zeros(1,N);a_lv_calc = zeros(1,N);
    x_fv=zeros(1,N);v_fv=zeros(1,N);a_fv = zeros(1,N);
    Rw=zeros(1,N); %warning range
    GrndTrth_Separation_Dist = zeros(1,N);
    RwCamp = zeros(1,N);
    RwCamph = zeros(1,N);
    RwKnipling = zeros(1,N);
    
    v_lv = 0.44704*(speedvc'); %change to m/s
    if (v_lv(1) < 0) v_lv(1) = 0; end;
    
    % Cleaning acceleration data, removing bias and smoothing acceleration
    acclong = G*(longAcc);% - mean(longAcc(1:69)) );
    acclat = G*(latAcc);%  - mean(latAcc(1:25)) );
    heading = 0.1*(yaw - mean(yaw(1:25)));
    accel = acclong.*cos(heading)-acclat.*sin(heading);
    nn = min (N,150);
    acc_bias = (   sum(accel(1:nn)) *timestep - (v_lv(nn)-v_lv(1)) )/(nn*timestep);
    a_lv_unbiased = accel - acc_bias;
    a_lv = filterAcc(a_lv_unbiased, 5); % odd number averaging filter
    %      figure(45); hold on; plot(a_lv);plot(a_lv_unbiased,'r');
    
    v_lv_calc(1) = v_lv(1);
    tripsamples = size (triptime,1);
    v_fv(1) = v_lv(1) ;%m/s
    x_fv(1) = 0; x_lv(1)= x_fv(1) + 40;
    cfmode = zeros(1,N);
    
    % Car following model from MITSIM is implemented here. It calculates
    % following vehicle (host vehicle) position and speed, using
    % actual information from lead vehicle (remote vehicle)
    for t=2:tripsamples
        if v_lv(t) < 0   % for the values of V that are missing, -1 is entered in VTTI data, use the previous speed instead
            v_lv(t) = v_lv(t-1);
        end
        % should we use constant acceleraton or constant speed model for
        a_lv_calc(t) = (v_lv(t) - v_lv(t-1))/timestep ;
        v_lv_calc(t) = v_lv_calc(t-1)+ a_lv(t-1)*timestep + 0.02*(0.5-rand(1));
        v_lv(t) = v_lv_calc(t) ;
        x_lv(t) = x_lv(t-1) + v_lv(t-1)*timestep + 0.5*a_lv(t-1)*timestep^2 + 0.02*(0.5-rand(1)); %assume constant acceleration from previous sample + 0.5*a_lv(t-1)*timestep^2; %assume constant acceleration from previous sample
        %v_lv(t) = v_lv(t-1) + a_lv(t-1)*timestep;
        
        x_fv(t) = x_fv(t-1) + v_fv(t-1)*timestep + 0.5*a_fv(t-1)*timestep^2; %assume constant acceleration from previous sample
        v_fv(t) = v_fv(t-1) + a_fv(t-1)*timestep;
        
        if ( v_fv(t) <= 0.44704*0)
            v_fv(t) = 0.44704*0.0;
            %         'FV stopped at '
            %         t
        end
        
        %car-following model
        VehLen = 4.5; %meters
        HlowerT = 0.5; HupperT = 1.36; Vdesired = 0.44704*65;
        Hupper = HupperT * ( v_fv(t) ) + 20 ;
        Hlower = HlowerT * ( v_fv(t) ) + 5;
        if( x_lv(t) - x_fv(t) -VehLen > Hupper )  %free driving
            cfmode(t) = 1;
            if (v_fv(t) < Vdesired)
                a_fv(t) = aplusf(v_fv(t));
            elseif v_fv(t) == Vdesired
                a_fv(t) = 0;
            else
                a_fv(t) = aminusf(v_fv(t));
            end
            
        elseif( x_lv(t) - x_fv(t) -VehLen < Hlower )  %emergency
            decel = 1*G; %.5G
            cfmode(t) = -1;
            if( x_lv(t) - x_fv(t) -VehLen <= 0 )
                if (DEBUG)Hlower,Hupper, end;
                separation_distance = x_lv(t) - x_fv(t)  -VehLen;
                x_fv(t) = x_lv(t) - VehLen -1;
            else
                if( v_fv(t) > v_lv(t) )
                    decel = min(aminusf(v_fv(t)), (a_lv(t)- 0.5*((v_fv(t) - v_lv(t))^2) / (x_lv(t) - VehLen - x_fv(t)) ) );
                else
                    decel = min(aminusf(v_fv(t)), a_lv(t) + 0.25*aminusf(v_fv(t)));
                end
            end
            a_fv(t) = decel;
            
        else %car following if ( x_lv(t) - x_fv(t) > 100 )
            cfmode(t) = 0;
            if ( v_fv(t) > v_lv(t) ) %deceleration
                alpha = 1.55;beta = 1.08; gamma = 1.65;
            else %acceleration
                alpha = 2.15;beta = -1.67; gamma = -0.89;
            end
            if( v_fv(t) > 0.44*2 )
                %Yang's formula
                a_fv(t) = alpha*(v_fv(t)^beta)*( (v_lv(t) - v_fv(t)) / ((x_lv(t) - VehLen- x_fv(t))^gamma) );
                if(a_fv(t)>0.5*G )
                    a_fv(t) = 0.5*G;
                elseif (a_fv(t) < -0.8*G)
                    a_fv(t) = -0.8*G;
                end
            else
                a_fv(t) = a_lv(t);
            end
            
        end
        GrndTrth_Separation_Dist(t) = x_lv(t) - x_fv(t) - VehLen;
        
        
        
    end%end of trip,
    
    %     figure(); hold on; plot(a_lv);plot(a_lv_calc,'r');plot(acclong,'k'); plot(acclat,'g'); %plot(a_lv,'b');
    %     figure();plot(v_lv); hold on; plot(v_lv_calc,'r');

    triptime = 0:timestep:(tripsamples*timestep)-timestep;
    
    %Warning Algorithm, establish ground truth for ideal communication case
    RwCamp = CAMPLinearWarningAlg(v_lv,v_fv,a_lv,a_fv) ; %returns warning range
    GroundTruthAlertRange = RwCamp;
    %         plot(triptime,Range,'b');
    %         plot(triptime,RwCamp,'r');

    
    % sample test with a particular Rate and PER value. This function
    % returns the received information of LV (remote vehicle) over a
    % network (using periodic sampling pf LV position). PBRate will be
    % almost Rate*(1-PER), 
    Rate = 10; PER = 0.90;
    [v_lv_rcvd, a_lv_rcvd, x_lv_rcvd, PBRate] = findReceivedLVInfoOverPBNetwork(v_lv,a_lv,x_lv,Rate,PER);
    
    %position tracking error - error in position of LV as calculated in the
    %FV (host vehicle)
    x_lv_est =  x_lv_rcvd;
    PTE(scenario)=prctile (abs (x_lv - x_lv_est),95);
    est_Separation_Dist = x_lv_est - x_fv - VehLen;
            
    %      figure();hold on; plot(v_lv,'b');plot(v_lv_rcvd,'r');
    %                 figure(21);hold on; plot(a_lv,'b');plot(a_lv_rcvd,'k');
    %                 figure(22);hold on; plot(v_lv,'b');plot(v_lv_rcvd,'k');plot(v_fv,'r')
    %                 figure(23);hold on; plot(x_lv,'b');plot(x_lv_est,'k');plot(x_fv,'r')
    
    WarningRangeCalculatedUsingReceivedInfo = CAMPLinearWarningAlg(v_lv_rcvd,v_fv,a_lv_rcvd, a_fv);
    [a, b, c, d, e, f, g] = analyzeaccuracy(WarningRangeCalculatedUsingReceivedInfo, GroundTruthAlertRange, est_Separation_Dist,GrndTrth_Separation_Dist);
    AnalysisResult_PB(scenario,1:7) = [a b c d e f g];
    
    %Save Ground Truth and Vehicle Data to file
    %(needed for machine learning)
    %saveVData('v_data_per0', filenum, v_lv, v_fv, a_lv, a_fv, GrndTrth_Separation_Dist, GroundTruthAlertRange)
    saveVData('v_data_per90', filenum, v_lv_rcvd,v_fv,a_lv_rcvd, a_fv, est_Separation_Dist, WarningRangeCalculatedUsingReceivedInfo)
end%filenum, scenario

    meanAcc = mean(AnalysisResult_PB(:,2))
    %figure();hold on;xlabel('rate of transmitted information (Hz)');ylabel('Accuracy');
    %     meanGmeanForEachRateBL(1:size(Raterange'.*(1-PERrange)',1))=mean(AnalysisResult_BL(:,1:size(PERrange,2),1:size(Raterange,2),7));
    %     semilogx( Raterange'.*(1-PERrange)' , meanGmeanForEachRateBL );
    %xlim([0.2 10]);
    save FCW_Model_Results1.mat
end

%Save Vehicle Data to file
%(needed for machine learning)
function [] = saveVData(directory, filenum, v_lv,v_fv,a_lv, a_fv, separation_dist, warning_range)
    if ~exist(strcat('..\\ml_data\\',directory), 'dir')
      mkdir(strcat('..\\ml_data\\',directory));
    end
    f = strcat('..\\ml_data\\',directory,'\\',num2str(filenum),'.txt');
    v_data = horzcat(v_lv', v_fv', a_lv, a_fv', separation_dist', warning_range');
    csvwrite(f,v_data)
end

%%for Data Communicated over a network, using periodic beaconing
function [v_lv_rcvd,a_lv_rcvd,x_lv_rcvd,PBRATE] = findReceivedLVInfoOverPBNetwork(v_lv,a_lv,x_lv, rate,PER)
%base rate is 10Hz
global CONST_SPEED_MODEL;
numsent = 1;
v_lv_rcvd = v_lv;a_lv_rcvd = a_lv;x_lv_rcvd=x_lv;
if (rate > 10)
    disp('Error, RATE should be less than 10');
    return;
end
interval = 10/rate;lastrxindx=1;datatimestep = (1/10);
for tt = 2:size(v_lv,2)
    rtt = ceil(interval*floor(tt/interval)); % time spaced to achieve the rate
    if(rtt == 0), rtt = tt;end;
    v_lv_rcvd(tt) = v_lv( rtt ); %use sample and hold -> leads to constant speed coasting
    a_lv_rcvd(tt) = a_lv( rtt );
    x_lv_rcvd(tt) = x_lv( rtt );
    lost = (rand(1)<PER) ;%getPERfromNetworkModel();
    if(lost == 0 && tt == rtt) % a new packet received
        lastrxindx = tt;
        numsent = numsent+1;
    end
    if CONST_SPEED_MODEL
        x_lv_rcvd(tt) =  x_lv(lastrxindx) + datatimestep*(tt - lastrxindx)*v_lv(lastrxindx);
        v_lv_rcvd(tt) =  v_lv(lastrxindx) ;  %use sample and hold -> leads to constant speed coasting
        if(tt == lastrxindx)
            a_lv_rcvd(tt) =  a_lv(tt);
        else
            a_lv_rcvd(tt) =  0;
        end
    else
        x_lv_rcvd(tt) =  x_lv(lastrxindx) + datatimestep*(tt - lastrxindx)*v_lv(lastrxindx) + a_lv(lastrxindx)*(datatimestep*(tt - lastrxindx))^2;
        v_lv_rcvd(tt) =  v_lv(lastrxindx) + (tt - lastrxindx)*a_lv(lastrxindx)*datatimestep;  %use sample and hold -> leads to constant speed coasting
        a_lv_rcvd(tt) =  a_lv(lastrxindx) ;
    end
    
end
PBRATE = (datatimestep/0.1)*numsent/(tt*datatimestep);
end

function [r_w] = CAMPLinearWarningAlg(Vlv,Vfv,Alv,Afv) % r is range,
%td is the total delay time including driverresponse time and brake system delay
td=2.05;
G = 9.8;

for tt = 1:size(Vlv,2)
    %predicted SV speed after the delay
    Vfvp(tt) = Vfv(tt) + Afv(tt)*(td); %Asv should be negative
    Vlvp(tt) = Vlv(tt) + Alv(tt)*(td); %Asv should be negative
    if Vfvp(tt) < 0
        Vfvp(tt) = 0;
    end
    if Vlvp(tt) < 0
        Vlvp(tt) = 0;
    end
    
    %required deceleration: ( in ft/s and ft/s^2 )
    %convert m/s to ft/s
    cnv = 3.28084; % m/s to MPH : 2.23694
    dec_req(tt)= -5.308 + 0.685*Alv(tt)*cnv + 2.57*(Vlv(tt)*cnv>0) - 0.086*(Vfvp(tt) - Vlvp(tt))*cnv;
    dec_req(tt) = dec_req(tt) / cnv;    %convert back to m
    
    %the following is in m/s and is correct and similar to the above required decel. (g) = -0.164 + 0.668(lead decel. in g’s) -0.00368(closing speed in MPH) + 0.078(if lead moving)
    %dec_req(tt)= G*( -0.165 + 0.685*(Alv(tt)/G) + 0.080*(Vlv(tt)>0) - 0.00877*(Vfvp(tt) - Vlvp(tt)) );
    %range_lost during td
    r_d(tt) = (Vfv(tt) - Vlv(tt))*td + 0.5*(Afv(tt) - Alv(tt))*(td^2);
    if( Vlv(tt) == 0) % case 1, LV stationary all the way
        BOR = (Vfvp(tt)^2) / (-2*dec_req(tt));
    elseif (Vlv(tt) > 0 && Vlvp(tt) > 0) % Case 2 LV moving , still moving
        dec_lv(tt) = Alv(tt);
        BOR = ((Vfvp(tt) - Vlvp(tt))^2) / (-2*(dec_req(tt) - dec_lv(tt)));
    elseif (Vlv(tt) > 0 && Vlvp(tt) <= 0)
        dec_lv(tt) = Alv(tt);
        BOR = (Vfvp(tt)^2)/(-2*(dec_req(tt))) -  (Vlvp(tt)^2)/(-2*(dec_lv(tt)));
    end
    
    r_w(tt) = BOR + r_d(tt);
    if(r_w(tt) < 0 )
        r_w(tt) = 0;
    end
end
end

function [Acc,Prc,TrPos,TrNeg,FlsPos,FlsNeg,gMeanTPP] = analyzeaccuracy(PredictedWrnRange, GroundTruthRw, est_separation_dist, GroundTruthSepDist)
% a,b,c,d from [Lee, Peng] UMichigan paper confusion matrix
% Confusion matrix.
%                             Actual data
%     pred                 Negative   Positive
% Negative (safe)               a       c
% Positive (threatening)        b       d
% Here we limit the search to areas where the distance between cars is in
% terms of time to crash (TODO) is less than 20m

importantRange = 200;
distancetowarning = GroundTruthSepDist - GroundTruthRw;

Acc = 0;Prc = 0;TrPos=0; TrNeg = 0; FlsPos= 0; FlsNeg = 0;

GTThreat = (GroundTruthRw>=GroundTruthSepDist);
PredThreat = (PredictedWrnRange>=est_separation_dist);
GTSafe = (GroundTruthRw<GroundTruthSepDist);
PredSafe = (PredictedWrnRange < est_separation_dist);
a = sum((GTSafe == PredSafe) & (GTSafe == 1) & (distancetowarning < importantRange));
d = sum(GTThreat == PredThreat & (GTThreat == 1) & (distancetowarning < importantRange));
b = sum((GTSafe == PredThreat) & (GTSafe == 1) & (distancetowarning < importantRange));
c = sum((GTThreat == PredSafe) & (GTThreat == 1) & (distancetowarning < importantRange)) ;

if( a+d > 0)
    Acc = (a+d)/(a+b+c+d);
end

if ( b+d > 0)
    Prc = d/(b+d);
end

if( c+d>0)
    TrPos = d/(c+d);
    FlsNeg = c/(c+d);
end

if( a+b>0 )
    TrNeg = a/(a+b);
    FlsPos= b/(a+b);
end

gMeanTPP = Acc;%(TrPos*Prc)^0.5;
% gMeanTPP = (FlsNeg*FlsPos)^0.5;

if(isnan(gMeanTPP) )
    msg = 'NaN found'
end


end

function [AminusNormal] = aminusf(v)
if (v < 6.1)
    AminusNormal = -8.8;
elseif ( v<12.2)
    AminusNormal = -5.2;
elseif ( v<18.3)
    AminusNormal = -4.4;
elseif ( v<24.4)
    AminusNormal = -2.9;
else
    AminusNormal = -2;
end

end

function [AplusNormal] = aplusf(v)
if (v < 6.1)
    AplusNormal = 7.8;
elseif ( v<12.2)
    AplusNormal = 6.7;
else
    AplusNormal = 4.8;
end

end

function filtered_acc = filterAcc(accel, TAPS)
NN = size(accel,1);
filtered_acc = accel;
timestep = 0.1;
Taprange = floor(TAPS/2);
for t=Taprange+1:NN-Taprange
    filtered_acc(t) = sum(accel(t-Taprange: t+Taprange))/(2*Taprange+1);
    %        est_x(t) = est_x(t-1) + v_lv_est(t-1)*timestep ; % constant speed model
end
end
% BSMs---------------------------------------------------------------------
% Connected V2V safety applications are built around the
% SAE J2735 BSM, which has two parts
% ? BSM Part 1:
% ? Contains the core data elements (vehicle size, position, speed, heading
% acceleration, brake system status)
% ? Transmitted approximately 10x per second
% ? BSM Part 2:
% ? Added to part 1 depending upon events (e.g., ABS activated)
% ? Contains a variable set of data elements drawn from many optional
% data elements (availability by vehicle model varies)
% ? Transmitted less frequently
% ? No on-vehicle BSM storage of BSM data
% ? The BSM is transmitted over DSRC (range ~1,000 meters)

% probTx = 1 - exp(-alpha*e(t-1));
% recvd = (rand<probTx);
%
% estimation every T second
% if recvd == 1
%     txCnt = txCnt+1;

%--------------------------------------------------------------------------
% automotive Acceleration (g)
% event 	typical car 	sports car 	F-1 race car 	large truck
% starting 	0.3–0.5 	0.5–0.9 	1.7 	< 0.2
% braking 	0.8–1.0 	1.0–1.3 	2 	~ 0.6
% cornering 	0.7–0.9 	0.9–1.0 	3 	??