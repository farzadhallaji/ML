%% Start of Program
clc
clear
close all

%% Data Loading

[imgs , labels] = readMNIST("/home/farzad/project/CI/MNIST/train-images.idx3-ubyte", "/home/farzad/project/CI/MNIST/train-labels.idx1-ubyte", 5000, 0);
x= reshape(imgs, [] , 5000);
x = 2.*x'-1 ;

% Data = Data.Data;
% 
% DD = [];
% for ii = 1:43
%     if mod(ii,4)==0
%         D = Data(1:366,ii);
%     else
%         D = Data(1:365,ii);
%     end
%     DD = [DD ; D]
% end
% 
% for ii = 366:(numel(DD)-30)
%     D_0(ii-365,1) = DD(ii-0);
%     D_1(ii-365,1) = DD(ii-1);
%     D_2(ii-365,1) = DD(ii-2);
%     D_3(ii-365,1) = DD(ii-3);
%     D_4(ii-365,1) = DD(ii-4);
%     D_5(ii-365,1) = DD(ii-5);
%     D_6(ii-365,1) = DD(ii-6);
%     D_7(ii-365,1) = DD(ii-7);
%     D_15(ii-365,1) = DD(ii-15);
%     D_30(ii-365,1) = DD(ii-30);
%     D_90(ii-365,1) = DD(ii-90);
%     D_120(ii-365,1) = DD(ii-120);
%     D_365(ii-365,1) = DD(ii-365);
%     
%     D_plus_30(ii-365,1) = DD(ii+30);
%     D_plus_1(ii-365,1) = DD(ii+30);
% end
% 
% X = [D_0  D_1  D_2  D_3  D_4  D_5  D_6  D_7  D_15  D_30  D_90  D_120  D_365];
y = labels;
% 
DataNum = size(x,1);
InputNum = size(x,2);
OutputNum = size(y,2);

% %% Normalization
% MinX = min(X);
% MaxX = max(X);
% 
% MinY = min(Y);
% MaxY = max(Y);
% 
XN = x;
YN = y;
% 
% for ii = 1:InputNum
%     XN(:,ii) = Normalize_Fcn(X(:,ii),MinX(ii),MaxX(ii));
% end
% 
% for ii = 1:OutputNum
%     YN(:,ii) = Normalize_Fcn(Y(:,ii),MinY(ii),MaxY(ii));
% end

%% Test and Train Data
TrPercent = 67;
TrNum = round(DataNum * TrPercent / 100);
TsNum = DataNum - TrNum;

R = randperm(DataNum);
trIndex = R(1 : TrNum);
tsIndex = R(1+TrNum : end);

Xtr = XN(trIndex,:);
Ytr = YN(trIndex,:);

Xts = XN(tsIndex,:);
Yts = YN(tsIndex,:);




%% Network Structure
pr = [-1 1];
PR = repmat(pr,InputNum,1);

Network = newff(PR,[10 5 OutputNum],{'tansig' 'tansig' 'tansig'});

%% Training

Network = train(Network,Xtr',Ytr');

%% Assesment
YtrNet = sim(Network,Xtr)';
YtsNet = sim(Network,Xts)';

MSEtr = mse(YtrNet - Ytr)
MSEts = mse(YtsNet - Yts)

%% Display
figure(1)
plot(Ytr,'-or');
hold on
plot(YtrNet,'-sb');
hold off

figure(2)
plot(Yts,'-or');
hold on
plot(YtsNet,'-sb');
hold off

figure(3)
t = -1:.1:1;
plot(t,t,'b','linewidth',2)
hold on
plot(Ytr,YtrNet,'ok')
hold off

figure(4)
t = -1:.1:1;
plot(t,t,'b','linewidth',2)
hold on
plot(Yts,YtsNet,'ok')
hold off





