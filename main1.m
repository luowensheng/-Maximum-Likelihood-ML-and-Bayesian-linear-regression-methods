clc
clear all
Training_data=load('Training_set.csv');
Testing_data=load('Testing_set.csv');
T_label=Testing_data(:,end);

Results=[];
UI={}; %cell to store ui values
UJ={};%cell to store uj values

Squared_Error={}; % cell to store thr squared error
S={}; % cell store s1 and s2 value


 minSoSE=100;%intitial to find the minimum error
 Results_Qxn={}; %cell to store phi(xn) values
 Results_Wml={}; %cell to store Wml values
 
 
 
 
for O1=2:6
for O2=2:6
[ ui uj s1 s2]=getcomponent(Training_data,O1,O2);

Qxn=featvec(Training_data,O1,O2,ui, uj, s1, s2);% Get feature vector for training data
Results_Qxn{O1,O2}=Qxn;
wml=inv(Qxn'*Qxn)*Qxn'*Training_data(:,end); % Use Maximum liklihood to find Wml
 Results_Wml{O1,O2}=wml;
clc %to remove the warnings from the screen

UI{1,O1}=[ui]; %store values for ui
UJ{1,O2}=[uj];%store values for uj
S=[S; s1 s2];%store values for s1 and s2
test=featvec(Testing_data,O1,O2,ui, uj, s1, s2);% Get feature vector for Test data


Prediction=(test*wml);
error=(Prediction-Testing_data(:,end)).^2;
Squared_Error{O1,O2}=error; %the columns==O1 value and row==O2 value
SoSquaredError=sum(error)/length(error); %compute the squared error

% Find minimum squared error and when it occurs 
if SoSquaredError< minSoSE
    minSoSE=SoSquaredError; %minimum error
   O=[O1 O2]; %store O1 and O2 values
end

Results=[Results; O1 O2 SoSquaredError ]; %store results

end
end


plot([1:1:length(Results)-7], Results(1:end-7,end))
ylabel('mse')
xlabel('itiration')
title(' How the mse varies after each iterarion of O1 and O2')
xticks([1:1:18])


Results= array2table(Results,...
    'VariableNames',{'O1','O2','Squared_error'}); %table showing the results

fprintf('The minimun squared error is %s and it occurs when the O1=%d and O2=%d',minSoSE,O(1),O(2)) 


%this function finds ui uj s1 s2

function [ ui uj s1 s2]=getcomponent(A,O1,O2)

X1=A(:,1);
X2=A(:,2);

x1_max=max(X1);
x1_min=min(X1);

x2_max=max(X2);
x2_min=min(X2);

ui=[];
uj=[];

% Itirate to find the mean vector for ui
for i=1:O1
    ux=((x1_max-x1_min)/(O1-1))*(i-1);
ui=[ui ux];   
end 

% Itirate to find the mean vector for uj
for j=1:O2
uy=((x2_max-x2_min)/(O2-1))*(j-1);
uj=[uj uy];
     
end
s1=(x1_max-x1_min)/(O1-1);
s2=(x2_max-x2_min)/(O2-1);

end

%this function creates a feature vector

function Qxn=featvec(A,O1,O2,ui, uj, s1, s2)
[m n]=size(A);

X1=A(:,1); %collect first row
X2=A(:,2);% collect second row

px=zeros(size(A)); % create empty matrix to store results

% Find the gaussian basis function 
for i=1:length(ui)
    
    for j=1:length(uj)
        
k=O2*(i-1)+j;

px(:,k)=exp(-((X1-ui(i)).^2)/(2*s1^2) - ((X2-uj(j)).^2)/(2*s2^2) );

    end
end

Qxn=[px A(:,3) ones(m,1)]; %store final rows of the feature vector

end



 



