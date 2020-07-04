clc; clear all;close all;

training=readtable('Training_set.csv'); % load training data
test=readtable('Testing_set.csv');% load testing data

test=table2array(test); %convert to matrix
training_label=table2array(training(:,end)); %get label


%find LS vectors for each Gre scores and Toefl scores 

[eqn traindata G0 G1]=findLSvectors(training,1); %for plots change to 1


% find weights
[xm]=makpred(traindata,eqn); 
X=[ones(length(xm),1) xm ];
wml=(X.'*X)\(X.'*training_label(:)); %find wml


%prepare for final prediction
preds1=makpred(test,eqn);
preds=[ones(length(preds1),1) preds1];

%make predictions
Predictions=preds*wml;
%find mean squared error
squared_error=(Predictions-test(:,end)).^2;
mse=(sum((Predictions-test(:,end)).^2)/length(Predictions));



%table for visualized results
Results=[Predictions test(:,end) squared_error];
Results=array2table(Results,...
    'VariableNames',{'Predictions','Label','squared_error'});

WeightVectors=array2table(eqn',...
    'VariableNames',{'G1_Gre','G1_Toefl','G0_Gre','G0_Toefl'});


%store group data

G0=array2table(G0,...
    'VariableNames',{'Gre_scores','Toefl_scores','without_research','Admission_chance'});

G1=array2table(G1,...
    'VariableNames',{'Gre_scores','Toefl_scores','with_research','Admission_chance'});






function [re ]=makpred(te,eqn)
eq1=eqn(1,:);
eq2=eqn(2,:);
eq3=eqn(3,:);
eq4=eqn(4,:);

a1=[];
a2=[];
% apply y=ax+b to make predictions 
for i =1:length(te)
    
    if te(i,3)==1 %Predictions for students with research experience
w1=te(i,1).*eq1(1)+eq1(2); % ax+b for Gre scores
a1=[a1; w1 ]; %store predictions
w2=te(i,2).*eq2(1)+eq2(2); % ax+b for Toefl scores
a2=[a2 ;w2 ];
    else  
      %Predictions for students without research experience  
w1=te(i,1).*eq3(1)+eq3(2); % ax+b for Gre scores
a1=[a1; w1 ];

w2=te(i,2).*eq4(1)+eq4(2);% ax+b for Toefl scores

a2=[a2 ;w2 ];     
        
    end



end
re=[a1 a2];%store all final predictions

end



function [eqn train_data G0 G1]=findLSvectors(q,i)


t3=logical(q.Var3); %make third column into a logical array


train_data=table2array(q); 

G1=train_data(t3,:); %group with research experience
G0=train_data(~t3,:); %group without research experience

%sort from least to greatest with admission chance for all elements

[G1_1r]=newsort1(G1,1); %sort for the first row
[G1_2r ]=newsort1(G1,2); %sort for the second row

[G0_1r]=newsort1(G0,1); %sort for the first row
[G0_2r]=newsort1(G0,2);%sort for the second row

% Find equations for y=ax+b 
%equations for students with research experience
eq1= linreg(G1_1r(:,1),G1_1r(:,end),i,1,3);
eq2= linreg(G1_2r(:,2),G1_2r(:,end),i,1,4);

%equations for students without research experience
eq3= linreg(G0_1r(:,1),G0_1r(:,end),i,1,3);
eq4= linreg(G0_2r(:,2),G0_2r(:,end),i,1,4);

eqn=[eq1;eq2;eq3;eq4]; %store equations

end



%this function finds the least squares line
function  [out] = linreg(xm,fm,i,a1,b1)

%store strings in a cell to be used for making plots
s={'Chance of admission with research experience'...
    'Chance of admission without research experience'...
    'GRE scores','TOEFL scores' };

N=length(xm);
X=[ones(N,1) xm(:) ];
m=(X.'*X)\(X.'*fm(:)); 



%find the a and b for ax+b=y

b=m(1);%b
a=m(2);%a

xa=min(xm);
xb=max(xm);
x=linspace(xa, xb, N);%make x vector to draw the regression line
y=a*x+b; % Find the y values for the regrssion line

% Plot
if i==1
figure,scatter(xm,fm);
title(sprintf('Regression line for %s and %s ',s{1,b1},s{1,a1}))
xlabel(sprintf('%s',s{1,b1}))
ylabel(sprintf('%s',s{1,a1}))
hold on
plot(x,y,'-r');
hold off 
end

%store output
out=[a b];

end




%this function sorts a row in a matrix and saves the mean of the other rows
function [out] =newsort1(train_data,c)
% c represents the row
qa=train_data(:,c); %row to be sorted
a=[];
i=1;
%b=[];


while i==1
    c=(qa==max(qa));%finds maximum 
   a=[a;mean(train_data(c,:),1)];
   qa=qa.*(~c);%remove the maximum
if sum(qa)==0 %
   i=0; 
end

end
out=a(end:-1:1,:); %sort from least to greatest
end

















