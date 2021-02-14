clear all; clc; close all;
% To use the lassoglm() function we need to to use matrices. 

currentFolder = pwd;
whole_data = readmatrix(sprintf('%s/Data/bank_dummy_coded.csv', pwd));
%This is the exact data set as bank.csv, but it was dummy coded in Python

% There is massive class imbalance in the data set. Here I have
% undersampled the over-represented class ('no'). I exlpain more in poster.
data_yes = whole_data(whole_data(:,2:2) == 1, :); % https://uk.mathworks.com/help/stats/datasample.html
data_no = datasample(whole_data(whole_data(:, 2:2) == 0, :), 521 ,'Replace', false);
whole_data = vertcat(data_yes, data_no);

% Seperating predictors from the dependant variable
X = whole_data(:,3:44);
y = whole_data(:,2:2);

% Splitting the data into training and testing sets
rng 'default';
[m,n] = size(whole_data);
P = 0.7; % proportion of training data (30% for the test data)
idx = randperm(m);
XTrain = X(idx(1:round(P*m)),:);
yTrain = y(idx(1:round(P*m)),:);
XTest = X(idx(round(P*m)+1:end),:);
yTest = y(idx(round(P*m)+1:end),:);

% Performing lasso regularisation assuming y is binomial find the
% coefficients corresponding to the Lambda with minimum expected variance
[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial',... % https://uk.mathworks.com/help/stats/regularize-logistic-regression.html
    'NumLambda',10,'CV',10);

% Plot a Deviance Plot
lassoPlot(B, FitInfo, 'PlotType', 'CV');
legend('show', 'Location', 'best') 

% Plot a Trace Plot of coefficients found by Lasso for each predictor as a function of Lambda
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');
ylabel('Coefficient');


% Predicting using the model coefficients found using lasso
idxLambdaMinDeviance = FitInfo.IndexMinDeviance; %https://uk.mathworks.com/help/stats/lassoglm.html
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)]
yhat = glmval(coef,XTrain,'logit');
yhatBinom = (yhat>=0.5);

% Calculating confusion matrix and plotting confusion chart
figure('pos',[1000 1000 500 400])
predC = double(yhatBinom);
grouporder = str2double({'0' '1'})
confMat = confusionmat(yTrain, predC, 'Order', grouporder );
confusionchart(confMat,{'yes', 'no'}) 

% Calculating Precision & Recall
for i =1:size(confMat,1) %https://uk.mathworks.com/matlabcentral/answers/262033-how-to-calculate-recall-and-precision
    precision(i)=confMat(i,i)/sum(confMat(i,:));
end
precision=sum(precision)/size(confMat,1);

for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(:,i));
end
recall(isnan(recall))=[];

% Calculating F Score
F_score=2*recall*precision/(precision+recall);

% Calculating Accuracy
accuracy = (confMat(1,1) + confMat(2,2)) / sum(sum(confMat));

% Displaying metrics in the command window
fprintf('Performance metrics -----------\n')
fprintf('Precision        : %.3f\n', precision)
fprintf('Recall           : %.3f\n', recall)
fprintf('F Score          : %.3f\n', F_score)
fprintf('Accuracy         : %.3f%%\n', accuracy*100)
