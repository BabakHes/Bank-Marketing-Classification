clear all; clc; close all;

currentFolder = pwd;
whole_data_table = readtable(sprintf('%s/Data/bank.csv', pwd));

% Data preparation
% Normalizing number variables
whole_data_table.age = zscore(whole_data_table.age);
whole_data_table.balance = zscore(whole_data_table.balance);
whole_data_table.day = zscore(whole_data_table.day);
whole_data_table.duration = zscore(whole_data_table.duration);
whole_data_table.campaign = zscore(whole_data_table.campaign);
whole_data_table.previous = zscore(whole_data_table.previous);

% Transforming categorial variables into categorical types
whole_data_table.y = categorical(whole_data_table.y);
whole_data_table.job = categorical(whole_data_table.job);
whole_data_table.marital = categorical(whole_data_table.marital);
whole_data_table.education = categorical(whole_data_table.education);
whole_data_table.default = categorical(whole_data_table.default);
whole_data_table.housing = categorical(whole_data_table.housing);
whole_data_table.loan = categorical(whole_data_table.loan);
whole_data_table.contact = categorical(whole_data_table.contact);
whole_data_table.month = categorical(whole_data_table.month);
whole_data_table.poutcome = categorical(whole_data_table.poutcome);

% There is massive class imbalance in the data set. Here I have
% undersampled the over-represented class ('no'). I exlpain more in poster.
data_yes = whole_data_table(whole_data_table.y == {'yes'}, :); % https://uk.mathworks.com/help/stats/datasample.html
data_no = datasample(whole_data_table(whole_data_table.y == {'no' }, :), 521 ,'Replace', false);
whole_data_table = vertcat(data_yes, data_no);

% Seperating predictors from the dependant variable
[m,n] = size(whole_data_table);
X = whole_data_table(:,1:n-1);
y = whole_data_table(:,n);

% Splitting the data into training and testing sets
rng 'default';
P = 0.7; % training data is 70% (test data is 30%)
idx = randperm(m);
XTrain = X(idx(1:round(P*m)),:);
yTrain = y(idx(1:round(P*m)),:);
XTest = X(idx(round(P*m)+1:end),:);
yTest = y(idx(round(P*m)+1:end),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Main Logistic Regression model %%%

% This method of using the formula input in the fitglm() function is a
% substitute for dummy coding. I tried both and the results were the same.
modelspec = 'y ~ age + job + marital + education + default + balance + housing + loan + contact + day + month + duration + campaign + pdays + previous + poutcome';
mdlLR = fitglm([XTrain yTrain], modelspec, 'Distribution','binomial','Link','logit');
scores = mdlLR.Fitted.Probability;
 
yTrainPred = predict(mdlLR, XTrain);
yTrainPred(yTrainPred >= 0.5) = 1;
yTrainPred(yTrainPred < 0.5) = 0;

% Calculating confusion matrix and plotting confusion chart
predC = categorical(yTrainPred,[0 1],{'no' 'yes'});
confMat = confusionmat(yTrain.y, predC, 'Order', {'yes' 'no'} );
figure('pos',[0 1000 500 400])
confusionchart(confMat,{'yes', 'no'}) 

% Plotting ROC curves
[X,Y,T,AUC] = perfcurve(yTrain.y, abs(scores(:,1)),'yes');

figure('pos',[1000 1000 500 400])
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes')
text(0.5,0.25,strcat('AUC=',num2str(AUC)),'EdgeColor','r')
dline = refline(1,0); 
dline.Color = [169/255 169/255 169/255];
legend('Logistic Regression', 'Random Geuss')

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
%%
save('LR.mat','mdlLR')