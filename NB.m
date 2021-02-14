clear all; clc; close all;

currentFolder = pwd;
whole_data_table = readtable(sprintf('%s/Data/bank.csv', pwd));

% Data preparation
% Normalizing number variables
%whole_data_table.age = discretize(whole_data_table.age, 5);
%whole_data_table.balance = discretize(whole_data_table.balance, 5);
%whole_data_table.day = discretize(whole_data_table.day, 5);
%whole_data_table.duration = discretize(whole_data_table.duration, 5);
%whole_data_table.campaign = discretize(whole_data_table.campaign, 5);
%whole_data_table.previous = discretize(whole_data_table.previous, 5);

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
%%% SIMPLE MODEL TRAINING (NO CROSS-VALIDATION) %%%
mdlNB = fitcnb(XTrain, yTrain);
[pred, score, loss] = predict(mdlNB, XTrain);

% Calculating confusion matrix and plotting confusion chart
confMat = confusionmat(yTrain.y, pred);

% Calculating ROC curve parameters to plot later
[X,Y,T,AUC] = perfcurve(yTrain.y, abs(score(:,2)),'yes');

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
fprintf('Performance metrics for non cross validated model ----\n')
fprintf('Precision        : %.3f\n', precision)
fprintf('Recall           : %.3f\n', recall)
fprintf('F Score          : %.3f\n', F_score)
fprintf('Accuracy         : %.3f%%\n', accuracy*100)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MODEL WITH CROSS-VALIDATION %%%

mdlNBCV=fitcnb(XTrain,yTrain,'ClassNames',{'no','yes'},'CrossVal','on');
[pred, score] = kfoldPredict(mdlNBCV);

% Calculating confusion matrix and plotting confusion chart
predC = categorical(pred);
confMatCV = confusionmat(yTrain.y, predC, 'order', {'yes', 'no'});

% Plotting ROC curves
[Xcv,Ycv,Tcv,AUCcv] = perfcurve(yTrain.y, abs(score(:,2)),'yes');
figure('pos',[1000 1000 500 400])
plot(X,Y)
text(0.5,0.35,strcat('AUC=',num2str(AUC)),'EdgeColor','r')
hold on
plot(Xcv,Ycv)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes')
text(0.5,0.25,strcat('AUC (with cv)=',num2str(AUCcv)),'EdgeColor','r')
dline = refline(1,0); 
dline.Color = [169/255 169/255 169/255];
legend('Naive Bayes', 'Naive Bayes - with crossval', 'Random Geuss')
hold off

% Calculating Precision & Recall
for i =1:size(confMatCV,1) %https://uk.mathworks.com/matlabcentral/answers/262033-how-to-calculate-recall-and-precision
    precisionCV(i)=confMatCV(i,i)/sum(confMatCV(i,:));
end
precisionCV=sum(precisionCV)/size(confMatCV,1);

for i =1:size(confMatCV,1)
    recallCV(i)=confMatCV(i,i)/sum(confMatCV(:,i));
end
recallCV(isnan(recallCV))=[];

% Calculating F Score
F_scoreCV=2*recall*precisionCV/(precisionCV+recallCV); %%F_score=2*1/((1/Precision)+(1/Recall));

% Calculating Accuracy
genError = kfoldLoss(mdlNBCV);
accuracyCV = 1 - kfoldLoss(mdlNBCV, 'LossFun', 'ClassifError');

% Displaying metrics in the command window
fprintf('\nPerformance metrics for cross validated model ----\n')
fprintf('Precision        : %.3f\n', precisionCV)
fprintf('Recall           : %.3f\n', recallCV)
fprintf('F Score          : %.3f\n', F_scoreCV)
fprintf('Accuracy         : %.3f%%\n', accuracyCV*100)

%Plotting confusion chart
figure('pos',[0 1000 500 1200])
tiledlayout(2,1)
nexttile
confusionchart(confMat,{'yes', 'no'}) 
title('Confusion Matrix for Classification by Naive Bayes - No Cross Validation')
nexttile
confusionchart(confMatCV,{'yes', 'no'}) 
title('Confusion Matrix for Classification by Naive Bayes - With Cross Validation')

save('NB.mat','mdlNB')