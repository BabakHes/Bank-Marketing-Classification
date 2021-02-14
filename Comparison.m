clear all; clc; close all;

% Load the data and get testing on our original test set!
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

% Loading the models that we saved from in individual 
mymodelsLR = load('LR.mat');  % for our Logistic Regression model: mdlLR 
mymodelsLR.mdlLR

mymodelsNB = load ('NB.mat');  % for our Naive Bayes model: mdlNB 
mymodelsNB.mdlNB


%%%%%%%%%%%%%%%%%% LR %%%%%%%%%%%%%%%%%%%%%
yTestPredLR = predict(mymodelsLR.mdlLR, XTest);
yTestPredLR(yTestPredLR >= 0.5) = 1;
yTestPredLR(yTestPredLR < 0.5) = 0;

scoresLR = mymodelsLR.mdlLR.Fitted.Probability;

% Calculating confusion matrix 
predLR = categorical(yTestPredLR,[0 1],{'no' 'yes'});
confMatLR = confusionmat(yTest.y, predLR, 'Order', {'yes' 'no'} );

% Calculating ROC curve parameters to plot later
[XLR,YLR,TLR,AUCLR] = perfcurve(yTrain.y, abs(scoresLR(:,1)),'yes');

% Calculating Precision & Recall
for i =1:size(confMatLR,1) %https://uk.mathworks.com/matlabcentral/answers/262033-how-to-calculate-recall-and-precision
    precisionLR(i)=confMatLR(i,i)/sum(confMatLR(i,:));
end
precisionLR=sum(precisionLR)/size(confMatLR,1);

for i =1:size(confMatLR,1)
    recallLR(i)=confMatLR(i,i)/sum(confMatLR(:,i));
end
recallLR(isnan(recallLR))=[];

% Calculating F Score
F_scoreLR=2*recallLR*precisionLR/(precisionLR+recallLR);

% Calculating Accuracy
accuracyLR = (confMatLR(1,1) + confMatLR(2,2)) / sum(sum(confMatLR));


%%%%%%%%%%%%%%%%%% NB %%%%%%%%%%%%%%%%%%%%%
 [predNB, scoreNB, loss] = predict(mymodelsNB.mdlNB, XTest);

% Calculating confusion matrix 
confMatNB = confusionmat(yTest.y, predNB)

% Calculating ROC curve parameters to plot later
[XNB,YNB,TNB,AUCNB] = perfcurve(yTest.y, abs(scoreNB(:,2)),'yes');

% Calculating Precision & Recall
for i =1:size(confMatNB,1) %https://uk.mathworks.com/matlabcentral/answers/262033-how-to-calculate-recall-and-precision
    precisionNB(i)=confMatNB(i,i)/sum(confMatNB(i,:));
end
precisionNB=sum(precisionNB)/size(confMatNB,1);

for i =1:size(confMatNB,1)
    recallNB(i)=confMatNB(i,i)/sum(confMatNB(:,i));
end
recallNB(isnan(recallNB))=[];

% Calculating F Score
F_scoreNB=2*recallLR*precisionNB/(precisionNB+recallNB); %%F_score=2*1/((1/Precision)+(1/Recall));

% Calculating Accuracy
accuracyNB = (confMatNB(1,1) + confMatNB(2,2)) / sum(sum(confMatNB));

%%%%%%%% Ploting ROC curves

figure('pos',[1000 1000 500 400])

plot(XLR,YLR)
text(0.5,0.35,strcat('AUC (Logistic Regression)=',num2str(AUCLR)),'EdgeColor','r')
hold on

plot(XNB,YNB)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Comparison')
text(0.5,0.25,strcat('AUC (Naive Bayes)=',num2str(AUCNB)),'EdgeColor','r')
dline = refline(1,0); 
dline.Color = [169/255 169/255 169/255];
legend('Logistic Regression', 'Naive Bayes', 'Random Geuss')
hold off


%%%%%%%% Plotting confusion matrices

%Plotting confusion chart
figure('pos',[0 700 500 500])
tiledlayout(2,1)
nexttile
confusionchart(confMatLR,{'yes', 'no'}, 'RowSummary','total-normalized') 
title('Confusion Matrix for Classification by Logistic Regression')
nexttile
confusionchart(confMatNB,{'yes', 'no'}, 'RowSummary','total-normalized') 
title('Confusion Matrix for Classification by Naive Bayes')


%%%%%%%% Timing training and testing times
modelspec = 'y ~ age + job + marital + education + default + balance + housing + loan + contact + day + month + duration + campaign + pdays + previous + poutcome';
opts = statset('glmfit');
opts.MaxIter = 50;

% Creating function argument for the timeit() function
LRfitF = @() fitglm([XTest yTest], modelspec, 'Distribution','binomial','Link','logit', 'options', opts);
%NBfitF = @() fitcnb(XTest,yTest,'ClassNames',{'no','yes'},'CrossVal','on');
NBfitF = @() fitcnb(XTest,yTest);

LRpredH = @() predict(mymodelsLR.mdlLR, XTest);
%NBpredH = @() kfoldPredict(mdlNBCV);
NBpredH = @() predict(mymodelsNB.mdlNB, XTest);

% Timing using the timeit() function
TLRfit = timeit(LRfitF);
TNBfit = timeit(NBfitF);
TLRpred = timeit(LRpredH);
TNBpred = timeit(NBpredH);

%%%%%%%% Displaying metrics in the command window

fprintf('\n------------------------------------------------\n')

% Displaying metrics in the command window - Logistic Regression
fprintf('\nPerformance metrics for Logistic Regression ----\n')
fprintf('Precision        : %.3f\n', precisionLR)
fprintf('Recall           : %.3f\n', recallLR)
fprintf('F Score          : %.3f\n', F_scoreLR)
fprintf('Accuracy         : %.3f%%\n', accuracyLR*100)
fprintf('* Training and Prediction times\n')
fprintf('Training time       : %.3f\n', TLRfit)
fprintf('Prediction time     : %.3f\n', TLRpred)

% Displaying metrics in the command window - Naive Bayes 
fprintf('\nPerformance metrics for Naive Bayes -----------\n')
fprintf('Precision        : %.3f\n', precisionNB)
fprintf('Recall           : %.3f\n', recallNB)
fprintf('F Score          : %.3f\n', F_scoreNB)
fprintf('Accuracy         : %.3f%%\n', accuracyNB*100)
fprintf('* Training and Prediction times\n')
fprintf('Training time       : %.3f\n', TNBfit)
fprintf('Prediction time     : %.3f\n', TNBpred)




%% Ranking predictors in order of contribution
% https://uk.mathworks.com/help/stats/fscchi2.html
%[idx,scores] = fscchi2(X,y);
%find(isinf(scores))
%idx = idx(1:8)
%bar(scores(idx), 1)
%xlabel('Predictor name')
%ylabel('Predictor importance score')
%itle('Ranking predictors in order of contribution - top 8')
%ticklabels(strrep(whole_data_table.Properties.VariableNames(idx),'_','\_'))
%xtickangle(90)


