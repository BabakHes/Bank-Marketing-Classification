
This readme file is aged to the contents of the SUbmission folder.


**** The scripts were written using MATLAB R2020a. 


*** All scripts should run by simply clicking running the script. 


*** Submission folder contains:

-- Poster

-- Supplementary Material

-- Comparison.m: 	main script to compare the performance of trained models on our test data. Returns ROC curves comparisons, confusion matrices, and prints performance measure including accuracy and F1 Score and timing information in the command window. 
Note: There will be 'iteration limit reached' warnings where the timeit() function is used. But these do not stop the model from running. 

-- LR.m: 		clean version of our Logistic Regression model. Returns ROC curves, confusion matrix, and prints performance measure including accuracy and F1 Score.
 
-- LR.mat: 	saved trained model from the LR script.

-- LR_Lasso.m: 	performs lasso regularisation and plots dviance and trace plots. It also returns confusion matrices, and prints performance measure including accuracy and F1 Score.

-- NB.m: 		clean version of our Naives Bayes models which includes a simple model and a cross-validated model. Returns ROC curves, confusion matrices,and prints performance measure including accuracy and F1 Score comparing the two models.

-- NB.mat: 	saved trained model from the NB script.

-- PreProcessing.html:	html version of the Python jupyternotebook used to prepare the data and perform exploratory data analysis.


** Data folder contains
	- bank.csv is the unaltered dataset form UCI
	- bank_dummy_coded.csv is the dummy coded version of the dataset that was proceeded in Python







