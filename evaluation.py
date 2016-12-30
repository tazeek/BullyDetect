import time
import os
import pickle

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, log_loss, brier_score_loss

def evaluatingModel(model, model_name, X, y, skv, file_name):

	# Create Confusion Matrix Dictionary
	cm_dict = { "tp": 0, "fp": 0, "tn": 0, "fn": 0}

	print("CLASSIFICATION FOR " + model_name + " STARTS HERE\n\n")

	# Array to store results
	accuracy_array = []
	precision_array = []
	fpr_array = []
	auc_array = []
	log_loss_array = []
	brier_array = []
	execution_time_array = []

	for train_cv, test_cv in skv.split(X,y):

		# Seperate the training and testing fold
		# NOTE: y_test corresponds to y_true
		X_train, X_test = X[train_cv], X[test_cv]
		y_train, y_test = y[train_cv], y[test_cv]

		# Train the model
		model.fit(X_train , y_train)

		# Predict and calculate run-time
		# NOTE: result corresponds to y_pred
		start = time.time()
		result = model.predict(X_test)
		end = time.time()

		execution_time = end - start

		# Get the probability scores
		# Use Logistic Regression for LinearSVC case
		if model_name == 'SVM':

			lr = LogisticRegression()
			lr.fit(X_train, y_train)

			y_scores = lr.predict_proba(X_test)

		else:

			y_scores = model.predict_proba(X_test)

		# Get AUC score, Log Loss
		auc_score = roc_auc_score(y_test, y_scores[:, 1])
		log_loss_score = log_loss(y_test, y_scores)
		brier_score = brier_score_loss(y_test, y_scores[:, 1])

		# Confusion Matrix
		tn, fp, fn, tp = confusion_matrix(y_test, result).ravel()

		# Add the results to confusion matrix
		cm_dict["tn"] += tn
		cm_dict["fp"] += fp 
		cm_dict["fn"] += fn 
		cm_dict["tp"] += tp

		# Evaluation Metrics
		accuracy = accuracy_score(y_test , result)
		precision = tp/(tp+fp)
		fpr = fp/(fp + tn) # False Positive Rate

		# Append results
		accuracy_array.append(accuracy)
		precision_array.append(precision)
		fpr_array.append(fpr)
		auc_array.append(auc_score)
		log_loss_array.append(log_loss_score)
		brier_array.append(brier_score)
		execution_time_array.append(execution_time)

	# Get mean results
	mean_accuracy = np.mean(accuracy_array)
	mean_precision = np.mean(precision_array)
	mean_fpr = np.mean(fpr_array)
	mean_auc = np.mean(auc_array)
	mean_log_loss = np.mean(log_loss_array)
	mean_brier = np.mean(brier_array)
	mean_execution_time = np.mean(execution_time_array)

	# Get standard deviation (population)
	accuracy_std = np.std(accuracy_array)
	precision_std = np.std(precision_array)
	fpr_std = np.std(fpr_array)
	auc_std = np.std(auc_array)
	log_std = np.std(log_loss_array)
	brier_std = np.std(brier_array)
	run_std = np.std(mean_execution_time)

	# Display results
	print("MEAN ACCURACY: %0.2f (+/- %0.2f) \n" % (mean_accuracy, accuracy_std))
	print("MEAN PRECISION: %0.2f (+/- %0.2f) \n" % (mean_precision, precision_std))
	print("MEAN FALSE POSITIVE RATE: %0.2f (+/- %0.2f) \n" % (mean_fpr, fpr_std))
	print("MEAN AUC SCORE: %0.2f (+/- %0.2f) \n" % (mean_auc, auc_std))
	print("MEAN LOG LOSS SCORE: %0.2f (+/- %0.2f) \n" % (mean_log_loss, log_std))
	print("MEAN BRIER SCORE LOSS: %0.2f (+/- %0.2f) \n" % (mean_brier, brier_std))
	print("MEAN RUN TIME: %0.2f (+/- %0.2f) \n" % (mean_execution_time, run_std))

	print("\n\nCLASSIFICATION FOR " + model_name + " STOPS HERE\n\n")

	# Save the confusion matrix using pickle
	FILE = "Confusion Matrix (Balanced)/" + model_name.lower() + "_" + file_name + ".pk"
	pickle.dump(cm_dict, open(FILE, "wb"))

def evaluate(X, y, file_name):

	# Implement Classifier(s) here and store in dictionary
	print("INITLIAZING CLASSIFIERS \n\n")
	nb = GaussianNB()
	rf = RandomForestClassifier(n_estimators=100)
	svm = LinearSVC()

	# Store them in a dicitonary
	models = { "NB": nb, "SVM": svm, "RF": rf}

	# Initialize directory
	DIR = "ML Models/"


	# Test with 10 fold Cross validation/Stratified K Fold
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	for key, value in models.items():

		#evaluatingModel(value, key, X, y, skf, file_name)
		FILE = DIR + key.lower() + "_" + file_name + ".sav"

		# Train the model fully
		value.fit(X,y)

		# Save the model 
		pickle.dump(key, open(FILE, 'wb'))