import time

import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def evaluatingModel(model, model_name, X, y, cv):

	print(model_name + " STARTS HERE\n\n")

	# Array to store results
	accuracy_array = []
	precision_array = []
	fdr_array = []
	fpr_array = []
	auc_array = []
	execution_time_array = []

	for train_cv, test_cv in cv.split(X,y):

		# Seperate the training and testing fold
		# NOTE: y_test corresponds to y_true
		X_train, X_test = X[train_cv], X[test_cv]
		y_train, y_test = y[train_cv], y[test_cv]

		# Train the model
		model.fit(X_train , y_train)

		# Predict and calculate run-time
		start = time.time()
		result = model.predict(X_test)
		end = time.time()

		execution_time = end - start

		# Get the probability scores
		# SVM uses 'decision_function'. Rest uses predict_proba
		# When using predict_proba, take precaution on the index
		if model_name == 'SVM':
			y_scores = model.decision_function(X_test)
		else:
			y_scores = model.predict_proba(X_test)[:, 1]

		# Get AUC score
		auc_score = roc_auc_score(y_test, y_scores)

		# Confusion Matrix
		cm = confusion_matrix(y_true=y_test, y_pred=result)

		tp = cm[1][1] # True positives
		fp = cm[0][1] # False positives
		tn = cm[0][0] # True negatives
		fn = cm[1][0] # False negatives

		# Evaluation Metrics
		accuracy = accuracy_score(y_true=y_test , y_pred=result)
		precision = tp/(tp+fp)
		fdr = 1 - precision # False Discovery Rate
		fpr = fp/(fp + tn) # False Positive Rate

		# Append results
		accuracy_array.append(accuracy)
		precision_array.append(precision)
		fdr_array.append(fdr)
		fpr_array.append(fpr)
		auc_array.append(auc_score)
		execution_time_array.append(execution_time)

	# Get mean results
	mean_accuracy = np.mean(accuracy_array)
	mean_precision = np.mean(precision_array)
	mean_fdr = np.mean(fdr_array)
	mean_fpr = np.mean(fpr_array)
	mean_auc = np.mean(auc_array)
	mean_execution_time = np.mean(execution_time_array)

	# Get standard deviation (population)
	accuracy_std = np.std(accuracy_array)
	precision_std = np.std(precision_array)
	fdr_std = np.std(fdr_array)
	fpr_std = np.std(fpr_array)
	auc_std = np.std(auc_array)
	run_std = np.std(mean_execution_time)

	# Display results
	print("MEAN ACCURACY: %0.2f (+/- %0.2f) \n" % (mean_accuracy*100, accuracy_std * 100))
	print("MEAN PRECISION: %0.2f (+/- %0.2f) \n" % (mean_precision*100, precision_std * 100))
	print("MEAN FALSE DISCOVERY RATE: %0.2f (+/- %0.2f) \n" % (mean_fdr*100, fdr_std*100))
	print("MEAN FALSE POSITIVE RATE: %0.2f (+/- %0.2f) \n" % (mean_fpr*100, fpr_std*100))
	print("MEAN AUC SCORE: %0.2f (+/- %0.2f) \n" % (mean_auc, auc_std))
	print("MEAN RUN TIME: %0.2f (+/- %0.2f) \n" % (mean_execution_time, run_std * 100))

	print("\n\n" + model_name + " STOPS HERE\n\n")