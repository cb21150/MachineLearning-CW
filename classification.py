import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import logistic
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
# Load data
np.random.seed(0)
data = fetch_covtype()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Standardize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


clf = LogisticRegression(max_iter=10000, penalty=None, solver='lbfgs', class_weight=None) 
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# Accuracy

#train accuracy 
#print(accuracy_score(y_train, y_pred_train))
print("logistic regression accuracy score on test data:  ", accuracy_score(y_test, y_pred))









np.random.seed(0)
# Load dataset
data = fetch_covtype()
X, y = data.data, data.target
#split data 80 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Load dataset
data = fetch_covtype()
X, y = data.data, data.target
#split data 80 20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train decision tree
dt = DecisionTreeClassifier(criterion='entropy', )
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Accuracy for decision tree on test set:", accuracy_score(y_test, y_pred))



# Load dataset
np.random.seed(0)
data = fetch_covtype()
X, y = data.data, data.target

# Split data 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''
# Cross-validation setup
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
'''
bootstrap_size = int(len(y_train) * 0.63)
# Bagging setup
num_models = 500
sample_size = bootstrap_size # Training set size.
feature_sample_size = 45
np.random.seed(0)

def bagging_predict(test_data, all_models):
    votes = np.zeros((test_data.shape[0], len(all_models))) 
    combined_predictions = np.zeros(test_data.shape[0])

    for idx, m in enumerate(all_models):
        votes[:, idx] = m.predict(test_data)
        
    for test_point in range(votes.shape[0]):
        combined_predictions[test_point] = np.bincount(np.int64(votes[test_point])).argmax()
    
    return combined_predictions

'''
# Perform cross-validation
cv_accuracies = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    all_models = []
    for m in range(num_models):
        # Sample with replacement from the training fold
        sample_idx = np.random.choice(X_train_fold.shape[0], sample_size)
        X_train_sample, y_train_sample = X_train_fold[sample_idx], y_train_fold[sample_idx]

        # Create a decision tree classifier with limited features for each split
        model = DecisionTreeClassifier(max_features=feature_sample_size, criterion='entropy')
        
        # Train the model on the random sample
        model.fit(X_train_sample, y_train_sample)
        all_models.append(model)

    # Validate using the current fold's validation data
    val_prediction = bagging_predict(X_val_fold, all_models)
    val_accuracy = accuracy_score(y_val_fold, val_prediction)
    cv_accuracies.append(val_accuracy)

# Calculate the average accuracy from cross-validation
average_cv_accuracy = np.mean(cv_accuracies)
print("Average cross-validation accuracy: {:.2f}%".format(average_cv_accuracy * 100))
'''
# Train all models using the entire training set and evaluate on the test set
all_models = []
for m in range(num_models):
    sample_idx = np.random.choice(X_train.shape[0], sample_size)
    X_train_sample, y_train_sample = X_train[sample_idx], y_train[sample_idx]

    model = DecisionTreeClassifier(max_features=feature_sample_size, criterion='entropy')
    model.fit(X_train_sample, y_train_sample)
    all_models.append(model)

# Evaluate on the test set
test_prediction = bagging_predict(X_test, all_models)
test_accuracy = accuracy_score(y_test, test_prediction)
print("random forest  accuracy on test set: {:.2f}%".format(test_accuracy * 100))
