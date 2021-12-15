import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')


# Correlation analysis
train_data['Dependents'].replace('3+', 3, inplace=True)

test_data['Dependents'].replace('3+', 3, inplace=True)

train_data['Loan_Status'].replace('N', 0, inplace=True)
train_data['Loan_Status'].replace('Y', 1, inplace=True)

train_data['Education'].replace('Not Graduate', 0, inplace=True)
train_data['Education'].replace('Graduate', 1, inplace=True)

train_data['Gender'].replace('Male', 0, inplace=True)
train_data['Gender'].replace('Female', 1, inplace=True)

train_data['Self_Employed'].replace('No', 0, inplace=True)
train_data['Self_Employed'].replace('Yes', 1, inplace=True)


train_data['Married'].replace('No', 0, inplace=True)
train_data['Married'].replace('Yes', 1, inplace=True)

train_data['Property_Area'].replace('Rural', 0, inplace=True)
train_data['Property_Area'].replace('Semiurban', 1, inplace=True)
train_data['Property_Area'].replace('Urban', 2, inplace=True)


# replace nan values
train_data['Dependents'].replace(np.nan, round(train_data['Dependents'][train_data['Dependents'].notnull()].astype(int).mean()), inplace=True)
test_data['Dependents'].replace(np.nan, round(test_data['Dependents'][test_data['Dependents'].notnull()].astype(int).mean()), inplace=True)
train_data['Credit_History'].replace(np.nan, round(train_data['Credit_History'][train_data['Credit_History'].notnull()].astype(int).mean()), inplace=True)
test_data['Credit_History'].replace(np.nan, round(test_data['Credit_History'][test_data['Credit_History'].notnull()].astype(int).mean()), inplace=True)
train_data['Gender'].replace(np.nan, train_data['Gender'].mode()[0], inplace=True)
test_data['Gender'].replace(np.nan, test_data['Gender'].mode()[0], inplace=True)
train_data['LoanAmount'].replace(np.nan, round(train_data['LoanAmount'][train_data['LoanAmount'].notnull()].astype(int).mean()), inplace=True)
test_data['LoanAmount'].replace(np.nan, round(test_data['LoanAmount'][test_data['LoanAmount'].notnull()].astype(int).mean()), inplace=True)
train_data['Loan_Amount_Term'].replace(np.nan, round(train_data['Loan_Amount_Term'][train_data['Loan_Amount_Term'].notnull()].astype(int).mean()), inplace=True)
test_data['Loan_Amount_Term'].replace(np.nan, round(test_data['Loan_Amount_Term'][test_data['Loan_Amount_Term'].notnull()].astype(int).mean()), inplace=True)
train_data['Self_Employed'].replace(np.nan, train_data['Self_Employed'].mode()[0], inplace=True)
test_data['Self_Employed'].replace(np.nan, test_data['Self_Employed'].mode()[0], inplace=True)
train_data['Married'].replace(np.nan, train_data['Married'].mode()[0], inplace=True)

X = train_data.copy()
Y = train_data["Loan_Status"]

# drop reduntant columns
X = X.drop("Loan_ID", axis=1)
X = X.drop("Loan_Status", axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

logistic_model = LogisticRegression(random_state=1, solver='newton-cg', penalty='l2')

logistic_model.fit(x_train, y_train)

pred_logistic = logistic_model.predict(x_test)
score_logistic = accuracy_score(pred_logistic, y_test) * 100
matrix_logistic = confusion_matrix(pred_logistic, y_test)
recall_score_logistic = recall_score(y_test, pred_logistic) * 100
precision_score_logistic = precision_score(y_test, pred_logistic) * 100

print("Logistic regression", score_logistic)

tree_model = DecisionTreeClassifier(random_state=1, criterion="entropy", splitter="random", max_depth=3)
tree_model.fit(x_train, y_train)

pred_tree = tree_model.predict(x_test)

score_tree = accuracy_score(pred_tree, y_test) * 100
matrix_tree = confusion_matrix(pred_tree, y_test)
recall_score_tree = recall_score(y_test, pred_tree) * 100
precision_score_tree = precision_score(y_test, pred_tree) * 100
print("Decision trees", score_tree)


from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(random_state=1, max_depth=6, n_estimators=20)
forest_model.fit(x_train, y_train)

pred_forest = forest_model.predict(x_test)
score_forest = accuracy_score(pred_forest, y_test) * 100
matrix_forest = confusion_matrix(pred_forest, y_test)
recall_score_forest = recall_score(y_test, pred_forest) * 100
precision_score_forest = precision_score(y_test, pred_forest) * 100
print("Forest with experimental values", score_forest)


# Find out the optimized values

# from sklearn.model_selection import GridSearchCV
# params_grid_search = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

# grid_search = GridSearchCV(RandomForestClassifier(random_state=1), params_grid_search)
#
# grid_search.fit(x_train, y_train)
# grid_search.best_estimator_
# obtained optimum solution RandomForestClassifier(max_depth=11, n_estimators=141, random_state=1)

grid_forest_model = RandomForestClassifier(random_state=1, max_depth=11, n_estimators=141)
grid_forest_model.fit(x_train, y_train)
pred_grid_forest = grid_forest_model.predict(x_test)
score_grid_forest = accuracy_score(pred_grid_forest, y_test)*100
matrix_grid_forest = confusion_matrix(pred_grid_forest, y_test)
recall_score_grid_forest = recall_score(y_test, pred_grid_forest) * 100
precision_score_grid_forest = precision_score(y_test, pred_grid_forest) * 100
print("Forest with grid search values", score_grid_forest)

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), SVC(random_state=1, gamma='auto', kernel="rbf", class_weight=None))
clf.fit(x_train, y_train)
pred_svm = clf.predict(x_test)
score_svm = accuracy_score(pred_svm, y_test) * 100
matrix_svm = confusion_matrix(pred_svm, y_test)
recall_score_svm = recall_score(y_test, pred_svm) * 100
precision_score_svm = precision_score(y_test, pred_svm) * 100
print("Support vector machines", score_svm)


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
pred_bayes = bnb.fit(x_train, y_train).predict(x_test)
score_bayes = accuracy_score(pred_bayes, y_test) * 100
matrix_bayes = confusion_matrix(pred_bayes, y_test)
recall_score_bayes = recall_score(y_test, pred_bayes) * 100
precision_score_bayes = precision_score(y_test, pred_bayes) * 100
print("Naive Bayes", score_bayes)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=45)
neigh.fit(x_train, y_train)
neigh_pred = neigh.predict(x_test)
score_neigh = accuracy_score(neigh_pred, y_test) * 100
matrix_neigh = confusion_matrix(neigh_pred, y_test)
recall_score_neigh = recall_score(y_test, neigh_pred) * 100
precision_score_neigh = precision_score(y_test, neigh_pred) * 100
print("KNN", score_neigh)


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
# from keras.losses import
# define the keras model
model = Sequential()
model.add(Dense(256, input_dim=11, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model

checkpoint_filepath = './checkpoint'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(np.asarray(x_train).astype('float32'), y_train, epochs=1000, batch_size=50, callbacks=[model_checkpoint_callback])
nn_loss = model.history.history['loss']
nn_acc = model.history.history['accuracy']
# model.load_weights(checkpoint_filepath)
# evaluate the keras model
_, accuracy = model.evaluate(np.asarray(x_test).astype('float32'), y_test)
score_nn = accuracy*100
print('Accuracy: %.2f' % (accuracy*100))

plt.plot(nn_loss[5:])
plt.title("NN Loss")
plt.show()

plt.plot(nn_acc[5:])
plt.title("NN Accuracy")
plt.show()

importance = pd.Series(forest_model.feature_importances_, index=X.columns)
importance.plot(kind='barh', figsize=(12, 8))
plt.title("Feature Importance")
plt.show()

confusion_matrix_df = pd.DataFrame({"Logistic": matrix_logistic.ravel(),
                       "SVM": matrix_svm.ravel(),
                       "Decision trees": matrix_tree.ravel(),
                       "Forest": matrix_forest.ravel(),
                       "Grid Forest": matrix_grid_forest.ravel(),
                       "Naive Bayes": matrix_bayes.ravel(),
                       "KNN": matrix_neigh.ravel()},
                       index=['TN', "FN", "FP", "TP"])
confusion_matrix_df.plot(kind='barh', figsize=(12, 8))
plt.title("Confusion matrix")
plt.show()

scores = pd.DataFrame({"Logistic": [score_logistic, recall_score_logistic, precision_score_logistic],
                       "SVM": [score_svm, recall_score_svm, precision_score_svm],
                       "Decision trees": [score_tree, recall_score_tree, precision_score_tree],
                       "Forest": [score_forest, recall_score_forest, precision_score_forest],
                       "Grid Forest": [score_grid_forest, recall_score_grid_forest, precision_score_grid_forest],
                       "Naive Bayes": [score_bayes, recall_score_bayes, precision_score_bayes],
                       "KNN": [score_neigh, recall_score_neigh, precision_score_neigh],
                       "NN": [score_nn, 0, 0]}, index=['Accuracy', "Recall: TP/TP+FN", "Precision: TP/TP+FP"])
scores.plot(kind='barh', figsize=(12, 8))
plt.title("Metrics")
plt.show()
