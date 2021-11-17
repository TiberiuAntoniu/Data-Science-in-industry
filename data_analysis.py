import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap, distplot

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# data columns and data type
print("Data columns/data type")
print(pd.DataFrame({"Train_Data": train_data.dtypes, "Test_data": test_data.dtypes}), end="\n\n")


# Data shape
print("Data shape")
print(f"Train_Data shape: {train_data.shape}, Test_data shape: {test_data.shape}", end="\n\n")

# Check if/how many null values are in the data
print("Null values in data")
print(pd.DataFrame({"Train_Data": train_data.isnull().sum(), "Test_data": test_data.isnull().sum()}), end="\n\n")

lst_columns = ["Loan_Status", "Gender", "Education", "Property_Area", "Married", "Credit_History"]
for column in lst_columns:
    # no of positive/negative cases present in the data
    print(f"No of {column} cases present in the data")
    print(train_data[column].value_counts(), end="\n\n")


lst_bins = [0, 2000, 4000, 6000, 85000]
lst_group = ['Low', 'Average', 'High', 'Very High']
dict_bin = {"ApplicantIncome": ([0, 2000, 4000, 6000, 85000], ['Low', 'Average', 'High', 'Very High']),
            "LoanAmount": ([0, 100, 200, 400, 700], ['Low', 'Average', 'High', 'Very High']),
            "Loan_Amount_Term": ([0, 200, 300, 400, 600], ['Low', 'Average', 'High', 'Very High']),
            }

for key in dict_bin:
    train_data[key + "_bin"] = pd.cut(train_data[key], dict_bin[key][0], labels=dict_bin[key][1])

for tuple_data in [("Gender", "Loan_Status"), ("Education", "Loan_Status"), ("Credit_History", "Loan_Status"),
                   ("Property_Area", "Loan_Status"), ("Married", "Loan_Status"), ("ApplicantIncome_bin", "Loan_Status"),
                   ("LoanAmount_bin", "Loan_Status"), ("Loan_Amount_Term_bin", "Loan_Status") ]:
    Category = pd.crosstab(train_data[tuple_data[0]], train_data[tuple_data[1]])
    Category.div(Category.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(14, 8))
    plt.xlabel(tuple_data[0])
    plt.ylabel("Percentage")
    plt.title(f"{tuple_data[0]} vs {tuple_data[1]}")
    plt.show()


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

# matrix = train_data.corr()
# f, ax = plt.subplots(figsize=(5, 5))
# heatmap(matrix, vmax=.8,  cmap="BuPu", annot=True)
# plt.show()

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


train_data['Dependents'] = train_data['Dependents'].astype(int)
test_data['Dependents'] = test_data['Dependents'].astype(int)

matrix = train_data.corr()
f, ax = plt.subplots(figsize=(12, 8))
heatmap(matrix, vmax=.8,  cmap="BuPu", annot=True)
plt.show()

for column in ["CoapplicantIncome", "ApplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
    plt.figure(1)
    plt.subplot(121)
    distplot(train_data[column])

    plt.subplot(122)
    train_data[column].plot.box(figsize=(16, 8))
    plt.show()

