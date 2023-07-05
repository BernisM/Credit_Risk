import numpy as np
import pandas as pd
import re

loan_data_backup = pd.read_csv('loan_data_2007_2014.csv')

loan_data = loan_data_backup.copy()

loan_data

loan_data['emp_length'].unique()

# Preprocessing few continious variables

# convert str to number
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year','')
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

# convert date string to datetime and calculate the number of days credit line issued
loan_data['earliest_cr_line']
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format= '%b-%y')
pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']

# new column to do the same in months
loan_data['months_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))

loan_data['months_earliest_cr_line'].describe()

loan_data.loc[:, ['earliest_cr_line','earliest_cr_line_date','months_earliest_cr_line']][loan_data['months_earliest_cr_line'] < 0]

loan_data['months_earliest_cr_line'][loan_data['months_earliest_cr_line'] < 0] = loan_data['months_earliest_cr_line'].max()

min(loan_data['months_earliest_cr_line'])

# Preprocessing for term variable
loan_data['term']
loan_data['term_months'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

loan_data['term_days'] = loan_data['term_months'] * 30

# Preprocessing for issue Date
loan_data['issue_d']
loan_data['issue_date'] = pd.to_datetime(loan_data['issue_d'], format= '%b-%y')
loan_data['months_issue_date'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['issue_date']) / np.timedelta64(1, 'M')))
loan_data['months_issue_date']

# Preprocessing few discrete variables
loan_data.info()

# Create dummy variables from a variable.
pd.get_dummies(loan_data['grade'])

pd.get_dummies(loan_data['grade'], prefix='grade', prefix_sep=': ')

# We create dummy variables from all 8 original independent variables, and save them into a list.
# Note that we are using a particular naming convention for all variables: original variable name, colon, category name.
loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')]

# We concatenate the dummy variables and this turns them into a dataframe.
loan_data_dummies = pd.concat(loan_data_dummies,axis=1)

type(loan_data_dummies)

# Concatenates two dataframes.
# Here we concatenate the dataframe with original data with the dataframe with dummy variables, along the columns. 
loan_data = pd.concat([loan_data,loan_data_dummies],axis=1)

loan_data.columns.values

# Checking for missing values
loan_data.isnull()

pd.options.display.max_rows = 100

# 'Total revolving high credit/ credit limit', so it makes sense that the missing values are equal to funded_amnt.
# We fill the missing values with the values of another variable.
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
# Check number if sum of missing value = 0
loan_data['total_rev_hi_lim'].isnull().sum()

# We fill the missing values with the mean value of the non-missing values.
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
loan_data['annual_inc'].isnull().sum()

loan_data['months_earliest_cr_line'].fillna(0,inplace=True)
loan_data['months_earliest_cr_line'].isnull().sum()

# We fill the missing values with zeroes.
loan_data['acc_now_delinq'].fillna(0,inplace=True)
loan_data['total_acc'].fillna(0,inplace=True)
loan_data['pub_rec'].fillna(0,inplace=True)
loan_data['open_acc'].fillna(0,inplace=True)
loan_data['inq_last_6mths'].fillna(0,inplace=True)
loan_data['delinq_2yrs'].fillna(0,inplace=True)
loan_data['emp_length_int'].fillna(0,inplace=True)

#### PD MODEL
# Data preparation 

loan_data['loan_status'].unique()

# Calculates the number of observations for each unique value of a variable.
loan_data['loan_status'].value_counts()

# We divide the number of observations for each unique value of a variable by the total number of observations.
# Thus, we get the proportion of observations for each unique value of a variable.
loan_data['loan_status'].value_counts() / loan_data['loan_status'].count()

# Good/ Bad Definition
# We create a new variable that has the value of '0' if a condition is met, and the value of '1' if it is not met.
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                       'Does not meet the credit policy. Status:Charged Off',
                                                       'Late (31-120 days)']), 0, 1)

loan_data['good_bad']

# Splitting Data
from sklearn.model_selection import train_test_split

# Takes a set of inputs and a set of targets as arguments. Splits the inputs and the targets into four dataframes:
# Inputs - Train, Inputs - Test, Targets - Train, Targets - Test.
train_test_split(loan_data.drop('good_bad',axis=1),loan_data['good_bad'])

# We split two dataframes with inputs and targets, each into a train and test dataframe, and store them in variables.
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad',axis=1),loan_data['good_bad'])

# 349'713 of observations x 207 variables
loan_data_inputs_train.shape
# 349'713 of observations
loan_data_targets_train.shape
# 116'572 of observations x 207 variables
loan_data_inputs_test.shape
# 116'572 of observations
loan_data_targets_test.shape

# # We split two dataframes with inputs and targets, each into a train and test dataframe, and store them in variables.
# This time we set the size of the test dataset to be 20%.
# Respectively, the size of the train dataset becomes 80%.
# We also set a specific random state.
# This would allow us to perform the exact same split multimple times.
# This means, to assign the exact same observations to the train and test datasets.
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad',axis=1),loan_data['good_bad'], test_size=0.2, random_state = 42)


df_inputs_prep = loan_data_inputs_train
df_targets_prepr = loan_data_targets_train

df_inputs_prep['grade'].unique()

df1 = pd.concat([df_inputs_prep['grade'],df_targets_prepr],axis=1)

# Groups the data according to a criterion contained in one column.
# Does not turn the names of the values of the criterion as indexes.
# Aggregates the data in another column, using a selected function.
# In this specific case, we group by the column with index 0 and we aggregate the values of the column with index 1.
# More specifically, we count them.
# In other words, we count the values in the column with index 1 for each value of the column with index 0.
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count()

# Groups the data according to a criterion contained in one column.
# Does not turn the names of the values of the criterion as indexes.
# Aggregates the data in another column, using a selected function.
# Here we calculate the mean of the values in the column with index 1 for each value of the column with index 0.
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()

# Concatenates two dataframes along the columns.
df1 = pd.concat([df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count(),
                df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()], axis = 1)

df1 = df1.iloc[:, [0, 1, 3]]

# Changes the names of the columns of a dataframe.
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']

# We divide the values of one column by he values of another column and save the result in a new variable.
df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()

df1['n_good'] = df1['prop_good'] * df1['n_obs']
# We multiply the values of one column by he values of another column and save the result in a new variable.
df1['n_bad'] = (1 - df1['prop_good']) * df1['n_obs']

df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()

# We take the natural logarithm of a variable and save the result in a nex variable.
df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'])

# Sorts a dataframe by the values of a given column.
df1 = df1.sort_values(['WoE'])
# We reset the index of a dataframe and overwrite it.
df1 = df1.reset_index(drop = True)

# We take the difference between two subsequent values of a column. Then, we take the absolute value of the result.
df1['diff_prop_good'] = df1['prop_good'].diff().abs()
# We take the difference between two subsequent values of a column. Then, we take the absolute value of the result.
df1['diff_WoE'] = df1['WoE'].diff().abs()

df1['IV'] = (df1['prop_n_good'] - df1['prop_n_bad']) * df1['WoE']
df1['IV'] = df1['IV'].sum()

df1




