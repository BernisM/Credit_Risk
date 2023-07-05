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

