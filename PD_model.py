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
loan_data['term'].unique()
loan_data['term_months'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

loan_data['term_days'] = loan_data['term_months'] * 30

# Preprocessing for issue Date
loan_data['issue_d'].unique()
loan_data['issue_date'] = pd.to_datetime(loan_data['issue_d'], format= '&b-%y')

loan_data['issue_d']

loan_data[]