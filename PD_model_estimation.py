import pandas as pd
import numpy as np
import os

os.chdir('C:/Users/massw/OneDrive/Bureau/Programmation/Python_R/Credit_Risk')
loan_data_inputs_train = pd.read_csv('loan_data_inputs_train.csv', index_col = 0)
loan_data_targets_train = pd.read_csv('loan_data_targets_train.csv', index_col = 0)
loan_data_inputs_test = pd.read_csv('loan_data_inputs_test.csv', index_col = 0)
loan_data_targets_test = pd.read_csv('loan_data_targets_test.csv', index_col = 0, header = None)

# Here we select a limited set of input variables in a new dataframe.
inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , ['grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E',
                                                            'grade:F', 'grade:G', 'home_ownership:RENT_OTHER_NONE_ANY',
                                                            'home_ownership:OWN', 'home_ownership:MORTGAGE', 'addr_state:ND_NE_IA_NV_FL_HI_AL',
                                                            'addr_state:NM_VA', 'addr_state:NY', 'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA',
                                                            'addr_state:UT_KY_AZ_NJ', 'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN',
                                                            'addr_state:GA_WA_OR', 'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT',
                                                            'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID', 
                                                            'verification_status:Not Verified', 'verification_status:Source Verified',
                                                            'verification_status:Verified', 'purpose:educ__sm_b__wedd__ren_en__mov__house',
                                                            'purpose:credit_card','purpose:debt_consolidation', 'purpose:oth__med__vacation',
                                                            'purpose:major_purch__car__home_impr', 'initial_list_status:f',
                                                            'initial_list_status:w', 'term:36', 'term:60', 'emp_length:0', 'emp_length:1',
                                                            'emp_length:2-4', 'emp_length:5-6', 'emp_length:7-9', 'emp_length:10',
                                                            'months_issue_date:<38','months_issue_date:38-39','months_issue_date:40-41',
                                                            'months_issue_date:42-48','months_issue_date:49-52','months_issue_date:53-64',
                                                            'months_issue_date:65-84','months_issue_date:>84','int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'months_earliest_cr_line:<140',
'months_earliest_cr_line:140-164',
'months_earliest_cr_line:165-247',
'months_earliest_cr_line:248-270',
'months_earliest_cr_line:271-352',
'months_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86',]]

# Here we store the names of the reference category dummy variables in a list.
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'months_issue_date:>84',
'int_rate:>20.281',
'months_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'open_acc:0',
'pub_rec:0-2',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
inputs_train.head()

# PD MODEL ESTIMATION
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# We create an instance of an object from the 'LogisticRegression' class.
reg = LogisticRegression()

# Sets the pandas dataframe options to display all columns/ rows.
pd.options.display.max_rows = None

# Estimates the coefficients of the object from the 'LogisticRegression' class
# with inputs (independent variables) contained in the first dataframe
# and targets (dependent variables) contained in the second dataframe.
reg.fit(inputs_train, loan_data_targets_train)

# Displays the intercept contain in the estimated ("fitted") object from the 'LogisticRegression' class.
reg.intercept_

# Displays the coefficients contained in the estimated ("fitted") object from the 'LogisticRegression' class.
reg.coef_

# Stores the names of the columns of a dataframe in a variable.
feature_name = inputs_train.columns.values

# Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)

# Creates a new column in the dataframe, called 'Coefficients',
# with row values the transposed coefficients from the 'LogisticRegression' object.
summary_table['Coefficients'] = np.transpose(reg.coef_)

# Increases the index of every row of the dataframe with 1.
summary_table.index = summary_table.index + 1
# Assigns values of the row with index 0 of the dataframe.
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# Sorts the dataframe by index.
summary_table = summary_table.sort_index()

summary_table