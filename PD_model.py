import numpy as np
import pandas as pd
import re
import os

os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation/Python_R/Credit_Risk")
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
df_inputs_prep = loan_data_inputs_test
df_targets_prepr = loan_data_targets_test


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

# Preprocessing discrete variables : Automation Calculations

# WoE function for discrete unordered variables
# Here we combine all of the operations above in a function.
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.
def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# We execute the function we defined with the necessary arguments: a dataframe, a string, and a dataframe.
# We store the result in a dataframe.
df_temp = woe_discrete(df_inputs_prep, 'grade', df_targets_prepr)
df_temp

# Preprocessing discrete variables : visualizing result

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Turns the values of the column with index 0 to strings, makes an array from these strings, and passes it to variable x.
    y = df_WoE['WoE']
    # Selects a column with label 'WoE' and passes it to variable y.
    plt.figure(figsize=(18, 6))
    # Sets the graph size to width 18 x height 6.
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    # Plots the datapoints with coordiantes variable x on the x-axis and variable y on the y-axis.
    # Sets the marker for each datapoint to a circle, the style line between the points to dashed, and the color to black.
    plt.xlabel(df_WoE.columns[0])
    # Names the x-axis with the name of the column with index 0.
    plt.ylabel('Weight of Evidence')
    # Names the y-axis 'Weight of Evidence'.
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    # Names the grapth 'Weight of Evidence by ' the name of the column with index 0.
    plt.xticks(rotation = rotation_of_x_axis_labels)
    # Rotates the labels of the x-axis a predefined number of degrees.
    plt.show()

# We execute the function we defined with the necessary arguments: a dataframe.
# We omit the number argument, which means the function will use its default value, 0.
# plot_by_woe(df_temp)

# Preprocessing discrete variables : Creating Dummy Variable, Part 1 

# 'home_ownership'
df_temp = woe_discrete(df_inputs_prep, 'home_ownership', df_targets_prepr)


# We plot the weight of evidence values.
# plot_by_woe(df_temp)

# There are many categories with very few observations and many categories with very different "good" %.
# Therefore, we create a new discrete variable where we combine some of the categories.
# 'OTHERS' and 'NONE' are riskiest but are very few. 'RENT' is the next riskiest.
# 'ANY' are least risky but are too few. Conceptually, they belong to the same category. Also, their inclusion would not change anything.
# We combine them in one category, 'RENT_OTHER_NONE_ANY'.
# We end up with 3 categories: 'RENT_OTHER_NONE_ANY', 'OWN', 'MORTGAGE'.
df_inputs_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prep['home_ownership:RENT'], df_inputs_prep['home_ownership:OTHER'],
                                                      df_inputs_prep['home_ownership:NONE'],df_inputs_prep['home_ownership:ANY']])
# 'RENT_OTHER_NONE_ANY' will be the reference category.

# Alternatively:
#loan_data.loc['home_ownership' in ['RENT', 'OTHER', 'NONE', 'ANY'], 'home_ownership:RENT_OTHER_NONE_ANY'] = 1
#loan_data.loc['home_ownership' not in ['RENT', 'OTHER', 'NONE', 'ANY'], 'home_ownership:RENT_OTHER_NONE_ANY'] = 0
#loan_data.loc['loan_status' not in ['OWN'], 'home_ownership:OWN'] = 1
#loan_data.loc['loan_status' not in ['OWN'], 'home_ownership:OWN'] = 0
#loan_data.loc['loan_status' not in ['MORTGAGE'], 'home_ownership:MORTGAGE'] = 1
#loan_data.loc['loan_status' not in ['MORTGAGE'], 'home_ownership:MORTGAGE'] = 0


# Preprocessing discrete variables : Creating Dummy Variable, Part 2
df_temp = woe_discrete(df_inputs_prep, 'addr_state', df_targets_prepr)
df_temp

# plot_by_woe(df_temp)

if ['addr_state:ND'] in df_inputs_prep.columns.values:
    pass
else:
    df_inputs_prep['addr_state:ND'] = 0

# plot_by_woe(df_temp.iloc[2: -2, : ])
# We plot the weight of evidence values.

# plot_by_woe(df_temp.iloc[6: -6, : ])
# We plot the weight of evidence values.

# We create the following categories:
# 'ND' 'NE' 'IA' NV' 'FL' 'HI' 'AL'
# 'NM' 'VA'
# 'NY'
# 'OK' 'TN' 'MO' 'LA' 'MD' 'NC'
# 'CA'
# 'UT' 'KY' 'AZ' 'NJ'
# 'AR' 'MI' 'PA' 'OH' 'MN'
# 'RI' 'MA' 'DE' 'SD' 'IN'
# 'GA' 'WA' 'OR'
# 'WI' 'MT'
# 'TX'
# 'IL' 'CT'
# 'KS' 'SC' 'CO' 'VT' 'AK' 'MS'
# 'WV' 'NH' 'WY' 'DC' 'ME' 'ID'

# 'IA_NV_HI_ID_AL_FL' will be the reference category.

df_inputs_prep['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prep['addr_state:ND'], df_inputs_prep['addr_state:NE'],
                                              df_inputs_prep['addr_state:IA'], df_inputs_prep['addr_state:NV'],
                                              df_inputs_prep['addr_state:FL'], df_inputs_prep['addr_state:HI'],
                                                          df_inputs_prep['addr_state:AL']])

df_inputs_prep['addr_state:NM_VA'] = sum([df_inputs_prep['addr_state:NM'], df_inputs_prep['addr_state:VA']])

df_inputs_prep['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prep['addr_state:OK'], df_inputs_prep['addr_state:TN'],
                                              df_inputs_prep['addr_state:MO'], df_inputs_prep['addr_state:LA'],
                                              df_inputs_prep['addr_state:MD'], df_inputs_prep['addr_state:NC']])

df_inputs_prep['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prep['addr_state:UT'], df_inputs_prep['addr_state:KY'],
                                              df_inputs_prep['addr_state:AZ'], df_inputs_prep['addr_state:NJ']])

df_inputs_prep['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prep['addr_state:AR'], df_inputs_prep['addr_state:MI'],
                                              df_inputs_prep['addr_state:PA'], df_inputs_prep['addr_state:OH'],
                                              df_inputs_prep['addr_state:MN']])

df_inputs_prep['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prep['addr_state:RI'], df_inputs_prep['addr_state:MA'],
                                              df_inputs_prep['addr_state:DE'], df_inputs_prep['addr_state:SD'],
                                              df_inputs_prep['addr_state:IN']])

df_inputs_prep['addr_state:GA_WA_OR'] = sum([df_inputs_prep['addr_state:GA'], df_inputs_prep['addr_state:WA'],
                                              df_inputs_prep['addr_state:OR']])

df_inputs_prep['addr_state:WI_MT'] = sum([df_inputs_prep['addr_state:WI'], df_inputs_prep['addr_state:MT']])

df_inputs_prep['addr_state:IL_CT'] = sum([df_inputs_prep['addr_state:IL'], df_inputs_prep['addr_state:CT']])

df_inputs_prep['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prep['addr_state:KS'], df_inputs_prep['addr_state:SC'],
                                              df_inputs_prep['addr_state:CO'], df_inputs_prep['addr_state:VT'],
                                              df_inputs_prep['addr_state:AK'], df_inputs_prep['addr_state:MS']])

df_inputs_prep['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prep['addr_state:WV'], df_inputs_prep['addr_state:NH'],
                                              df_inputs_prep['addr_state:WY'], df_inputs_prep['addr_state:DC'],
                                              df_inputs_prep['addr_state:ME'], df_inputs_prep['addr_state:ID']])


# Preprocessing discrete variables : Creating Dummy Variable, Part 3 : VERIFICATION STATUS
df_temp = woe_discrete(df_inputs_prep, 'verification_status', df_targets_prepr)
# plot_by_woe(df_temp)

# Preprocessing discrete variables : Creating Dummy Variable, Part 3 : PURPOSE
df_temp = woe_discrete(df_inputs_prep, 'purpose', df_targets_prepr)
# plot_by_woe(df_temp,90)

df_inputs_prep['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prep['purpose:small_business'], 
                                                                 df_inputs_prep['purpose:educational'],
                                                                 df_inputs_prep['purpose:moving'],
                                                                 df_inputs_prep['purpose:house'],
                                                                 df_inputs_prep['purpose:renewable_energy'],
                                                                 df_inputs_prep['purpose:wedding']])

df_inputs_prep['purpose:oth__med__vacation'] = sum([df_inputs_prep['purpose:other'],
                                                    df_inputs_prep['purpose:medical'],
                                                    df_inputs_prep['purpose:vacation']])

df_inputs_prep['purpose:major_purch__car__home_impr'] = sum([df_inputs_prep['purpose:major_purchase'], df_inputs_prep['purpose:car'],
                                                        df_inputs_prep['purpose:home_improvement']])


# Preprocessing discrete variables : Creating Dummy Variable, Part 3 : INITIAL LIST STATUS
df_temp = woe_discrete(df_inputs_prep, 'initial_list_status', df_targets_prepr)
# plot_by_woe(df_temp)


# Preprocessing Continuous Variables: Automating Calculations and Visualizing Results
# WoE function for ordered discrete and continuous variables
def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
# Here we define a function similar to the one above, ...
# ... with one slight difference: we order the results by the values of a different column.
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.

df_inputs_prep['term_months'].unique()

df_temp = woe_ordered_continuous(df_inputs_prep, 'term_months', df_targets_prepr)
df_temp

# plot_by_woe(df_temp)

df_inputs_prep['term:36'] = np.where((df_inputs_prep['term_months'] == 36), 1, 0)
df_inputs_prep['term:60'] = np.where((df_inputs_prep['term_months'] == 60), 1, 0)

df_inputs_prep['emp_length_int'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'emp_length_int', df_targets_prepr)
# plot_by_woe(df_temp)

df_inputs_prep['emp_length:0'] = np.where(df_inputs_prep['emp_length_int'].isin([0]), 1, 0)
df_inputs_prep['emp_length:1'] = np.where(df_inputs_prep['emp_length_int'].isin([1]), 1, 0)
df_inputs_prep['emp_length:2-4'] = np.where(df_inputs_prep['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prep['emp_length:5-6'] = np.where(df_inputs_prep['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prep['emp_length:7-9'] = np.where(df_inputs_prep['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prep['emp_length:10'] = np.where(df_inputs_prep['emp_length_int'].isin([10]), 1, 0)

# Preprocessing Continuous Variables: Automating Calculations and Visualizing Results, Part 2
df_inputs_prep['months_issue_date'].unique()

df_inputs_prep['months_issue_date_factors'] = pd.cut(df_inputs_prep['months_issue_date'], 50)
df_temp = woe_ordered_continuous(df_inputs_prep, 'months_issue_date_factors', df_targets_prepr)
df_temp
# plot_by_woe(df_temp, 90)

df_inputs_prep['months_issue_date:<38'] = np.where(df_inputs_prep['months_issue_date'].isin(range(38)), 1, 0)
df_inputs_prep['months_issue_date:38-39'] = np.where(df_inputs_prep['months_issue_date'].isin(range(38, 40)), 1, 0)
df_inputs_prep['months_issue_date:40-41'] = np.where(df_inputs_prep['months_issue_date'].isin(range(40, 42)), 1, 0)
df_inputs_prep['months_issue_date:42-48'] = np.where(df_inputs_prep['months_issue_date'].isin(range(42, 49)), 1, 0)
df_inputs_prep['months_issue_date:49-52'] = np.where(df_inputs_prep['months_issue_date'].isin(range(49, 53)), 1, 0)
df_inputs_prep['months_issue_date:53-64'] = np.where(df_inputs_prep['months_issue_date'].isin(range(53, 65)), 1, 0)
df_inputs_prep['months_issue_date:65-84'] = np.where(df_inputs_prep['months_issue_date'].isin(range(65, 85)), 1, 0)
df_inputs_prep['months_issue_date:>84'] = np.where(df_inputs_prep['months_issue_date'].isin(range(85, int(df_inputs_prep['months_issue_date'].max()))), 1, 0)


df_inputs_prep['int_rate_factor'] = pd.cut(df_inputs_prep['int_rate'],50)
df_temp = woe_ordered_continuous(df_inputs_prep, 'int_rate_factor', df_targets_prepr)
# plot_by_woe(df_temp, 90)

df_inputs_prep['int_rate:<9.548'] = np.where((df_inputs_prep['int_rate'] <= 9.548), 1, 0)
df_inputs_prep['int_rate:9.548-12.025'] = np.where((df_inputs_prep['int_rate'] > 9.548) & (df_inputs_prep['int_rate'] <= 12.025), 1, 0)
df_inputs_prep['int_rate:12.025-15.74'] = np.where((df_inputs_prep['int_rate'] > 12.025) & (df_inputs_prep['int_rate'] <= 15.74), 1, 0)
df_inputs_prep['int_rate:15.74-20.281'] = np.where((df_inputs_prep['int_rate'] > 15.74) & (df_inputs_prep['int_rate'] <= 20.281), 1, 0)
df_inputs_prep['int_rate:>20.281'] = np.where((df_inputs_prep['int_rate'] > 20.281), 1, 0)


df_inputs_prep['funded_amnt_factor'] = pd.cut(df_inputs_prep['funded_amnt'],50)
df_temp = woe_ordered_continuous(df_inputs_prep, 'funded_amnt_factor', df_targets_prepr)
# plot_by_woe(df_temp,90)

df_inputs_prep['months_earliest_cr_line'].unique()
df_inputs_prep['months_earliest_cr_linefactor'] = pd.cut(df_inputs_prep['months_earliest_cr_line'],50)
df_temp = woe_ordered_continuous(df_inputs_prep, 'months_earliest_cr_linefactor', df_targets_prepr)
# plot_by_woe(df_temp, 90)

df_inputs_prep['months_earliest_cr_line:<140'] = np.where(df_inputs_prep['months_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prep['months_earliest_cr_line:140-164'] = np.where(df_inputs_prep['months_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prep['months_earliest_cr_line:165-247'] = np.where(df_inputs_prep['months_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prep['months_earliest_cr_line:248-270'] = np.where(df_inputs_prep['months_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prep['months_earliest_cr_line:271-352'] = np.where(df_inputs_prep['months_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prep['months_earliest_cr_line:>352'] = np.where(df_inputs_prep['months_earliest_cr_line'].isin(range(353, int(df_inputs_prep['months_earliest_cr_line'].max()))), 1, 0)

df_inputs_prep['installment_factor'] = pd.cut(df_inputs_prep['installment'], 50)
df_temp = woe_ordered_continuous(df_inputs_prep, 'installment_factor', df_targets_prepr)
# plot_by_woe(df_temp)

df_inputs_prep['delinq_2yrs'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'delinq_2yrs', df_targets_prepr)
# plot_by_woe(df_temp)

df_inputs_prep['delinq_2yrs:0'] = np.where((df_inputs_prep['delinq_2yrs'] == 0),1,0)
df_inputs_prep['delinq_2yrs:1-3'] = np.where((df_inputs_prep['delinq_2yrs'] >= 1) & (df_inputs_prep['delinq_2yrs'] <= 3),1,0)
df_inputs_prep['delinq_2yrs:>=4'] = np.where((df_inputs_prep['delinq_2yrs'] >= 9),1,0)


df_inputs_prep['inq_last_6mths'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'inq_last_6mths', df_targets_prepr)
# plot_by_woe(df_temp,90)
df_inputs_prep['inq_last_6mths:0'] = np.where((df_inputs_prep['inq_last_6mths'] == 0), 1, 0)
df_inputs_prep['inq_last_6mths:1-2'] = np.where((df_inputs_prep['inq_last_6mths'] >= 1) & (df_inputs_prep['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prep['inq_last_6mths:3-6'] = np.where((df_inputs_prep['inq_last_6mths'] >= 3) & (df_inputs_prep['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prep['inq_last_6mths:>6'] = np.where((df_inputs_prep['inq_last_6mths'] > 6), 1, 0)


df_inputs_prep['open_acc'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'open_acc', df_targets_prepr)
# plot_by_woe(df_temp,90)
df_inputs_prep['open_acc:0'] = np.where((df_inputs_prep['open_acc'] == 0), 1, 0)
df_inputs_prep['open_acc:1-3'] = np.where((df_inputs_prep['open_acc'] >= 1) & (df_inputs_prep['open_acc'] <= 3), 1, 0)
df_inputs_prep['open_acc:4-12'] = np.where((df_inputs_prep['open_acc'] >= 4) & (df_inputs_prep['open_acc'] <= 12), 1, 0)
df_inputs_prep['open_acc:13-17'] = np.where((df_inputs_prep['open_acc'] >= 13) & (df_inputs_prep['open_acc'] <= 17), 1, 0)
df_inputs_prep['open_acc:18-22'] = np.where((df_inputs_prep['open_acc'] >= 18) & (df_inputs_prep['open_acc'] <= 22), 1, 0)
df_inputs_prep['open_acc:23-25'] = np.where((df_inputs_prep['open_acc'] >= 23) & (df_inputs_prep['open_acc'] <= 25), 1, 0)
df_inputs_prep['open_acc:26-30'] = np.where((df_inputs_prep['open_acc'] >= 26) & (df_inputs_prep['open_acc'] <= 30), 1, 0)
df_inputs_prep['open_acc:>=31'] = np.where((df_inputs_prep['open_acc'] >= 31), 1, 0)

df_inputs_prep['pub_rec'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'pub_rec', df_targets_prepr)
# plot_by_woe(df_temp,90)
df_inputs_prep['pub_rec:0-2'] = np.where((df_inputs_prep['pub_rec'] >= 0) & (df_inputs_prep['pub_rec'] <= 2), 1, 0)
df_inputs_prep['pub_rec:3-4'] = np.where((df_inputs_prep['pub_rec'] >= 3) & (df_inputs_prep['pub_rec'] <= 4), 1, 0)
df_inputs_prep['pub_rec:>=5'] = np.where((df_inputs_prep['pub_rec'] >= 5), 1, 0)

df_inputs_prep['total_acc'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'total_acc', df_targets_prepr)
# plot_by_woe(df_temp,90)
df_inputs_prep['total_acc:<=27'] = np.where((df_inputs_prep['total_acc'] <= 27), 1, 0)
df_inputs_prep['total_acc:28-51'] = np.where((df_inputs_prep['total_acc'] >= 28) & (df_inputs_prep['total_acc'] <= 51), 1, 0)
df_inputs_prep['total_acc:>=52'] = np.where((df_inputs_prep['total_acc'] >= 52), 1, 0)

df_inputs_prep['acc_now_delinq'].unique()
df_temp = woe_ordered_continuous(df_inputs_prep, 'acc_now_delinq', df_targets_prepr)
# plot_by_woe(df_temp,90)
df_inputs_prep['acc_now_delinq:0'] = np.where((df_inputs_prep['acc_now_delinq'] == 0), 1, 0)
df_inputs_prep['acc_now_delinq:>=1'] = np.where((df_inputs_prep['acc_now_delinq'] >= 1), 1, 0)

# total_rev_hi_lim
df_inputs_prep['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prep['total_rev_hi_lim'], 2000)
# Here we do fine-classing: using the 'cut' method, we split the variable into 2000 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prep, 'total_rev_hi_lim_factor', df_targets_prepr)
# We calculate weight of evidence.
df_temp

# plot_by_woe(df_temp.iloc[: 50, : ], 90)
df_inputs_prep['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prep['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prep['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 5000) & (df_inputs_prep['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prep['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 10000) & (df_inputs_prep['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prep['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 20000) & (df_inputs_prep['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prep['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 30000) & (df_inputs_prep['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prep['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 40000) & (df_inputs_prep['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prep['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 55000) & (df_inputs_prep['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prep['total_rev_hi_lim:>95K'] = np.where((df_inputs_prep['total_rev_hi_lim'] > 95000), 1, 0)

df_inputs_prep['annual_inc_factor'] = pd.cut(df_inputs_prep['annual_inc'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prep, 'annual_inc_factor', df_targets_prepr)
# We calculate weight of evidence.
df_temp

# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prep_temp = df_inputs_prep.loc[df_inputs_prep['annual_inc'] <= 140000, : ]

df_inputs_prep_temp["annual_inc_factor"] = pd.cut(df_inputs_prep_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prep_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prep_temp.index])
# We calculate weight of evidence.
df_temp
# plot_by_woe(df_temp,90)
# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prep['annual_inc:<20K'] = np.where((df_inputs_prep['annual_inc'] <= 20000), 1, 0)
df_inputs_prep['annual_inc:20K-30K'] = np.where((df_inputs_prep['annual_inc'] > 20000) & (df_inputs_prep['annual_inc'] <= 30000), 1, 0)
df_inputs_prep['annual_inc:30K-40K'] = np.where((df_inputs_prep['annual_inc'] > 30000) & (df_inputs_prep['annual_inc'] <= 40000), 1, 0)
df_inputs_prep['annual_inc:40K-50K'] = np.where((df_inputs_prep['annual_inc'] > 40000) & (df_inputs_prep['annual_inc'] <= 50000), 1, 0)
df_inputs_prep['annual_inc:50K-60K'] = np.where((df_inputs_prep['annual_inc'] > 50000) & (df_inputs_prep['annual_inc'] <= 60000), 1, 0)
df_inputs_prep['annual_inc:60K-70K'] = np.where((df_inputs_prep['annual_inc'] > 60000) & (df_inputs_prep['annual_inc'] <= 70000), 1, 0)
df_inputs_prep['annual_inc:70K-80K'] = np.where((df_inputs_prep['annual_inc'] > 70000) & (df_inputs_prep['annual_inc'] <= 80000), 1, 0)
df_inputs_prep['annual_inc:80K-90K'] = np.where((df_inputs_prep['annual_inc'] > 80000) & (df_inputs_prep['annual_inc'] <= 90000), 1, 0)
df_inputs_prep['annual_inc:90K-100K'] = np.where((df_inputs_prep['annual_inc'] > 90000) & (df_inputs_prep['annual_inc'] <= 100000), 1, 0)
df_inputs_prep['annual_inc:100K-120K'] = np.where((df_inputs_prep['annual_inc'] > 100000) & (df_inputs_prep['annual_inc'] <= 120000), 1, 0)
df_inputs_prep['annual_inc:120K-140K'] = np.where((df_inputs_prep['annual_inc'] > 120000) & (df_inputs_prep['annual_inc'] <= 140000), 1, 0)
df_inputs_prep['annual_inc:>140K'] = np.where((df_inputs_prep['annual_inc'] > 140000), 1, 0)

# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prep_temp = df_inputs_prep[pd.notnull(df_inputs_prep['mths_since_last_delinq'])]
df_inputs_prep_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prep_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuous(df_inputs_prep_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prep_temp.index])
# We calculate weight of evidence.
df_temp

# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prep['mths_since_last_delinq:Missing'] = np.where((df_inputs_prep['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prep['mths_since_last_delinq:0-3'] = np.where((df_inputs_prep['mths_since_last_delinq'] >= 0) & (df_inputs_prep['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prep['mths_since_last_delinq:4-30'] = np.where((df_inputs_prep['mths_since_last_delinq'] >= 4) & (df_inputs_prep['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prep['mths_since_last_delinq:31-56'] = np.where((df_inputs_prep['mths_since_last_delinq'] >= 31) & (df_inputs_prep['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prep['mths_since_last_delinq:>=57'] = np.where((df_inputs_prep['mths_since_last_delinq'] >= 57), 1, 0)

#Categories : Debt to income ratio
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_inputs_prep['dti_factor'] = pd.cut(df_inputs_prep['dti'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prep['dti_factor'] = pd.cut(df_inputs_prep['dti'], 50)
# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prep_temp = df_inputs_prep.loc[df_inputs_prep['dti'] <= 35, : ]
# We calculate weight of evidence.
df_temp = woe_ordered_continuous(df_inputs_prep_temp, 'dti_factor', df_targets_prepr[df_inputs_prep_temp.index])
# plot_by_woe(df_temp,90)
# Categories:
df_inputs_prep['dti:<=1.4'] = np.where((df_inputs_prep['dti'] <= 1.4), 1, 0)
df_inputs_prep['dti:1.4-3.5'] = np.where((df_inputs_prep['dti'] > 1.4) & (df_inputs_prep['dti'] <= 3.5), 1, 0)
df_inputs_prep['dti:3.5-7.7'] = np.where((df_inputs_prep['dti'] > 3.5) & (df_inputs_prep['dti'] <= 7.7), 1, 0)
df_inputs_prep['dti:7.7-10.5'] = np.where((df_inputs_prep['dti'] > 7.7) & (df_inputs_prep['dti'] <= 10.5), 1, 0)
df_inputs_prep['dti:10.5-16.1'] = np.where((df_inputs_prep['dti'] > 10.5) & (df_inputs_prep['dti'] <= 16.1), 1, 0)
df_inputs_prep['dti:16.1-20.3'] = np.where((df_inputs_prep['dti'] > 16.1) & (df_inputs_prep['dti'] <= 20.3), 1, 0)
df_inputs_prep['dti:20.3-21.7'] = np.where((df_inputs_prep['dti'] > 20.3) & (df_inputs_prep['dti'] <= 21.7), 1, 0)
df_inputs_prep['dti:21.7-22.4'] = np.where((df_inputs_prep['dti'] > 21.7) & (df_inputs_prep['dti'] <= 22.4), 1, 0)
df_inputs_prep['dti:22.4-35'] = np.where((df_inputs_prep['dti'] > 22.4) & (df_inputs_prep['dti'] <= 35), 1, 0)
df_inputs_prep['dti:>35'] = np.where((df_inputs_prep['dti'] > 35), 1, 0)

# mths_since_last_record
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prep_temp = df_inputs_prep[pd.notnull(df_inputs_prep['mths_since_last_record'])]
#sum(loan_data_temp['mths_since_last_record'].isnull())
df_inputs_prep_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prep_temp['mths_since_last_record'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prep_temp, 'mths_since_last_record_factor', df_targets_prepr[df_inputs_prep_temp.index])
# We calculate weight of evidence.
df_temp
# plot_by_woe(df_temp,90)

# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prep['mths_since_last_record:Missing'] = np.where((df_inputs_prep['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prep['mths_since_last_record:0-2'] = np.where((df_inputs_prep['mths_since_last_record'] >= 0) & (df_inputs_prep['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prep['mths_since_last_record:3-20'] = np.where((df_inputs_prep['mths_since_last_record'] >= 3) & (df_inputs_prep['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prep['mths_since_last_record:21-31'] = np.where((df_inputs_prep['mths_since_last_record'] >= 21) & (df_inputs_prep['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prep['mths_since_last_record:32-80'] = np.where((df_inputs_prep['mths_since_last_record'] >= 32) & (df_inputs_prep['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prep['mths_since_last_record:81-86'] = np.where((df_inputs_prep['mths_since_last_record'] >= 81) & (df_inputs_prep['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prep['mths_since_last_record:>86'] = np.where((df_inputs_prep['mths_since_last_record'] > 86), 1, 0)

loan_data_inputs_train = df_inputs_prep
# loan_data_inputs_test = df_inputs_prep

#loan_data_inputs_train.to_csv('loan_data_inputs_train.csv')
# loan_data_targets_train.to_csv('loan_data_targets_train.csv')
loan_data_inputs_test.to_csv('loan_data_inputs_test.csv')
loan_data_targets_test.to_csv('loan_data_targets_test.csv')