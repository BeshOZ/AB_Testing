
############################################
# AB Testing
############################################

###################################################
# Imports, Functions and Settings.
###################################################

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 500)

def check_df(dataframe,head=5):
    print("##Shape##")
    print(dataframe.shape)
    print("##Types##")
    print(dataframe.dtypes)
    print("##Head##")
    print(dataframe.head(head))
    print("##Tail##")
    print(dataframe.tail(head))
    print("##Missingentries##")
    print(dataframe.isnull().sum())
    print("##Quantiles##")
    print(dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)
    print("##generalinformation##")
    print(dataframe.describe().T)


###################################################
# Importing data and taking a first look
###################################################

df_control = pd.read_excel("ab_testing_veri/ab_testing.xlsx" , sheet_name="Control Group")
df_test = pd.read_excel("ab_testing_veri/ab_testing.xlsx" , sheet_name="Test Group")

check_df(df_control)
check_df(df_test)
originalCols = df_control.columns
#Adding the test group to the control group
df_control["Group"] = "Control"
df_test["Group"] = "Test"

df = pd.concat([df_control,df_test],axis=0,ignore_index=False)
df.head()

## A/B Test

# H0 : M1 = M2 There is no significant difference between the purchases of the test and control groups.
# H1 : M1!= M2 There is a significant difference between the purchases of the test and control groups.

# Average purchases for both test and control groups

df.groupby("Group").mean()
#           Impression      Click  Purchase    Earning
# Group
# Control 101711.44907 5100.65737 550.89406 1908.56830
# Test    120512.41176 3967.54976 582.10610 2514.89073
# We can see that there is an increment, but it might happen by chance.

################################################################

## Before getting into A/B testings, we should check the assumptions first by
##Testing normality and homogeneity of variance for independent samples t-tests

## Normality Test:

# H0: the population is normally distributed
# H1: the population is not normally distributed
# p < 0.05 H0 Rejected
# p > 0.05 H0 Can't be rejected

## Check the normality in the Control group:

for col in originalCols:
    test_stat, pvalue = shapiro(df.loc[df["Group"] == "Control", col])
    print(col + ' Results: \n Test Stat = %.4f, p-value = %.4f\n' % (test_stat, pvalue))

# Control Group normality test:

# Impression Results:
#  Test Stat = 0.9697, p-value = 0.3514

# Click Results:
#  Test Stat = 0.9844, p-value = 0.8461

# Purchase Results:
#  Test Stat = 0.9773, p-value = 0.5891

# Earning Results:
#  Test Stat = 0.9756, p-value = 0.5306

# HO Can't be rejected for all the variables, which means it's normally distributed.


## Check the normality in the Test group:
for col in originalCols:
    test_stat, pvalue = shapiro(df.loc[df["Group"] == "Test", col])
    print(col + ' Results: \n Test Stat = %.4f, p-value = %.4f\n' % (test_stat, pvalue))

# Test Group normality test:

# Impression Results:
#  Test Stat = 0.9720, p-value = 0.4148
#
# Click Results:
#  Test Stat = 0.9896, p-value = 0.9699
#
# Purchase Results:
#  Test Stat = 0.9589, p-value = 0.1541
#
# Earning Results:
#  Test Stat = 0.9780, p-value = 0.6163

# Same as the control group, HO Can't be rejected for all the variables, which means it's normally distributed.

# Now that we found that both groups are normally distributed, it's time to check the variance homogeneity
# Alternative: If the normality assumption wasn't satisfied we can use another test: Mann-Whitney

###########################################################

## The Variance Homogeneity:

# H0: There is no significant statistical difference between the variances of purchases of the test and control groups.
# H1: There is a significant statistical difference between the variances of purchases of the test and control groups.
# p < 0.05 H0 Rejected
# p > 0.05 H0 Can't be rejected

for col in originalCols:
    test_stat, pvalue = levene(df.loc[df["Group"] == "Control", col],
                               df.loc[df["Group"] == "Test", col])
    print(col + ' results: \n Test Stat = %.4f, p-value = %.4f\n' % (test_stat, pvalue))

# Impression results:
# Test Stat = 0.5865, p - value = 0.4461

# Click results:
# Test Stat = 6.3041, p - value = 0.0141

# Purchase results:
# Test Stat = 2.6393, p - value = 0.1083

# Earning results:
# Test Stat = 0.3532, p - value = 0.5540

# HO Can't be rejected for all variables except Click.
# we can say that there is NO statistically significant difference between the variance distributions of the variables
# of the 2 groups, except for Click values There is statistically significant difference between the variance
# distributions of the Click values of the 2 groups

# Alternative: If the Variance Homogeneity assumption wasn't satisfied we could add an argument to the function
# to work that situation. (equal_var=False )
###########################################################

## A/B Testing:

##  Purchases

test_stat,pvalue = ttest_ind(df.loc[df["Group"] == "Control","Purchase"],df.loc[df["Group"] == "Test","Purchase"],equal_var=True)

print("Test Stat = %.4f, pvalue = %.4f" %(test_stat,pvalue)) # Test Stat = -0.9416, pvalue = 0.3493

# H0 can't be rejected.

######################################################

##  Impression

test_stat,pvalue = ttest_ind(df.loc[df["Group"] == "Control","Impression"],df.loc[df["Group"] == "Test","Impression"],equal_var=True)

print("Test Stat = %.4f, pvalue = %.4f" %(test_stat,pvalue)) # Test Stat = -4.2966, pvalue = 0.0000

# H0 is rejected.

######################################################

##  Earning

test_stat,pvalue = ttest_ind(df.loc[df["Group"] == "Control","Earning"],df.loc[df["Group"] == "Test","Earning"],equal_var=True)

print("Test Stat = %.4f, pvalue = %.4f" %(test_stat,pvalue)) # Test Stat = -9.2545, pvalue = 0.0000

# H0 is rejected.

######################################################

##  Click

test_stat,pvalue = mannwhitneyu(df.loc[df["Group"] == "Control","Click"],df.loc[df["Group"] == "Test","Click"])

print("Test Stat = %.4f, pvalue = %.4f" %(test_stat,pvalue)) # Test Stat = 1198.0000, pvalue = 0.0001

# H0 is rejected.


##############################################################
# Conclusions
##############################################################

# The Independent Samples t Test was chosen to make the A/B testing as both assumption were
# satisfied in (Purchases,Impression,Earning).

# The mannwhitneyu method was chosen to make the A/B testing for Click as there is statistically significant difference
# between the variance distributions of the Click values of the 2 groups.

# H0 can't be rejected, which means there is no significant difference between the new bidding system and the old one.

##### Results ######
# Purchase: Maximum bidding (Control Group) and Average bidding (Test Group) has the same average
# Click: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same average (Higher)
# Impression: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same average (Higher)
# Earning: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same average (Higher)



##############################################################
# Two Group Ratio Comparison
##############################################################

# Check if both tests have at least 30 samples:
df_control.shape # 40
df_test.shape # 40


df["Purchase_Per_Impression"] = df["Purchase"] / df["Impression"]
df["Purchase_Per_Click"] = df["Purchase"] / df["Click"]
df["Earning_Per_Click"] = df["Earning"] / df["Click"]
df["Earning_Per_Impression"] = df["Earning"] / df["Impression"]

df.head()

#     Impression      Click  Purchase    Earning    Group  Purchase_Per_Impression  Purchase_Per_Click  Earning_Per_Click  Earning_Per_Impression
# 0  82529.45927 6090.07732 665.21125 2311.27714  Control                  0.00806             0.10923            0.37952                 0.02801
# 1  98050.45193 3382.86179 315.08489 1742.80686  Control                  0.00321             0.09314            0.51519                 0.01777
# 2  82696.02355 4167.96575 458.08374 1797.82745  Control                  0.00554             0.10991            0.43134                 0.02174
# 3 109914.40040 4910.88224 487.09077 1696.22918  Control                  0.00443             0.09919            0.34540                 0.01543
# 4 108457.76263 5987.65581 441.03405 1543.72018  Control                  0.00407             0.07366            0.25782                 0.01423

df.groupby("Group")["Purchase_Per_Impression","Purchase_Per_Click","Earning_Per_Click","Earning_Per_Impression"].mean()

#          Purchase_Per_Impression  Purchase_Per_Click  Earning_Per_Click  Earning_Per_Impression
# Group
# Control                  0.00558             0.11593            0.40835                 0.01947
# Test                     0.00492             0.15657            0.66830                 0.02140

purchase_sum = np.array ([df_control["Purchase"].sum (), df_test["Purchase"].sum ()])
click_sum = np.array ([df_control["Click"].sum (), df_test["Click"].sum ()])
impression_sum = np.array ([df_control["Impression"].sum (), df_test["Impression"].sum ()])
Earning_sum = np.array ([df_control["Earning"].sum (), df_test["Earning"].sum ()])

#############################################################################

# Now the data is ready to be compared:

# Purchase_Per_Impression
ttest_z, pvalue = proportions_ztest (purchase_sum, impression_sum)
print('Test Stat = %.4f, p-value = %.4f' % (ttest_z, pvalue))
# Test Stat = 12.2212, p-value = 0.0000


# Purchase_Per_Click
ttest_z, pvalue = proportions_ztest (purchase_sum, click_sum)
print('Test Stat = %.4f, p-value = %.4f' % (ttest_z, pvalue))
# Test Stat = -34.9800, p-value = 0.0000


# Earning_Per_Click
ttest_z, pvalue = proportions_ztest (Earning_sum, click_sum)
print('Test Stat = %.4f, p-value = %.4f' % (ttest_z, pvalue))
# Test Stat = -155.2202, p-value = 0.0000


# Earning_Per_Impression
ttest_z, pvalue = proportions_ztest (Earning_sum, impression_sum)
print('Test Stat = %.4f, p-value = %.4f' % (ttest_z, pvalue))
# Test Stat = -22.3725, p-value = 0.0000


#############################################################################

##### Results ######

# Purchase_Per_Impression: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same
# average (lower)

# Purchase_Per_Click: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same
# average (Higher)

# Earning_Per_Click: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same
# average (Higher)

# Earning_Per_Impression: Maximum bidding (Control Group) and Average bidding (Test Group) doesn't have the same
# average (Higher)
