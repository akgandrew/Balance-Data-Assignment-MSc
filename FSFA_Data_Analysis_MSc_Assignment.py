"""
Created on Sat Nov 19 01:26:48 2022

@author: ag11afr
"""

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as stats

# name files to be used for typical data demonstration
file_eyes_open = "scp03_Session_1_eyes_open_29_9_2021_Balance_Per_Foot.xls"
file_eyes_closed = "scp03_Session_2_eyes_closed_29_9_2021_Balance_Per_Foot.xls"

# Create dataFrame for COP data, then seperate ML and AP 
# (ML = Mediloateral, AP = Anteroposterior) for single 
# participant eyes open
df_eyes_open = pd.read_excel(file_eyes_open)
df_eyes_open_COP_ML = df_eyes_open.iloc[16:,4]
df_eyes_open_COP_AP = df_eyes_open.iloc[16:,5]


# Create DataFrame for COP Data, then seperate ML and AP
#  (ML = Mediloateral, AP = Anteroposterior) for single 
# participant eyes closed
file = "scp03_Session_2_eyes_closed_29_9_2021_Balance_Per_Foot.xls"
df_eyes_closed = pd.read_excel(file_eyes_closed)
df_eyes_closed_COP_ML = df_eyes_closed.iloc[16:,4]
df_eyes_closed_COP_AP = df_eyes_closed.iloc[16:,5]


#fuction created to normalise COP data specifically
def zero_cop_data(cop_data):
    """
    adjusts data so mean of cop data is zero 
    so the data is normalised between participants.
    """
    cop_data = cop_data - cop_data.mean(axis = 0)
    return cop_data   


# all cop data zeroed about the mean (normalised)
df_eyes_closed_COP_ML = zero_cop_data(df_eyes_closed_COP_ML)
df_eyes_closed_COP_AP = zero_cop_data(df_eyes_closed_COP_AP)
df_eyes_open_COP_ML = zero_cop_data(df_eyes_open_COP_ML)
df_eyes_open_COP_AP = zero_cop_data(df_eyes_open_COP_AP)


# plot of COP line with ML and AP as x and y axis respectivly
plt.figure()
plt.plot(df_eyes_open_COP_ML, df_eyes_open_COP_AP,'r', label="Eyes open")
plt.plot(df_eyes_closed_COP_ML, df_eyes_closed_COP_AP, 'b', label="Eyes closed")
plt.xlabel("COP AP displacement (mm)")
plt.ylabel("COP ML displacement (mm)")
plt.legend()
plt.xlim(-0.5, 0.5)

# Save the figure and show
plt.tight_layout()
plt.savefig('Typical COP Data.png')
plt.show()


# Creata DataFrame of Pressure FsFa Data
df = pd.read_excel("COP_FsFA_Processed_Data.xls")


#Seperate data into closed and open eyes dataframes
open_eyes_data = df.loc[df['RAW File_Name'].str.contains("Open", case=False)]
closed_eyes_data = df.loc[df['RAW File_Name'].str.contains("closed", case=False)]


# Create function to report mean and std deviation for 
# dataframe named column (data_col) from dataframe (df) 
def cop_mean_stdev (data_col, df):
    mean = df[data_col].mean(axis=0)
    stdev = df[data_col].std(axis=0)
    mean_and_stdev = np.array([mean, stdev])
    return mean_and_stdev

# calculating mean for FSFA Score from chosen variable 
# e.g. ML Low Alpha (cop_def)
# for independent variable (e.g.eyes open or closed)data
mean_closed_eyes = cop_mean_stdev("RIGHTCOPMLLOW_alpha",closed_eyes_data )[0]
stdev_closed_eyes = cop_mean_stdev("RIGHTCOPMLLOW_alpha",closed_eyes_data )[1]
mean_open_eyes = cop_mean_stdev("RIGHTCOPMLLOW_alpha",open_eyes_data )[0]
stdev_open_eyes = cop_mean_stdev("RIGHTCOPMLLOW_alpha",open_eyes_data )[1]


# Make array of titles
conditions =  ['Eyes Closed', 'Eyes Open']

# Makes seperate arrays of mediolateral, low alpha, rightfoot FSFA Score in
# closed and open eyes then performs Ttest reporting tvalue[0] and p value[1].
# Enter dataframe column name for extraction (data_col) 
# from dataframe 1 (df1) and dataframe 2 (df2)
def cop_Ttest(data_col, df1, df2):
    """Performs Ttest on same columns of data from 2 different dataframes reporting t value[0] and p value[1]"""
    Ttest_result = stats.ttest_rel(df1[data_col], df2[data_col])
    return Ttest_result

# identifies p value and t value for alpha data comparing open and closed eyes
p_value = cop_Ttest("RIGHTCOPMLLOW_alpha", open_eyes_data, closed_eyes_data)[1]
t_value = cop_Ttest("RIGHTCOPMLLOW_alpha", open_eyes_data, closed_eyes_data)[0]

#creates array of means ready for plot
means = [mean_closed_eyes, mean_open_eyes]

#creates array of means ready for plot
stdevs = [stdev_closed_eyes, stdev_open_eyes]

#creates array with the number of data sets (bars) in the graph
x_pos = np.arange(len(conditions))


# Build bar chart comparing means with stdev error bars
fig, ax = plt.subplots()
ax.bar(x_pos, means, yerr=stdevs, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('α long term on COP position')
ax.set_xticks(x_pos)
ax.set_xticklabels(conditions)
ax.yaxis.grid(False)
plt.text(-0.015, 1.42, "*")

# Save the figure and show
plt.tight_layout()
plt.savefig('Eyes Open vs Eyes Closed ml_la_rt.png')
plt.show()



# create function to identify the ratio of one variable (var1) 
# to another ((var2) within subject so equal amounts, 
# and reports percentage of variable being larger above 
# a percentage freshold (p) 
def ratio_with_sig_dif(var1, var2, p):
    """
    Identifies ratio of one variable (var1) to another (var2) 
    within subject so equal amounts, and reports percentage of 
    variable being larger abover a percentage freshold (p)
    """
    var_size = var1.shape[0]
    var1_var_2_ratio = var1 / var2
    var1_sig_higher = var1_var_2_ratio  > 1 + p
    num_var1_sig_higher = var1_sig_higher.sum()
    var2_sig_higher = var1_var_2_ratio < 1 - p
    num_var2_sig_higher = var2_sig_higher.sum()
    num_non_sig_dif = var_size - num_var1_sig_higher - num_var2_sig_higher
    var1_var_2_no_sig = np.array([num_var1_sig_higher/var_size * 100, num_var2_sig_higher/var_size * 100, num_non_sig_dif/var_size * 100])
    return var1_var_2_no_sig


#Returns bias with number of right side highest, then left then no sig difference at p<0.05. note high score is poor balance so left and right switch in pie charts.
right_left_non_sig_bias_eyes_open = ratio_with_sig_dif(open_eyes_data["RIGHTCOPMLLOW_alpha"] , open_eyes_data["LEFTCOPMLLOW_alpha"] , 0.05)
right_left_non_sig_bias_eyes_closed = ratio_with_sig_dif(closed_eyes_data["RIGHTCOPMLLOW_alpha"] , closed_eyes_data["LEFTCOPMLLOW_alpha"] , 0.05)



#Plots a pie chart comparing right and left bias in participants with eyes open
balance_bias_labels = ["Non-dominant", "Dominant", "None"]
plt.figure()
plt.pie(right_left_non_sig_bias_eyes_open, labels=balance_bias_labels, normalize=True)
plt.title("Eyes open - α long term on COP position side bias")


# Save the figure and show
plt.tight_layout()
plt.savefig('Eyes open balance side bias.png')
plt.show()


# Plots a pie chart comparing right and left bias in participants with eyes closed
balance_bias_labels = ["Non-dominant", "Dominant", "None"]
plt.figure()
plt.pie(right_left_non_sig_bias_eyes_closed, labels=balance_bias_labels, normalize=True)
plt.title("Eyes closed - α long term on COP position side bias")


# Save the figure and show
plt.tight_layout()
plt.savefig('Eyes closed balance side bias.png')
plt.show()
