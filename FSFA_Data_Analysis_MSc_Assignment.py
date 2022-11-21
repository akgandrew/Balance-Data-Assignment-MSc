# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 01:26:48 2022

@author: ag11afr
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:26:54 2022

@author: ag11afr
"""

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

import scipy.stats as stats

# Change folder to C:\Users\ag11afr\.spyder-py3\Balance_Data



# Create dataFrame for COP data, then seperate ML and AP (ML = Mediloateral, AP = Anteroposterior) for single participant eyes open

df_eyes_open = pd.read_excel("scp03_Session_1_eyes_open_29_9_2021_Balance_Per_Foot.xls")

df_eyes_open_COP_ML = df_eyes_open.iloc[16:,4]
df_eyes_open_COP_AP = df_eyes_open.iloc[16:,5]

# Create DataFrame for COP Data, then seperate ML and AP (ML = Mediloateral, AP = Anteroposterior) for single participant eyes closed
df_eyes_closed = pd.read_excel("scp03_Session_2_eyes_closed_29_9_2021_Balance_Per_Foot.xls")

df_eyes_closed_COP_ML = df_eyes_closed.iloc[16:,4]
df_eyes_closed_COP_AP = df_eyes_closed.iloc[16:,5]


"""
Funtion sets mean of cop data to zero so the data is normalised between participants.
"""
def zero_cop_data(cop_data):
   cop_data = cop_data - cop_data.mean(axis = 0)
   return cop_data   

# all cop data zeroed about the mean
df_eyes_closed_COP_ML = zero_cop_data(df_eyes_closed_COP_ML)
df_eyes_closed_COP_AP = zero_cop_data(df_eyes_closed_COP_AP)
df_eyes_open_COP_ML = zero_cop_data(df_eyes_open_COP_ML)
df_eyes_open_COP_AP = zero_cop_data(df_eyes_open_COP_AP)

# plot of cop line with ML and AP as x and y axis respectivly

plt.figure()
plt.plot(df_eyes_open_COP_ML, df_eyes_open_COP_AP,'r', label="Eyes open")
plt.plot(df_eyes_closed_COP_ML, df_eyes_closed_COP_AP, 'b', label="Eyes closed")

plt.xlabel("COP Mediolateral displacement (mm)")
plt.ylabel("COP Anteriorposterior displacement (mm)")
plt.legend()
plt.xlim(-0.5, 0.5)
plt.show()
plt.tight_layout()
plt.savefig('Typical COP Data.png')

# Creata DataFrame of Pressure FsFa Data

df = pd.read_excel("COP_FsFA_Processed_Data.xls")

print(df)



#Seperate Data into Closed and Open Eyes DataFrames
Open_Eyes_Data = df.loc[df['RAW File_Name'].str.contains("Open", case=False)]
Closed_Eyes_Data = df.loc[df['RAW File_Name'].str.contains("closed", case=False)]

#calculating mean for FSFA Score from chosen variable e.g. ML Low Alpha (cop_def)
#for independent variable (e.g.eyes open or closed)data
def cop_mean(cop_def, cop_data):
    cop_data = cop_data[cop_def].mean(axis=0)
    return cop_data

Mean_Closed_Eyes =cop_mean("RIGHTCOPMLLOW_alpha",Closed_Eyes_Data )
Mean_Open_Eyes =cop_mean("RIGHTCOPMLLOW_alpha",Open_Eyes_Data )

#calculating standad deviation for FSFA Score from chosen variable e.g. ML Low Alpha (cop_def)
#for FSFa COP data defined by independent variable e.g.eyes open or closed ((cop_data))

def cop_stdev(cop_def, cop_data):
    cop_data = cop_data[cop_def].std(axis=0)
    return cop_data

Stdev_Closed_Eyes =cop_stdev("RIGHTCOPMLLOW_alpha",Closed_Eyes_Data )
Stdev_Open_Eyes =cop_stdev("RIGHTCOPMLLOW_alpha",Open_Eyes_Data )



conditions =  ['Eyes Open', 'Eyes Closed']

#Makes seperate arrays of mediolateral, low alpha, rightfoot FSFA Score in closed and open eyes
eyes_open_data = Open_Eyes_Data["RIGHTCOPMLLOW_alpha"]
eyes_closed_data = Closed_Eyes_Data["RIGHTCOPMLLOW_alpha"]

def cop_Ttest(cop_def, cop_data1, cop_data2):
    Ttest_result = stats.ttest_rel(cop_data1[cop_def], cop_data2[cop_def])
    return Ttest_result

Ttest_Result = cop_Ttest("RIGHTCOPMLLOW_alpha", Open_Eyes_Data, Closed_Eyes_Data)



#creates array of means ready for plot
means = [Mean_Closed_Eyes, Mean_Open_Eyes]

#creates array of means ready for plot
stdevs = [Stdev_Closed_Eyes, Stdev_Open_Eyes]

#creates array with the number of data sets (bars) in the graph
x_pos = np.arange(len(conditions))



# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, means, yerr=stdevs, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Low frequencey COP position (mm)')
ax.set_xticks(x_pos)
ax.set_xticklabels(conditions)
ax.set_title('Balance Condition')
ax.yaxis.grid(False)
plt.text(-0.015, 1.42, "*")

# Save the figure and show
plt.tight_layout()
plt.savefig('Eyes Open vs Eyes Closed ml_la_rt.png')
plt.show()


var1 = Open_Eyes_Data["RIGHTCOPMLLOW_alpha"] 
var2 = Open_Eyes_Data["LEFTCOPMLLOW_alpha"] 
p = 0.05
##Identifies ratio of one variable (var1) to another ((var2) within subject so equal amounts, and reports percentage of variable being larger abover a percentage freshold (p) 
def ratio_with_sig_dif(var1, var2, p):
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
right_left_non_sig_bias_eyes_open = ratio_with_sig_dif(Open_Eyes_Data["RIGHTCOPMLLOW_alpha"] , Open_Eyes_Data["LEFTCOPMLLOW_alpha"] , 0.05)
right_left_non_sig_bias_eyes_closed = ratio_with_sig_dif(Closed_Eyes_Data["RIGHTCOPMLLOW_alpha"] , Closed_Eyes_Data["LEFTCOPMLLOW_alpha"] , 0.05)






#Plots a pie chart comparing right and left bias in participants with eyes open

balance_bias_labels = ["Left", "Right", "None"]
plt.figure()
plt.pie(right_left_non_sig_bias_eyes_open, labels=balance_bias_labels, normalize=True)
plt.title("Eyes open balance side bias")
plt.show()
plt.tight_layout()
plt.savefig('Eyes open balance side bias.png')


#Plots a pie chart comparing right and left bias in participants with eyes closed


balance_bias_labels = ["Left", "Right", "None"]
plt.figure()
plt.pie(right_left_non_sig_bias_eyes_closed, labels=balance_bias_labels, normalize=True)
plt.title("Eyes closed balance side bias")
plt.show()
plt.tight_layout()
plt.savefig('test Eyes Open vs Eyes Closed ml_la_rt.png')
plt.show()
