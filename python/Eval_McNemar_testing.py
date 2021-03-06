# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""


from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

"""
    File 11 
    
    This file conduscts the McNemars test ont he different pairs of
    networks. The primary focus is between the CNN and the Capsule Network, 
    output is recorded as part of the analysis.
    
"""
# read in the file for mcnemars test
file = ("D:/College/year_4/semester_2/Software_project/Discovery2/results/McNemar_data.xlsx")
df = pd.read_excel(file)

# crosstab of results
table = pd.crosstab(df['CNN'], df['CapsNet'])


# calculate mcnemar test
result = mcnemar(table, exact=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')