__author__ = "Kevin Trejos Vargas"
__email__  = "kevin.trejosvargas@ucr.ac.cr"

"""
MIT License

Copyright (c) 2023 Kevin Trejos Vargas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


################################################################################
# --                          Necessary libraries                           -- #
################################################################################
from scipy.stats import normaltest, ttest_ind
import matplotlib.pyplot as plt
import os, sys


################################################################################
# --                            Global variables                            -- #
################################################################################
outWD                  = os.getcwd().replace('src','output')
svcAccuraciesfile      = open(f"{outWD}/accuracies_SVC.log", 'r')
nnAccuraciesfile       = open(f"{outWD}/accuracies_NN.log" , 'r')
accuraciesSVC          = []
accuraciesNN           = []
accuraciesSVC_averaged = []
accuraciesSNN_averaged = []
iSampleCounter         = 0
iNumbersToAverage      = 7

print("AVERAGING DEFINITION:")
if len(sys.argv) >= 2:
    iNumbersToAverage = int(sys.argv[1])
    print(f"    Averaging {iNumbersToAverage} samples.")
else:
    print(f"    Defaulted to average {iNumbersToAverage} samples.")


################################################################################
# --                         Extract the accuracies                         -- #
################################################################################
for line in svcAccuraciesfile:
    accuraciesSVC.append(float(line.split(" ")[-1].replace("\n", "")))
for line in nnAccuraciesfile:
    accuraciesNN.append(float(line.split(" ")[-1].replace("\n", "")))
print(accuraciesSVC_averaged)


################################################################################
# --       Average the samples according to the central limit theorem       -- #
################################################################################
# -- Average the samples according to the central limit theorem
for value in range(len(accuraciesSVC)):
    if value + iNumbersToAverage - 1 < len(accuraciesSVC):
        fSummation  = sum(accuraciesSVC[value:value+iNumbersToAverage])
        iSamplesQty = len(accuraciesSVC[value:value+iNumbersToAverage])
        accuraciesSVC_averaged.append(fSummation/iSamplesQty)
    iSampleCounter += 1
for value in range(len(accuraciesNN)):
    if value + iNumbersToAverage - 1 < len(accuraciesNN):
        fSummation  = sum(accuraciesNN[value:value+iNumbersToAverage])
        iSamplesQty = len(accuraciesNN[value:value+iNumbersToAverage])
        accuraciesSNN_averaged.append(fSummation/iSamplesQty)
    iSampleCounter += 1


################################################################################
# --                      Create a combined histogram                       -- #
################################################################################
plt.hist(
    accuraciesSNN_averaged,
    bins      = 30,
    edgecolor = 'black',
    color     = 'skyblue',
    alpha     = 0.5
)
plt.hist(
    accuraciesSVC,
    bins      = 30,
    edgecolor = 'black',
    color     = 'orange',
    alpha     = 0.5
)
plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.title('Histogram of the Accuracy - SNN (skyblue) - SVC (orange)')
plt.show()


################################################################################
# --               Perform a normality test over SVC approach               -- #
################################################################################
print("NORMALITY TEST FOR SVC APPROACH:")
statistic, fPValue = normaltest(accuraciesSVC_averaged)
print(f"    P-value: {fPValue}")
if fPValue > 0.05:
    print("    Data appears to be normally distributed.")
else:
    print("    Data does not appear to be normally distributed.")


################################################################################
# --               Perform a normality test over SNN approach               -- #
################################################################################
print("NORMALITY TEST FOR SNN APPROACH:")
statistic, fPValue = normaltest(accuraciesSNN_averaged)
print(f"    P-value: {fPValue}")
if fPValue > 0.05:
    print("    Data appears to be normally distributed.")
else:
    print("    Data does not appear to be normally distributed.")


################################################################################
# --                      Perform the hipothesis test                       -- #
################################################################################
print("HYPOTHESIS TEST:")
t_statistic, fPValue = ttest_ind(
    accuraciesSNN_averaged,
    accuraciesSVC_averaged,
    alternative = 'greater'
)
print(f"    P-value: {fPValue}")
if fPValue < 0.05:
    print("    Mean of SNN is significantly higher than the mean of SVC.")
else:
    print("    Mean of SNN is not significantly higher than the mean of SVC.")
