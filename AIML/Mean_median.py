from numpy import *
from scipy import  stats
from statistics import *
speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
Mean=mean(speed)
Median=median(speed)
Mode=stats.mode(speed)
Standard_deviation=stdev(speed)
Percentile=percentile(speed,100)
print("Mean:",Mean)
print("Median:",Median)
print("Mode:",Mode)
print("Standard Deviation:",Standard_deviation)
print("Percentile:",Percentile)
