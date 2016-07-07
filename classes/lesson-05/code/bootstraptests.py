import bootstrap
import pandas as pd
import numpy as np
#We're going to do a simulation 1,000 times to show that even with a sample of 100 people out a population of 1,000,000, our 95% confidence intervals will indeed capture the true mean of the population from that sample 95% of the time

#Make an empty list to hold whether or not our confidence intervals captured the true population mean 
results_bootstrap=[]
for i in range(1000):
	
	#population = pd.Series(np.random.normal(10,5,10000))
	
	#uncomment the line blow to see how well the confidence interval approximates a non-normal distribution
	#make a population with a heavily skewed distribution and that has 1,000,000 people in it
	population = pd.Series(np.random.power(9,1000000))
	
	#this is True mean of the entire 1,000,000 population
	true_pop_mean = population.mean()
	
	#We are going to collect a sample of 100 people from the 1,000,000 population and then calculate a boostrap
	sample_data = population.sample(100)
	
	#We calculate a 95% confidence interval on the data
	ci = bootstrap.confidence_interval(sample_data)
	
	#Get the lower and upper bounds of the confidence interval
	boot_ci_L=ci["0.95% Confidence Interval"][0]
	boot_ci_U=ci["0.95% Confidence Interval"][1]

	#See if our true population mean is with the bounds
	if true_pop_mean >= boot_ci_L and true_pop_mean <= boot_ci_U:
		#if it is, count it as a success (1)
		results_bootstrap.append(1)
	#if the true population mean is not within the bounds, count it as a failure
	else: results_bootstrap.append(0)
	
	#Print the average of all of the successes and failures up to this point
	print np.mean(results_bootstrap)
	
"""

#The Central Limit Theorem does not  apply to heavily tailed data
#However, as seen in the abov example, bias correction procedures in the bootstrap code allow you to attain accurate confidence intervals for heavily skewed non-normal data
import pandas as pd
import numpy as np
population = pd.Series(np.random.binomial(1,.999,1000))
samples = (population.sample(len(population),replace=True) for i in range(100000))
means = map(np.mean,samples)

import matplotlib.pyplot as plt
plt.hist(means)
plt.title("Histogram")
plt.xlabel("Mean of Bootstrap Samples")
plt.ylabel("Frequency")
plt.show()
plt.gcf()

"""

