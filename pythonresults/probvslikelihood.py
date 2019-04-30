#!/usr/bin/env python
# coding: utf-8

# ## Probability vs Likelihood
# 
# Just a short explanation to explain the difference between the two.
# 
# Virtually all of the explanation, and even the examples, has been taken from [this youtube video](https://www.youtube.com/watch?v=pYxNSUDSFH4) (great channel).

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


# First, create an array of normally distributed random numbers. To match the video linked above, use a mean of 32 with a standard distribution of 2.5, the location and shape of the distribution. A sample of 10,000 should be more than enough for what we're trying.

# In[2]:


np.random.seed(101)
x = np.random.normal(loc = 32, scale = 2.5, size = 10000)


# In[3]:


axs = sns.kdeplot(x)


# Easy enough. We see our data centered at around 32 and most of the data falling within 3 standard deviations ($3 \times 2.5 = 7.5$) of the mean. We'll use examples from this plot to compare probability and likelihood.
# 
# ### Probability
# 
# What is the probability a randomly selected observation from this distribution will be between 32 and 34?
# 
# A bit more formally, $p(32 < x < 34 \;  | \;  mean = 32\; and \;sd = 2.5) = \;?$
# 
# Well, it's just the area under the curve between 32 and 34. Probability is between 0 and 1, and the full area under the curve is 1. We just need the proportion that falls between 32 and 34. It's no different than splitting up a pie.

# In[4]:


kde_x, kde_y = axs.lines[0].get_data()

axs.fill_between(kde_x, kde_y, where=(np.logical_and(kde_x<34, kde_x>32)) , 
                interpolate=False, color='#EF9A9A')

axs.figure


# OK, that's the nice picture, but what's the actual area/probability?
# 
# We can use `scipy.stats.norm.cdf` to figure it out. We want to take the area under the curve for everything below 34, then subtract off the area of the curve below 32. The remainder will be everything from 32 to 34. In R, we use `pnorm()` to accomplish the same thing.

# In[5]:


norm.cdf(34, loc = 32, scale = 2.5) - norm.cdf(32, loc = 32, scale = 2.5)


# So, the *probability* that a randomly selected number from a normal distribution with mean 32 and standard deviation 2.5 will be between 32 and 34 is about 29% Or,
# 
# $p(32 < x < 34 \;  | \;  mean = 32\; and \;sd = 2.5) = \;0.29$
# 
# When we talk about probabilities, we're talking about a distribution that is described by the mean and standard deviation (the "given that" in the equation above). The area of the curve is described by $32 < x < 34$. So, using the exact same distribution, if we want to find the probability of a new area, we just change the $32 < x < 34$ to some other range.
# 
# ##### Verifying the probability
# 
# Well, above we've already created the variable `x`, which is 10,000 randomly generated numbers from a similar distribution. Of the 10,000, a rough 2,900 should be between 32 and 34.

# In[6]:


len(x[np.logical_and(x>32,x<34)])


# Looks to be right on what we'd expect!
# 
# 
# ##### A few more examples
# 
# Before moving on to likelihood and how it's different, we'll check **the probability that a randomly selected number from this distribution will be over 47**.

# In[7]:


1 - norm.cdf(37, loc = 32, scale = 2.5)


# A little over 2%.
# 
# Let's do two more for a refresher on normal probability and so we have an example for the "less thans". **What is the probability a randomly selected number from our distribution will be less than 32 (the mean)?**
# 
# ...
# 
# Yes, 0.5! In a (true) normal distribution, we expect half of the values to fall below the mean, and half fall above.
# 
# Finally, **what is the probability that a randomly selected number from this distribution will be less than 30?**

# In[8]:


norm.cdf(30, loc = 32, scale = 2.5)


# OK, about 21%. It's just the entire area of the curve to the left of the x value we choose, which is what norm.cdf returns anyway.

# ### Likelihood
# 
# To talk about likelihood, *assume* you have already selected a number from the distribution.
# 
# So we'll assume we have a number, 34, and it came from a distribution with mean 32 and standard deviation 25.
# 
# $L(mean = 32\; and \;sd = 2.5  | \;  x = 34\;  ) = \; ?$
# 
# Our left side is flipped. To get an answer here, we need the height of our curve when x = 34, not an area.

# In[9]:


linedat = pd.DataFrame([[34,0.12]], columns = ['x','y'])
sns.kdeplot(x)
plt.axvline(34, linestyle = '--')
axs = plt.axhline(norm.pdf(34, 32, 2.5), linestyle = '--')


# In[10]:


norm.pdf(34, 32, 2.5)


# The likelihood that the number 34 came from a distribution with mean 32 and standard deviation 2.5 is a little under 12%. The same function is available in R: `dnorm()`
# 
# $L(mean = 32\; and \;sd = 2.5  | \;  x = 34\;  ) = \; 0.116$
# 
# So, with likelihood, the *measurements* are fixed. We already know that x = 34. We want to know how likely it is that it came from a distribution with the location/shape of our distribution (the mean/sd).
# 
# ### Summary
# 
# Probabilities are the areas under a fixed distribution.
# 
# $p(data \; | \;  distribution \;  )$
# 
# Likelihoods are the y-axis values for fixed data points with distributions that can be moved.
# 
# $L(distribution \; | \;  data \;  )$
# 
