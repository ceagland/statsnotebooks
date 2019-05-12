#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# ## Regularization Methods
# 
# Regularization methods help prevent overfitting. The basic idea is to reduce variance by allowing some bias.
# 
# ### Ridge Regression
# 
# https://www.youtube.com/watch?v=Q81RR3yKn30&t=14s
# 
# Say we start with a linear regression that looks something like this:  
# $size = 0.4 + 1.3 \times weight$
# 
# With a lot of measurements, we can be somewhat sure the equation represents something close to the true relationship between size and weight.
# 
# Linear regression finds coefficients/weights by minimizing the sum of squared residuals, or SSR:
# 
# $\sum_{i=1}^{n} \left ( y_i - \hat{y_i} \right )^2$
# 
# If we only have 2 measurements, the SSR is 0... it's simply a line connecting the 2 points.
# 
# If we apply this model created with 2 measurements to a new set of data, it will likely have a high SSR. We would say our model has high variance... despite the line being a perfect predictor of the training set, it is **overfit** to the training data, and will likely not fit new data well.
# 
# In ridge regression, the idea is to fit a line that doesn't fit the training data as well. (Wait, what!?) No really... we introduce a little bias to get a meaningful drop in variance. With a slightly worse fit on the training data, we can get better long term predictions.
# 
# We accomplish this tradeoff by penalizing the coefficients in the model.
# 
# #### Ridge Regression Penalty
# 
# Linear Regression:  
# $size = intercept + slope \times weight$, where the slope selected minimizes the SSR
# 
# Ridge Regression:  
# Minimizes $SSR + \lambda \times slope^2$,
# 
# or more formally for multiple predictors:  
# 
# $\sum_{i=1}^{n} \left ( y_i - \hat{y_i} \right )^2 + \lambda \times \sum_{j=1}^{p} \beta_{j}^2 $
# 
# The second term is the penalty added to the least squares method. $\lambda$ is the severity of the penality.
# 
# So, for our equation fit with 2 data points:
# 
# $size = 0.4 + 1.3 \times weight$  
# 
# 

# In[2]:


x = [1, 3]
y = [1.7, 4.3]
x_line = [0, 6]
y_line = [0.4, 8.2]
plt.plot(x, y, 'ro')
plt.plot(x_line, y_line, 'k-')
plt.xlim(0,5)
a = plt.ylim(0,5)


# We just draw a line through the two points. The error is 0, as is the SSR, since of course summing up the 0 error is just 0.
# 
# Let's set $\lambda = 1$ and see what happens.
# 
# $adjusted SSR = 0 + 1 \times 1.3^2 = 1.69$  
# 
# That's what happens when we use the adjusted SSR on our current equation. If we perform ridge regression and select weights by minimizing the SSR + penalty, we get a different equation:
# 
# $size = 0.9 + 0.8 \times weight$  
# 
# In this case, we get an adjusted SSR of:  
# 
# $adjusted SSR = 0.3^2 + 0.1^2 + 1 \times 0.8^2 = 0.74$
# 
# So, the adjusted SSR is lower for ridge regression.
# 
# Linear regression is unbiased but has high variance. With ridge, we've introduced some bias, but it should fit the new data better.
# 
# When the slope of our model line is small, predictions are lsss sensitive to changes in weight.
# 
# A ridge regression line will have a smaller slope, so predictions with ridge regression are less sensitive.
# 
# #### Choosing $\lambda$
# 
# $\lambda$ can take on any value from $[0, \infty]$
# 
# If $\lambda = 0$, ridge regression and least squares are the same (since the penalty is nullified). The higher the $\lambda$, the smaller the slope of the resulting fit line. As we increase $\lambda$ to infinity, the slope gets closer to 0.
# 
# How do we choose $\lambda$? The most typical method is 10-fold cross validation.
# 
# #### Discrete data
# 
# Ridge regression will work with discrete data as well.  
# $size = 0.5 + 0.8 \times highfatdiet$
# 
# Let's pretend we're predicting the size of mice based on whether they were on a high fat diet vs a normal diet (whatever normal diet means in this case).
# 
# So, ridge regression will shrink the slope, and therefore shrink the effect of differences due to diet.  
# 
# This is very useful with small sample sizes.
# 
# #### Ridge in logistic regression  
# 
# $isobese = intercept + slope \times weight$
# 
# As with linear regression, ridge will still shrink the slope, making the prediction of obesity less sensitive to weight.  
# 
# In this case, ridge optimizes the sum of the likelihoods instead of the squared residuals.
# 
# #### What if there's more than one x variable?
# 
# In general, the ridge regression penalty contains all parameters except the y-intercept. It's just applied to every slope.  
# 
# $\lambda(\text{dietdifference}^2 + \text{astrologicaloffset}^2 + \text{airspeed}^2)$  
# 
# Every parameter is scaled by the measurements... and that's why we don't include the y-intercept.
# 
# #### One of the coolest things about ridge regression? (Maybe a bit subjective)  
# 
# In linear regression, to make a line, we need at least 2 points. So to fit one linear regression model, we need at least 2 data points. But, if we have 3 parameters, 2 data points isn't enough. We have to fit a plane. So, we need 3 parameters. With 10,001 parameters, we need 10,001 data points.
# 
# What if we have an equation with 10,000 parameters... but only 500 samples?
# 
# Ridge regression can do it! We can solve with fewer samples than parameters using cross validation.
# 
# #### Summary
# 
# When sample sizes are small, ridge regression can improve predictions made on new data (reduce variance) by making the predictions less sensitive to the training data.
# 
# This is done by adding ridge regression penalty to whatever is being minimized.
# 
# Even with fewer samples than variables/parameters, ridge regression can get estimates using cross validation.
# 
# ### Lasso Regression
# 
# Ridge = linear regression with ridge penalty (squared $\beta$s multiplied by a regularization parameter $\lambda$)
# 
# Ridge has more bias than linear regression, but at the gain of a more meaningful drop in variance.
# 
# Lasso is similar, but instead of squaring, we take absolute values:
# 
# $\sum_{i=1}^{n} \left ( y_i - \hat{y_i} \right )^2 + \lambda \times \sum_{j=1}^{p} \left | \beta_{j} \right |$
# 
# As with ridge, lasso trades off a gain in bias for a decrease in variance.
# 
# #### How else are lasso and ridge regression similar?
# 
# Both lasso and ridge:  
# 
# - Make predictions less sensitive to small sample sizes  
# - Can be applied in the same contexts:
#     - Linear regression
#     - Logistic regression
#     - Other more complicated models... the point is they can applied in a similar fashion  
#     
# When either ridge/lasso shrink parameters, they don't have to shrink them all equally.
# 
# An example using a binary variable for whether the subject was on a high fat diet (red points/line in plot below) or a normal diet (green points/line).
# 

# In[6]:


np.random.seed(1)
randomdata = {'x' : np.random.randint(low = 1, high = 10, size = 20), 'y' : np.random.randint(low = 1, high = 10, size = 20)}
randomdata = pd.DataFrame(data = randomdata)
randomdata['highfatdiet'] = randomdata['y'] > 4
randomdata['xiyi'] = randomdata['x'] * randomdata['y']
randomdata['xsq']  = randomdata['x']**2
means = randomdata.groupby('highfatdiet')[['x','y']].mean()
sums  = randomdata.groupby('highfatdiet')[['x','y','xsq','xiyi']].sum()
xbar_norm, xbar_high = means['x'][0], means['x'][1]
ybar_norm, ybar_high = means['y'][0], means['y'][1]
xiyisum_norm, xiyisum_high = sums['xiyi'][0], sums['xiyi'][1]
xsum_norm, xsum_high= sums['x'][0], sums['x'][1]
xsqsum_norm, xsqsum_high  = sums['xsq'][0], sums['xsq'][1]

b1_high = (xiyisum_high - (ybar_high * xsum_high)) / ((xbar_high * xsum_high) - xsqsum_high)
b0_high = ybar_high - b1_high*xbar_high

b1_norm = (xiyisum_norm - (ybar_norm * xsum_norm)) / ((xbar_norm * xsum_norm) - xsqsum_norm)
b0_norm = ybar_norm - b1_norm*xbar_norm

line_high = randomdata[randomdata['highfatdiet'] == True]['x'] * b1_high + b0_high
line_norm = randomdata[randomdata['highfatdiet'] == False]['x'] * b1_norm + b0_norm

randomdata['colors'] = np.where(randomdata['highfatdiet'] == True,'red', 'green')

plt.scatter(randomdata['x'],randomdata['y'],marker='o', c = randomdata['colors'])
plt.plot(randomdata[randomdata['highfatdiet'] == True]['x'], line_high, c = 'red')
a = plt.plot(randomdata[randomdata['highfatdiet'] == False]['x'], line_norm, c = 'green')


# The vertical distance between these lines is the difference in diets. 
# 
# Regularization may shrink these lines closer to one another more strongly than it shrinks their slopes.
# 
# #### Now, how are ridge and lasso different?
# 
# Ridge: Shrinks the slope asymptotically close to 0  
# Lasso: Can shrink the slope all the way to 0
# 
# $size = \beta_0 + slope \times weight + \text{dietdifference} \times HighFatDiet + \text{astrologicaloffset} \times sign + \text{airspeedscalar} \times AirSpeedofSwallow$  
# 
# Using the example formula above (where we know a couple of terms likely have nothing to do with the size of whatever we're measuring):
# 
# Ridge will shrink the slope of diet difference (probably meaningful) less than the slope of astrological offset and airspeed (both probably not meaningful), but both of the latter 2 will still be greater than 0.
# 
# Lasso will shrink the two useless variables all the way to 0, and we'll end up with:
# 
# $size = \beta_0 + slope \times weight + \text{dietdifference} \times HighFatDiet$  
# 
# Because of this feature, Lasso is a bit more effective in models with a lot of useless variables.
# 
# ### Elastic Net Regression
# 
# $Ridge: SSR + \lambda \sum \beta^2$  
# $Lasso: SSR + \lambda \sum \left | \beta \right |$  
# 
# What if we have a lot of parameters? Should we use Lasso or should we use Ridge?  
# 
# Don't choose! We have the option of elastic net.
# 
# $ElasticNet: SSR + \lambda \sum \beta^2 + \lambda \sum \left | \beta \right |$  
# 
# Note that we have 2 lambdas. In implementation, we'll typically pick a $\lambda$ and then a weight, so if the weight is 0.5 we'll apply half of the lambda value to ridge and half to lasso.
# 
# This works espeically well in situations when there is a lot of correlation between the variables.
# 
# Why?
# 
# Lasso: Picks one of the correlated variables  
# Ridge: Shrinks all parameters of correlated variables together
