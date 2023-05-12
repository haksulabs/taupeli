# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:20:51 2023

# plot normal distribution function for illustration 

@author: t
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mu, sigma = 1, 0.5  # set mean and standard deviation
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)  # create 100 evenly spaced values between 3 standard deviations below and above the mean
y = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-mu)**2/(2*sigma**2))  # calculate the normal distribution for each value of x



mu2 = 0.7

y2 = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-mu2)**2/(2*sigma**2))  # calculate the normal distribution for each value of x

# Set the number of tick marks on the x and y axes
# Set the number of tick marks on the x and y axes
num_ticks_x = 6
num_ticks_y = 5

# Create the plot
fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(x, y2)

# Draw a vertical line at x=0
ax.axvline(x=0, color='black', linestyle='--')

# Set the x and y axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('Probability density')
ax.set_title('Normal Distribution with mean = 1 and std = 0.5')

# Set the number of tick marks on each axis
ax.set_xticks(np.linspace(-0.5, 2, num_ticks_x))
ax.set_yticks(np.linspace(0, 0.8, num_ticks_y))

# Center the x-axis numbering
xticks = ax.get_xticks()
xticks = np.append(xticks, [0])
xticks.sort()
mid_tick = len(xticks)//2
#ax.set_xticks(xticks[mid_tick-2:mid_tick+3])

# Show the plot

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


plt.savefig('normal_distribution.svg', format='svg')


plt.show()
