import numpy as np
from odor_tracking_sim.utility import speed_sigmoid_func
from odor_tracking_sim.utility import new_speed_sigmoid_func_perfect_controller as new_speed_sigmoid_func
import matplotlib.pyplot as plt

input_speeds = np.linspace(-3.,6.,1000.)
plt.figure()
ax = plt.subplot()
output = speed_sigmoid_func(input_speeds)
plt.plot(input_speeds,output,label='Old Version')
output = new_speed_sigmoid_func(input_speeds)
plt.plot(input_speeds,output,label='Perfect Controller, Asymmetric Thrusts')

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')


plt.legend(loc=(0.0,0.9))
plt.show()
