'''
Using the previous model, we'll simulate real time data acquistion from the pressure and indicator sensors.
That will be fed into th model and we'll have probabilities to classify the several states for the mechanical valve.
We'll plot this time using matplotlib library
'''

import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.animation import FuncAnimation
import tensorflow as tf

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,6))

axis2 = axes[0].twinx()
axes[0].set_title('Time Series Sensory Data')
axes[0].set_ylabel('Pressure (PSI)')
axes[0].set_xlim([0,250])
axes[0].set_ylim([10,85])
axis2.set_ylabel('Binary State')
axis2.set_ylim([0,5])
axis2.yaxis.set_ticks([0,1])

line1, = axes[0].plot([],[], color='green')
line2, = axes[0].plot([],[], color='blue')
line3, = axis2.plot([],[], color='black')
line4, = axis2.plot([],[], color='red')

# Define arrays for simulated data, initilized as empty
p1_test = []
p2_test = []
input_test = []
feedback_test = []
time = []

# Define categorical groups
x = ['Normal, Open','Normal, Close','Reverse, Open','Reverse, Close','Stuck Open','Stuck Close']
x_pos = [i for i,_ in enumerate(x)]

# Import tensorflow model
model = tf.keras.models.load_model('./models/Dense.h5')

# The animate function is passed to the FuncAnimation object. This function
# is ran every frame therefore it contains our model which we pass the randomly
# generated data to. It then appends the arrays and plots the data overtime.

def animate(i):
    # Generate random pressure data
    p1 = random.uniform(20.0,75.0)
    p2 = random.uniform(20.0,75.0)
    n1 = float(random.randint(0,1))
    n2 = float(random.randint(0,1))

    # Determine max pressure
    max_pressure = np.max([p1,p2])
    
    p1_test.append(p1)
    p2_test.append(p2)
    input_test.append(n1)
    feedback_test.append(n2)

    # Do a simply normalization of pressure data to feed model
    input_data = [p1/max_pressure,p2/max_pressure,n1,n2]
    input_data = np.expand_dims(input_data,axis=0)
    # Returns an array of probabilites for 6 classifcation categories
    forecast = model.predict(input_data)
    forecast = forecast[0,:]
    
    line1.set_data(np.arange(len(p1_test)),p1_test)
    line2.set_data(np.arange(len(p1_test)),p2_test) 
    line3.set_data(np.arange(len(p1_test)),input_test)
    line4.set_data(np.arange(len(p1_test)),feedback_test)
    axes[1].cla()
    axes[1].set_title('Current Equipment State Probabilities')
    axes[1].set_xlim([0,1])
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(x)
    axes[1].barh(x_pos,forecast)
    fig.tight_layout()
    
    return line1,line2,line3,line4

anim = FuncAnimation(fig,animate,frames=250,interval=100)

anim.save('./figs/AnimatedClassification.mp4')
