'''
Main script - Classification State for Mechanical Valve

This is the main script for creating a model and predictinga simulation of live data for the state of a solenoid valve. It is a categorical classification based on two pressure sensors and two positional indicators. A DNN is trained on simulated data and randomly generated data is fed into the model to classify equipment states.
'''

# Import Libraries
import pandas as pd
import numpy as np
from src.data import DataMetrics,MinMaxNormalization,OneShot,ModelSplit
from src.model import compile_and_fit,Dense

# Enter csv data path if different or change file name
csv_path = "./data/mech_state_labels.csv"

# Read the data info a dataframe
df = pd.read_csv(csv_path)

# Print metrics of dataframe
DataMetrics(df)

# This data was generated in sequence and is not randomized. First we'll shuffle the data
df = df.sample(frac = 1)

# The pressure data has a wide range we'll normalize specific to each column.
df = MinMaxNormalization(df, 'Pressure_1', 'Pressure_2')

# Now we'll one shot the unique labels and return two data frames. One with data and the other with labels.
df, label_df = OneShot(df, 'labels')

# Next we split the two dataframes into test and training data.
X_train, X_test, y_train, y_test = ModelSplit(df, label_df, 0.3)

# Now we'll import a model we've defined, compile the model and begin training.
MAX_EPOCHS = 5
history = compile_and_fit(Dense, X_train, y_train, X_test, y_test, MAX_EPOCHS)

# We can export this model
print('\nSaving model...')
Dense.save('./models/Dense.h5')
