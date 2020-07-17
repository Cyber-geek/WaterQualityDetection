import pandas as pd
WQI_parameters = pd.read_csv('WaterQualityData.csv')
df = pd.DataFrame(WQI_parameters, columns=["pH","TN",	"BOD5","TP",	"NH3+","COD", "Iron", "Copper", "Zinc","DO","TDS","Ca","Mg","Na","Cl-",	"HCO", "SO4", "PO4","Cr"])
print (df)

df = df.fillna(0)

import numpy as np
W = np.array([0.072, 0.054, 0.091, 0.054, 0.054, 0.072, 0.018, 0.018, 0.036, 0.018, 0.018, 0.036, 0.036, 0.036, 0.054, 0.054, 0.072, 0.091, 0.018])
S = np.array([7.65, 0.50, 3.00, 0.10, 0.50, 15.00, 0.30, 1.00, 1.00, 6.00, 450.00, 300.00, 30.00, 200.00, 250.00, 1, 250.00, 50.00, 0.05])
df_y = pd.DataFrame(columns=['WQI'])
for index, row in df.iterrows():
  	C = np.array(row)
  	Q = (C / S) * 100
  	SI = W * Q
  	df_y.loc[index] = [np.sum(SI)]

df_y.plot(style=["o"])

df_y[(df_y['WQI'] > 800.00)].count()

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = df
Y = df_y

# define base model
            def baseline_model():
# create model
	model = Sequential()
	model.add(Dense(19, input_dim=19, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def larger_model():

# create model
	model = Sequential()
	model.add(Dense(19, input_dim=19, kernel_initializer='normal', activation='relu'))
	model.add(Dense(9, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
