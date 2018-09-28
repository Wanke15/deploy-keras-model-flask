import numpy as np
from numpy.random import RandomState
r = RandomState(101)

import pandas as pd

from keras.layers import Input, Dense
from keras.models import Model

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.externals import joblib

DATA_NUM = 1000

defect_type = r.randint(0, 3, DATA_NUM)
defect_group = r.randint(0, 3, DATA_NUM)
defect_priority = r.randint(0, 3, DATA_NUM)
defect_severity = r.randint(0, 3, DATA_NUM)

defect_description_vec = r.randint(0, 100, DATA_NUM)
X = pd.DataFrame({"Type":defect_type, "Group":defect_group, "Priority":defect_priority, "Severity":defect_severity, 
				"Desc_vec":defect_description_vec})
				
numeric_features = ['Desc_vec']
numeric_transformer = StandardScaler()

categorical_features = ['Type', 'Group', 'Priority', 'Severity']
categorical_transformer = OneHotEncoder(categories='auto')

preprocessor = ColumnTransformer(
	transformers=[
	('num', numeric_transformer, numeric_features),
	('cat', categorical_transformer, categorical_features)],
	remainder='passthrough')
X_new = preprocessor.fit_transform(X)

### Save preprocessor
joblib.dump(preprocessor, 'preprocessor.prep')

defect_cost = (4*X_new[:,0] + 2.6*X_new[:,1] + 3.3*X_new[:,2]**2 + 3*X_new[:,3]**3 + X_new[:,4] + 3.7*X_new[:,5]**2 + 
4*X_new[:,6] + 2.6*X_new[:,7] + 3.3*X_new[:,8] + 3*X_new[:,9]**3 + X_new[:,10] + 3.7*X_new[:,11] + 5.1*X_new[:,12])
defect_duration = (7*X_new[:,0] + 3.7*X_new[:,1] + 3.1*X_new[:,2]**2 + 3*X_new[:,3]**3 + X_new[:,4] + 3.7*X_new[:,5]**2 + 
1*X_new[:,6] + 6.2*X_new[:,7] + 9.5*X_new[:,8] + 4.3*X_new[:,9] + X_new[:,10] + 3.7*X_new[:,11] + 5.1*X_new[:,12])

y_cost = pd.DataFrame({"Cost":defect_cost})
y_duration = pd.DataFrame({"Duration":defect_duration})

train_indexs = np.arange(len(X))
r.shuffle(train_indexs)

### Split train test data
X_train, X_test = pd.DataFrame(X_new).iloc[train_indexs], pd.DataFrame(X_new).iloc[~train_indexs]
y_cost_train, y_cost_test = y_cost.iloc[train_indexs], y_cost.iloc[~train_indexs]
y_duration_train, y_duration_test = y_duration.iloc[train_indexs], y_duration.iloc[~train_indexs]

### Build neural network
Input_1= Input(shape=(X_train.shape[1], ))
x = Dense(100, activation='relu')(Input_1)

out_cost = Dense(1,  activation='relu', name='cost_layer')(x)
out_duration = Dense(1,  activation='relu', name='duration_layer')(x)

model = Model(inputs=Input_1, outputs=[out_cost, out_duration])
model.compile(optimizer = "rmsprop", loss = 'mse')

EPOCHS = 20
BATCH_SIZE = 8

history_train = model.fit(X_train, [y_cost_train, y_duration_train], batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS)

model_json = model.to_json()
with open("./model/model.json", 'w') as json_file:
	json_file.write(model_json)
	
model.save_weights("./model/model_weights.h5")