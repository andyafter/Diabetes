import h2o
import time
import seaborn
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

h2o.init()

# this is a script from kaggle https://www.kaggle.com/sudalairajkumar/getting-started-with-h2o
diabetes_df = h2o.import_file("diabetes.csv", destination_frame="diabetes_df")
# diabetes_df.describe()


# 把数据分成三份，0.6, 0.2, 0.2, 分别用来训练，验证和测试
train, valid, test = diabetes_df.split_frame(ratios=[0.6,0.2], seed=1234)
response = "Outcome"
train[response] = train[response].asfactor()
valid[response] = valid[response].asfactor()
test[response] = test[response].asfactor()
print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])

predictors = diabetes_df.columns[:-1]
gbm = H2OGradientBoostingEstimator()
gbm.train(x=predictors, y=response, training_frame=train)

perf = gbm.model_performance(valid)
print(perf)
