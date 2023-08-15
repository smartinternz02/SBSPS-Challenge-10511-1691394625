import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import lightgbm as lgb
import pickle

colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
           '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
           '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']

cropdf = pd.read_csv("crop_recommendation.csv")
crop_summary = pd.pivot_table(cropdf,index=['label'],aggfunc='mean')
X = cropdf.drop('label', axis=1)
y = cropdf['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    shuffle = True, random_state = 0)

# build the lightgbm model

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open('save_model.pkl','wb'))