import pandas as pd
import lightgbm as lgb
import pickle

cropdf = pd.read_csv("crop_recommendation.csv")
crop_summary = pd.pivot_table(cropdf,index=['label'],aggfunc='mean')
X = cropdf.drop('label', axis=1)
y = cropdf['label']

model = lgb.LGBMClassifier()
model.fit(X, y)

pickle.dump(model, open('save_model.pkl','wb'))
