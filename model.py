import pandas as pd
import lightgbm as lgb
import pickle

cropdf = pd.read_csv("crop_recommendation.csv")
crop_summary = pd.pivot_table(cropdf,index=['label'],aggfunc='mean')
X = cropdf.drop('label', axis=1)
y = cropdf['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 0)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open('save_model.pkl','wb'))