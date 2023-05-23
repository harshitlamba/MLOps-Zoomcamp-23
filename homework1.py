import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

PATH = "./"

def readData(path: str) -> dict:
    data = {}     
    for file in os.listdir(path):
        if (file.endswith(".parquet")):
            data[file.split("_")[-1].split(".")[0]] = pd.read_parquet(os.path.join(path,file),engine="pyarrow")
    return data

tripData = readData(PATH)
januaryData = tripData['2022-01']
februaryData = tripData['2022-02']
beforeOutlierRemoval = januaryData.shape[0]
print(f"Q1. Data for January and February 2022 read. The number of columns in January data are: {len(januaryData.columns)}.")

januaryData['duration'] = [(i.total_seconds())/60 for i in (januaryData['tpep_dropoff_datetime'] - januaryData['tpep_pickup_datetime'])]
februaryData['duration'] = [(i.total_seconds())/60 for i in (februaryData['tpep_dropoff_datetime'] - februaryData['tpep_pickup_datetime'])]
print(f"Q2. The standard deviation of the trips duration in January is: {round(np.std(januaryData['duration']),2)}")

januaryData = januaryData[(januaryData['duration'] >= 1) & (januaryData['duration'] <= 60)]
februaryData = februaryData[(februaryData['duration'] >= 1) & (februaryData['duration'] <= 60)]
recordsLeft = round(januaryData.shape[0]/beforeOutlierRemoval,2)
print(f"Q3. The fraction of records left after dropping outliers is: {recordsLeft}")

catVars = ['PULocationID','DOLocationID']
for var in catVars:
    januaryData.loc[:,var] = [str(i) for i in januaryData[var]]
    februaryData.loc[:,var] = [str(i) for i in februaryData[var]]
januaryDataSubsetDict = januaryData[catVars].T.to_dict().values()
februaryDataSubsetDict = februaryData[catVars].T.to_dict().values()
vectorizer = DictVectorizer()
vectorizer.fit(januaryDataSubsetDict)
train = vectorizer.transform(januaryDataSubsetDict)
validation = vectorizer.transform(februaryDataSubsetDict)
print(f"Q4. The dimensionality of the feature matrix used for model training is: {train.shape[1]}")

X_train, y_train = train, januaryData['duration'].values
X_validation, y_validation = validation, februaryData['duration'].values
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Q5. RMSE on train: {np.round(mean_squared_error(y_train,model.predict(X_train),squared=False),2)}")
print(f"Q6. RMSE on validation: {np.round(mean_squared_error(y_validation,model.predict(X_validation),squared=False),2)}")
