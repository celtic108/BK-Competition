import pandas as pd
import numpy as np
import pylab as P
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../Data/train_numeric.csv', header = 0, nrows=99999)
test = pd.read_csv('../Data/train_numeric.csv', header = 0, skiprows=100000, nrows=99999)
#print type(df)
#df.types --- causes error
#print df.info()
print df.describe()  #Displays summary data about columns
#print df.head(3)
df.fillna(-9, inplace = True)
#print df.head(3)
test.fillna(-9, inplace = True)

#df['L0_S0_F4'].hist() #this shows a definite bimodal distribution
#P.show()


traindata = df.values
testdata = test.values
#code for random forest classifier
#forest = RandomForestClassifier(n_estimators = 968)
#forest = forest.fit(traindata[0::,0:968], traindata[0::,969])
#score = forest.score(testdata[0::,0:968], testdata[0::,969])
#output = forest.predict(testdata[0::,0:968])

#code for boosted clasifier
booster = GradientBoostingClassifier(n_estimators = 968)
booster = booster.fit(traindata[0::,0:968], traindata[0::,969])
score = booster.score(testdata[0::,0:968], testdata[0::,969])
output = booster.predict(testdata[0::,0:968])


print score
print type(output)
np.savetxt("../output files/fake_trial.csv", output, delimiter=",")
#output.to_csv('../output files/fake_trial.csv, sep=',')