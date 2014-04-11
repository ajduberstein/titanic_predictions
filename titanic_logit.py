import pandas as pd
import numpy as np
from sklearn import linear_model

"""
Goal: Given features of a Titanic passenger, predict whether that passenger lived or died.

Survived = was_first +
           was_second +
           was_female + 
           age +
           fare +
           sibsp +
           parch +
           wasCherbourg + 
           wasQueenstown

"""
#Create the feature set
def make_features(filename):
  df = pd.read_csv(filename)
  df['was_first_class'] = df['Pclass'] == 1
  df['was_second_class'] = df['Pclass'] == 2
  df['was_female'] = df['Sex'] == 'female'
  df['was_Cherbourg'] = df['Embarked'] == 'C'
  df['was_Queenstown'] = df['Embarked'] == 'Q'
  df = df.fillna(df.mean())
  if 'test' in filename:
    return df.drop(['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked'], axis=1), df['PassengerId']
  outcomes = df['Survived']
  return df.drop(['Survived','PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked'], axis=1), outcomes
df, outcomes = make_features("train.csv")
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(df,outcomes)
df, passengers = make_features("test.csv")
pred_set = logreg.predict(df)
out = open('test_pred.csv','w')
out.write('PassengerId,Survived\n')
for outcome, pass_id in pred_set, passengers:
  out.write("%s,%s\n" % (pass_id,outcome))
out.close()
