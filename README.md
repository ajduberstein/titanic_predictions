titanic_predictions
===================
This is my initial submission to [the Titanic Kaggle competition](https://www.kaggle.com/c/titanic-gettingStarted)
using the Python [pandas](http://pandas.pydata.org/) library and [scikit-learn](http://scikit-learn.org/stable/).
The goal of this project is to answer a straightforward question: Given a list of attributes about Titanic passengers, can we predict who died and survived?

About 60% of passengers died, so a means model would be predicting death 100% of the time, which would have 60% accuracy.
This logistic regression model has a 72% accuracy on the test set, arguably a significant improvement on the naive means model.
