# Cross Sell Prediction

## **Description** 

This is for the datascience competition organised by Analytics Vidhya.
The task is to build a model to predict whether a customer would be interested in Vehicle Insurance which is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue. Now, in order to predict, whether the customer would be interested in Vehicle insurance, the dataset contains information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.

The evaluation metric for this competition is ROC_AUC score. 

It is a binary classification task to predict whether customer is interested(1), Customer is not interested(0).

## **Approach**

The approach here is that 3 different models LGBMClassifier, XGBClassifier and CatBoost Classifier were used to predict the probability. And the outputs from these three models were blended to further
 improve the score. The weights of the outputs used in blending were found by trial and error method. 
 
 ## **Result**
 
 On the test data the ROC_AUC score was 0.8589 and on the final benchmark dataset the score was 0.863444.
