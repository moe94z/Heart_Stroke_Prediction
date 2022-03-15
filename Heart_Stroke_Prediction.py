# %% [markdown]
# ### Mohamad Quteifan
# ### Professor Catie Williams
# ### DSC 680: Applied Data Science
# ### Project Draft

# %% [markdown]
# # Project Proposal
# The purpose of the research is to create a model that will effectively predict which patients will have a heart stroke. The model that will be utilized is still under review and model evolution will be an essential to providing insight on the data. 
# 
# This has an opportunity to save lives if the model is effective.
# List of possible questions:
# 1. What is the most significant factor in heart strokes?
# 2. Average age for heart strokes?
# 3. Are the features presented in the data significant enough to produce an effective model?
# 4. Is Machine Learning the best method to tackle heart strokes? 
# 5. Do married patients at a higher risk of heart strokes compared to unmarried patience?
# 6. How important is BMI to heart strokes? 
# 7. What is the most important feature in the data?
# 8. Do the job types of the patients have an impact/relationship with strokes?
# 9. How many patients in the data has had a stroke?
# 10. What is the average glucose level of stroke patients?
#  

# %% [markdown]
# # Exploratory Data Analysis

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
from scipy import stats
!pip install missingno
import warnings
import time
import pickle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, f1_score, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from scipy import stats
!pip install pandas_profiling
import pandas_profiling

warnings.filterwarnings('ignore')
import pip
#pip.main(['install', 'xgboost'])
import sys
#!{sys.executable} -m pip install xgboost
#import xgboost as xgb
import missingno as msno 
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.linear_model import RandomForest, LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler

#from xgboost import XGBClassifier, XGBRegressor
#import xgboost as xgb


# %%
# import dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# %%
print("The shape of the data:\n",
      df.shape, 
      "\nThe first 5 rows are:\n", 
      df.head(), 
      "\nThe last 5 rows are:\n",
      df.tail(), 
      "\nThe column names are:\n",
      df.columns)

# %%
# pd.profile_report
df.profile_report()

# %%


# %% [markdown]
# ### Missing Data/Empty Values

# %%
df.isnull().sum()

# %% [markdown]
# The data is relatively clean, and the only feature that is missing values is the value BMI. 

# %% [markdown]
# #### Mean imputations, use the Mean inpace of the 201 BMI missing values

# %%
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df.info()

# %% [markdown]
# #### Remove ID feature from the data

# %%
print("Shape before dropping", df.shape)
df = df.drop(columns ='id')
print("Shape after dropping", df.shape)

# %% [markdown]
# #### Gender

# %%
df.gender.value_counts()

# %% [markdown]
# There are a few more female patients than male patients. This may be an issue in modeling we have to continue to review the data. I will be removing the value "other", there is only one individual who goes by anything other than other and it will not be useful in the research. Nothing against people who identify as anything outside the two genders, just for the sake of modeling including the "other" value would not be beneficial. 

# %%
# Drop other
df = df[df.gender != "Other" ]
df.shape


# %%
#This technique was taken from a medium article that I have linked in the reference section

df_cat = ['gender','hypertension','heart_disease','ever_married',
          'work_type','Residence_type','smoking_status', 'stroke']


# %% [markdown]
# ### Gender

# %%
sns.countplot(x=df["gender"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Gender & Stroke", weight='bold')
plt.xlabel('Gender', weight = 'bold')
plt.ylabel('Count', weight='bold')
plt.show()

# %% [markdown]
# #### Quick Analysis 
# Not much can be concluded from this information. It shows that females are slighty at a higher risk of suffering a stroke, but this should not be taken as evidence considering that the data consisted of more females. 

# %% [markdown]
# ### Hypertension

# %%
sns.countplot(x=df["hypertension"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Hypertension & Stroke", weight='bold')
plt.xlabel('Hypertension', weight = 'bold')
plt.ylabel('Count', weight='bold')
plt.show()

# %% [markdown]
# #### Analysis of the relationship between hypertension and stroke
# There seems to be a strong relationship between the two variables. The patients who were diagnosed with hypertension were likily to have a heart stroke. This does not come as a surprise to anyone and it makes sense. 

# %% [markdown]
# ### Heart Disease

# %%
sns.countplot(x=df["heart_disease"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Heart Disease & Stroke", weight='bold')
plt.xlabel("Heart Disease",weight='bold')
plt.ylabel("Count",weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Heart Disease and stroke
# The feature is nearly identical to the other feature hypertension. The biggest difference is that heart diseases seems to have an even more significant impact on stroke. We can conclude that individuals with a heart disease are more likely to suffer a stroke than an ordinary individual(if all other metrics are the same) 

# %% [markdown]
# ### Married

# %%
sns.countplot(x=df["ever_married"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Marriage & Stroke", weight='bold')
plt.xlabel("Marriage Status",weight='bold')
plt.ylabel("Count",weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Marriage and stroke
# Married individuals are more likely to sufffer a stroke compared to the never married individuals. I do want to point out that there are quite a few more married individuals which could mean that the conclusion may not be reliable.

# %% [markdown]
# ### Work Type

# %%
sns.countplot(x=df["work_type"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Work Type & Stroke", weight='bold')
plt.xlabel('Work Type', weight = 'bold')
plt.ylabel('Count', weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Work Type and stroke
# The analysis concluded that the individuals who were self employed or working in a private sector were at a much higher chance of suffering a stroke. The one main issue with the finding is that majority of the patients in the study worked in the private sector and the unequal distribution lead to the conclusion. We can conclude with confidence that self-employed patients are at the highest risk of suffering a stroke.

# %% [markdown]
# ### Residence_type

# %%
sns.countplot(x=df["Residence_type"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Residence Type & Stroke", weight='bold')
plt.xlabel('Residence Type', weight = 'bold')
plt.ylabel('Count', weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Residence Type and stroke
# I was not surprised to find that there is not relationship between Residence Type and stroke. The patients who lived in an Urban area were equally likely to get a stroke as the patients living in Rural areas. 

# %% [markdown]
# ### Smoking_status

# %%
sns.countplot(x=df["smoking_status"], data=df, hue =df['stroke'], palette = 'colorblind')
plt.title("Bar chart of Smoke Status & Stroke", weight='bold')
plt.xlabel('Smoke Status', weight = 'bold')
plt.ylabel('Count', weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Smoking Status and Stroke
# At first glance it seemed that patients who never smoked were likely to suffer a stroke at a higher rate than patients who formerly or currently smoke because of how the data is distributed. The analysis actually concludes that former and current smokers are at a higher risk of heart stroke, which we expected. 

# %% [markdown]
# ### Quantitative Features

# %%
df_quat = [ "age", "avg_glucose_level", "bmi"]

# %% [markdown]
# ### Age

# %%
sns.boxplot(x="stroke", y= "age", data=df,  palette = 'colorblind')
plt.title("Relationship between Age & Stroke", weight='bold')
plt.xlabel("Stroke", weight = 'bold')
plt.ylabel("Age", weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Age and Stroke
# The chart indicates that the older you are, the more likely you are to suffer a stroke. The mean age for the stroke patients is much higher than the mean for the non-stroke patients. This indicates a clear relationship between age and the dependent variable. 

# %% [markdown]
# ### Average Glucose Level

# %%
sns.boxplot(x="stroke", y= "avg_glucose_level", data=df,  palette = 'colorblind')
plt.title("Relationship between Average Glucose Level & Stroke", weight='bold')
plt.xlabel("Stroke", weight = 'bold')
plt.ylabel("Average Glucose Level", weight='bold')
plt.show()

# %%
sns.boxplot(x="stroke", y= "bmi", data=df,  palette = 'colorblind')
plt.title("Relationship between BMI & Stroke", weight='bold')
plt.xlabel("Stroke", weight = 'bold')
plt.ylabel("BMI", weight='bold')
plt.show()

# %% [markdown]
# #### Analysis on the relationship between the Residence Type and stroke
# I was unable to conclude anything from the data, it shows that the patients without a stroke contain a few outliers (individuals with a BMI greater than 55). The data in the BMI for patients without a stroke varies significantly compared to the patients with a stroke. The data points that are greater than 48 (above the 3 IQR) are considered outliers and we will need to remove the outliers from the data. Also, 48 is the highest a BMI can get based on https://www.medicalnewstoday.com/articles/323586#cutoff-points this health article. BMI 48 is considered super obese. 

# %%
outliers=df.loc[df['bmi']>48]
outliers.shape


# %%
### Replace outliers in BMI feature. 
df["bmi"] = pd.to_numeric(df["bmi"])
df["bmi"] = df["bmi"].apply(lambda x: 48 if x>48 else x)


# %% [markdown]
# # Outlier removal for Quantitative Features

# %%
from sklearn.ensemble import IsolationForest
isolation = IsolationForest(n_estimators = 1000, contamination = 0.03)
out = pd.Series(isolation.fit_predict(df[['bmi', 'avg_glucose_level']]),
                 name = 'outliers')
out.value_counts()
df = pd.concat([out.reset_index(), df.reset_index()], axis = 1,
               ignore_index = False).drop(columns = 'index')
df = df[df['outliers'] == 1]
df['stroke'].value_counts()


# %% [markdown]
# # Stroke, Unbalanced Data Set

# %%
sns.distplot(df.stroke)
plt.title("Distribution of Stroke", weight='bold')
plt.xlabel('Stroke', weight = 'bold')
plt.ylabel("Density", weight = "bold")
plt.show()

# %% [markdown]
# ### Analysis Stroke
# We clearly see an imbalance in the data. Although the outlier detection in BMI and average glucose level reduce the imbalance it is still significant and needs to be transformed to produce an effective model. The model that is created now will more than likely predict that all patients will not have a stroke which is incorrected and defeats the purpose of the research. 

# %%
#Balancing Stroke data 
scaler = StandardScaler()
df[df_quat] = scaler.fit_transform(df[df_quat])

# %%
sns.displot(df.stroke)
plt.show()

# %% [markdown]
# # Modeling

# %%
## I forgot to label encode the categorical features
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['gender'] = encoder.fit_transform(df['gender'])
df['ever_married'] = encoder.fit_transform(df['ever_married'])
df['work_type'] = encoder.fit_transform(df['work_type'])
df['Residence_type'] = encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = encoder.fit_transform(df['smoking_status'])

df_encoded = df
df_encoded.head()

# %%


# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

from imblearn.over_sampling import RandomOverSampler
def classification_eval(y_test, y_pred):
    print(f'accuracy  = {np.round(accuracy_score(y_test, y_pred), 3)}')
    print(f'precision = {np.round(precision_score(y_test, y_pred), 3)}')
    print(f'recall    = {np.round(recall_score(y_test, y_pred), 3)}')
    print(f'f1-score  = {np.round(f1_score(y_test, y_pred), 3)}')
    print(f'roc auc   = {np.round(roc_auc_score(y_test, y_pred), 3)}')
    print(f'null accuracy = {round(max(y_test.mean(), 1 - y_test.mean()), 2)}')



X = df_encoded.drop(columns = ['stroke', 'outliers'])
y = df_encoded.stroke


oversampler = RandomOverSampler()
X, y = oversampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                             stratify = y)
randomf = RandomForestClassifier()
randomf.fit(X_train, y_train)
y_pred = randomf.predict(X_test)

classification_eval(y_test, y_pred)

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                             stratify = y)
randomf = RandomForestClassifier()
randomf.fit(X_train, y_train)
y_pred = randomf.predict(X_test)
classification_eval(y_test, y_pred)

# %%
plot_confusion_matrix(randomf, X_test, y_test, cmap = "BuPu")
plt.title("Confusion Matrix" )
plt.show()

# %%



