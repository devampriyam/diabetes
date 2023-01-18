#!/usr/bin/env python
# coding: utf-8

# In[391]:


import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# In[392]:


df=pd.read_csv('diabetes_data.csv').sample(5000).reset_index(drop=True)


# In[393]:


# def correlation():
#     return df.corr(method='spearman')['diabetes']


# In[394]:


# df.shape


# In[395]:


# df.describe(include='all')


# In[396]:


df.columns=df.columns.str.lower()


# In[397]:


# df.head(5)


# In[398]:


# sns.kdeplot(x='age',hue='diabetes',data=df)


# In[399]:


# df['age_chance']=np.where(df['age']>8,1,0)


# In[400]:


# plt.hist(df['age'],edgecolor='black',bins=3)[1]


# In[401]:


# label=['low','mid','high']
# df['age_range']=pd.cut(df['age'],[ 0,  5.,  9., 13.],labels=label)
# df.head()


# In[402]:


# from scipy.stats import chi2_contingency
# p_val=chi2_contingency(pd.crosstab(df['age_range'],df['diabetes']))[1]
# if(p_val<0.05):
#     print('Important feature')
# else:
#     print('Not important feature')


# In[403]:


# df['age_range']=df['age_range'].replace({'low':0,'mid':1,'high':2}).astype(int)


# In[404]:


# sns.kdeplot(x='bmi',hue='diabetes',data=df)
# plt.xticks(range(10,100,10))
# plt.grid(axis='x')
# plt.show()


# In[405]:


# df['bmi_chance']=np.where(df['bmi']>30,1,0)


# In[406]:


# plt.hist(df['bmi'],edgecolor='black',bins=3)[1]


# In[407]:


# label=['low','mid','high']
# df['bmi_range']=pd.cut(df['bmi'],[14       , 40.66666667, 66.33333333, 92.        ],labels=label)
# df.head()


# In[408]:


# p_val=chi2_contingency(pd.crosstab(df['bmi_range'],df['diabetes']))[1]
# if(p_val<0.05):
#     print('Important feature')
# else:
#     print('Not important feature')


# In[409]:


# df['bmi_range']=df['bmi_range'].replace({'low':0,'mid':1,'high':2}).astype(int)


# In[410]:


# df.head()


# In[411]:


# df.columns


# In[412]:


# df['genhlth'].unique()


# In[413]:


# sns.countplot(x='genhlth',hue='diabetes',data=df)


# In[414]:


# df['genhlth_chance']=np.where(df['genhlth']>=3,1,0)


# In[415]:


# df.columns


# In[416]:


# df['menthlth'].unique()


# In[417]:


# sns.countplot(x='menthlth',hue='diabetes',data=df)
# plt.tight_layout()
# plt.xticks(rotation=75)
# plt.show()


# In[418]:


# df['menthlth_chance']=np.where(df['menthlth']>=10,1,0)


# In[419]:


# from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
# X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=['diabetes']),df['diabetes'],test_size=0.2,
#                                               random_state=42,stratify=df['diabetes'])


# In[420]:


# from sklearn.feature_selection import VarianceThreshold
# var=VarianceThreshold(threshold=0.01)


# In[421]:


# var.fit(X_train,y_train)


# In[422]:


# var.get_support()


# In[423]:


# X_train.T.duplicated().sum()
# # No duplicated


# In[424]:


# def multi(df,thres):
#     corr=df.corr(method='spearman')
#     rem_col=set()
#     for i in range(len(corr.columns)):
#         for j in range(i):
#             if(abs(corr.iloc[i,j])>thres):
#                 rem_col.add(corr.columns[i])
#     return rem_col


# In[425]:


# rem_col=multi(X_train,0.85)


# In[426]:


# X_train.drop(columns=rem_col,inplace=True)
# X_test.drop(columns=rem_col,inplace=True)


# In[427]:


# from sklearn.feature_selection import mutual_info_classif
# clss=[]
# for i in X_train.columns:
#     clss.append(mutual_info_classif(X_train[i].values.reshape(-1,1),y_train)[0])


# In[428]:


# mi=pd.Series(clss,index=X_train.columns)
# rem_col=mi[mi==0].index


# In[429]:


# X_train.drop(columns=rem_col,inplace=True)
# X_test.drop(columns=rem_col,inplace=True)


# In[430]:


# X_train.shape


# In[431]:


# check_train=X_train.sample(100)
# check_test=X_test.sample(100)
# check_train['tag']=1
# check_test['tag']=0
# # check=check_train.append(check_test)
# from sklearn.tree import DecisionTreeClassifier
# dt=DecisionTreeClassifier(random_state=42)
# dt.fit(check.drop(columns='tag'),check['tag'])
# # plt.barh(dt.feature_names_in_,dt.feature_importances_)


# In[432]:


# rem_col=['bmi','age']


# In[433]:


# X_train.drop(columns=rem_col,inplace=True)
# X_test.drop(columns=rem_col,inplace=True)


# #### Applying KNN Algorithm first and the metric is recall

# In[434]:


# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier()
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# from sklearn.metrics import recall_score


# In[435]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE
# for i in range(1,X_train.shape[1]+1):
#     rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=i)
#     rfe.fit(X_train,y_train)
#     X_train_rfe=rfe.transform(X_train)
#     X_test_rfe=rfe.transform(X_test)
#     sc.fit(X_train_rfe,y_train)
#     knn.fit(sc.transform(X_train_rfe),y_train)
#     print('The number of features are:',i,' and the recall is=',recall_score(y_test,knn.predict(sc.transform(X_test_rfe))))


# In[436]:


# rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=4)
# rfe.fit(X_train,y_train)


# In[437]:


# X_train_rfe=X_train[X_train.columns[rfe.get_support()]]
# X_test_rfe=X_test[X_train_rfe.columns]
# sc.fit(X_train_rfe,y_train)
# X_train_sc=sc.transform(X_train_rfe)
# X_test_sc=sc.transform(X_test_rfe)


# In[438]:


# train_Score=cross_val_score(knn,X_train_sc,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[439]:


# test_Score=cross_val_score(knn,X_test_sc,y_test,cv=10,scoring='recall')
# # print(test_Score)
# print(test_Score.mean())


# In[440]:


# param_knn={'n_neighbors':[int(x) for x in np.linspace(5,37,25)],
#           'p':[1,2],
#           'weights':['uniform','distance']}


# In[441]:


# grid_knn=GridSearchCV(knn,param_knn,verbose=3,cv=5)


# In[442]:


# grid_knn.fit(X_train_sc,y_train)


# In[443]:


# train_Score=cross_val_score(grid_knn.best_estimator_,X_train_sc,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[444]:


# test_Score=cross_val_score(grid_knn.best_estimator_,X_test_sc,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[445]:


# best_knn=grid_knn.best_params_


# In[446]:


# for i in range(1,X_train.shape[1]+1):
#     rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=i)
#     rfe.fit(X_train,y_train)
#     X_train_rfe=rfe.transform(X_train)
#     X_test_rfe=rfe.transform(X_test)
#     dt.fit(X_train_rfe,y_train)
#     print('The number of features are:',i,' and the recall is=',recall_score(y_test,dt.predict(X_test_rfe)))


# In[447]:


# rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=3)
# rfe.fit(X_train,y_train)
# X_train_rfe=X_train[X_train.columns[rfe.get_support()]]
# X_test_rfe=X_test[X_train_rfe.columns]


# In[448]:


# from sklearn.tree import DecisionTreeClassifier
# dt=DecisionTreeClassifier(random_state=42)
# test_Score=cross_val_score(dt,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())
# print('')
# train_Score=cross_val_score(dt,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[449]:


# param_dt={'ccp_alpha':dt.cost_complexity_pruning_path(X_train_rfe,y_train).ccp_alphas}


# In[450]:


# grid_dt=GridSearchCV(dt,param_dt,cv=5,scoring='recall',verbose=3)


# In[451]:


# grid_dt.fit(X_train_rfe,y_train)


# In[452]:


# train_Score=cross_val_score(grid_dt.best_estimator_,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[453]:


# test_Score=cross_val_score(grid_dt.best_estimator_,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[454]:


# best_dt=grid_dt.best_params_


# In[455]:


# from sklearn.ensemble import BaggingClassifier
# bgc=BaggingClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=best_dt['ccp_alpha']))


# In[456]:


# from sklearn.metrics import roc_auc_score
# for i in range(1,X_train.shape[1]+1):
#     rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=i)
#     rfe.fit(X_train,y_train)
#     X_train_rfe=rfe.transform(X_train)
#     X_test_rfe=rfe.transform(X_test)
#     bgc.fit(X_train_rfe,y_train)
#     print('The number of features are:',i,' and the recall is=',recall_score(y_test,bgc.predict(X_test_rfe)),
#          ' ',roc_auc_score(y_test,bgc.predict(X_test_rfe)))


# In[457]:


# rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=2)
# rfe.fit(X_train,y_train)
# X_train_rfe=X_train[X_train.columns[rfe.get_support()]]
# X_test_rfe=X_test[X_train_rfe.columns]


# In[458]:


# train_Score=cross_val_score(bgc,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[459]:


# test_Score=cross_val_score(bgc,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[460]:


# bgc.fit(X_train_rfe,y_train)


# In[461]:


# rfc=RandomForestClassifier(random_state=42)


# In[462]:


# from sklearn.metrics import roc_auc_score
# for i in range(1,X_train.shape[1]+1):
#     rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=i)
#     rfe.fit(X_train,y_train)
#     X_train_rfe=rfe.transform(X_train)
#     X_test_rfe=rfe.transform(X_test)
#     rfc.fit(X_train_rfe,y_train)
#     print('The number of features are:',i,' and the recall is=',recall_score(y_test,rfc.predict(X_test_rfe)),
#          ' ',roc_auc_score(y_test,rfc.predict(X_test_rfe)))


# In[463]:


# rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=10)
# rfe.fit(X_train,y_train)
# X_train_rfe=X_train[X_train.columns[rfe.get_support()]]
# X_test_rfe=X_test[X_train_rfe.columns]


# In[464]:


# train_Score=cross_val_score(rfc,X_train_rfe,y_train,cv=10,scoring='recall')
# # print(train_Score)
# print(train_Score.mean())


# In[465]:


# test_Score=cross_val_score(rfc,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[466]:


# param_rfc={'n_estimators':[int(x) for x in np.linspace(12,125,5)],
#           'max_depth':[int(x) for x in np.linspace(12,125,5)],
#           'max_samples':np.linspace(0.5,0.75,3),
#           'max_features':np.linspace(0.3,0.45,3)}


# In[467]:


# grid_rfc=GridSearchCV(rfc,param_rfc,cv=5,scoring=
#                      'recall',verbose=3)


# In[468]:


# grid_rfc.fit(X_train_rfe,y_train)


# In[469]:


# train_Score=cross_val_score(grid_rfc.best_estimator_,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[470]:


# test_Score=cross_val_score(grid_rfc.best_estimator_,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[471]:


# best_rfc=grid_rfc.best_params_


# In[472]:


# from sklearn.ensemble import GradientBoostingClassifier
# gbc=GradientBoostingClassifier(random_state=dom_state=42)


# In[473]:


# from sklearn.metrics import roc_auc_score
# for i in range(1,X_train.shape[1]+1):
#     rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=i)
#     rfe.fit(X_train,y_train)
#     X_train_rfe=rfe.transform(X_train)
#     X_test_rfe=rfe.transform(X_test)
#     gbc.fit(X_train_rfe,y_train)
#     print('The number of features are:',i,' and the recall is=',recall_score(y_test,gbc.predict(X_test_rfe)),
#          ' ',roc_auc_score(y_test,gbc.predict(X_test_rfe)))


# In[474]:


# rfe=RFE(RandomForestClassifier(random_state=42),n_features_to_select=5)
# rfe.fit(X_train,y_train)
# X_train_rfe=X_train[X_train.columns[rfe.get_support()]]
# X_test_rfe=X_test[X_train_rfe.columns]


# In[475]:


# test_Score=cross_val_score(gbc,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[476]:


# train_Score=cross_val_score(gbc,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# # print(train_Score.mean())


# In[477]:


# param_gbc={'n_estimators':[int(x) for x in np.linspace(12,125,5)],
#           'max_leaf_nodes':[int(x) for x in np.linspace(8,32,5)],
#           'learning_rate':[0.001,0.01,0.1,1]}


# In[478]:


# grid_gbc=GridSearchCV(gbc,param_gbc,cv=5,scoring='recall',verbose=3)


# In[479]:


# grid_gbc.fit(X_train_rfe,y_train)


# In[480]:


# test_Score=cross_val_score(grid_gbc.best_estimator_,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[481]:


# train_Score=cross_val_score(grid_gbc.best_estimator_,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[482]:


# best_gbc=grid_gbc.best_params_


# In[483]:


# def score(model,X_train,X_test,y_train,y_test):
#     model.fit(X_train,y_train)
#     return recall_score(y_test,model.predict(X_test))


# In[484]:


# from sklearn.ensemble import VotingClassifier
# est=[('rfc',grid_rfc.best_estimator_),('gbc',grid_gbc.best_estimator_)]
# vr=VotingClassifier(est)


# In[485]:


# test_Score=cross_val_score(vr,X_test_rfe,y_test,cv=10,scoring='recall')
# print(test_Score)
# print(test_Score.mean())


# In[486]:


# train_Score=cross_val_score(vr,X_train_rfe,y_train,cv=10,scoring='recall')
# print(train_Score)
# print(train_Score.mean())


# In[487]:


# X_train_rfe.columns


# In[ ]:





# In[488]:


# best_knn


# In[489]:


from sklearn.neighbors import KNeighborsClassifier


# In[490]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[491]:


columns=['genhlth', 'menthlth', 'physhlth', 'highbp']


# In[492]:


out_knn=KNeighborsClassifier(n_neighbors=26,p=1,weights='uniform')


# In[493]:


out_knn.fit(sc.fit_transform(df[columns]),df['diabetes'])


# In[494]:


st.header('Diabetes prediction')


# In[495]:


option=st.sidebar.radio('Navigation',['Home','Prediction'])
if(option=='Home'):
    st.title('In this project I have used Knn,Decision Tree ,Random Forest,Gradient Boosting,Bagging Classifier,Voting Classifier as models to predict.I found Knn was the most accurate model in terms of prediction.I have used recall score as a metric to evaluate the model.')
    st.balloons()
else:
    st.write('Choose the genral health rating 1:Good and 5: Poor')
    gen=st.multiselect('Genral Health',[1.0,2.0,3.0,4.0,5.0],1)
    st.write('Choose the mental health rating ')
    ment=st.number_input('Mental Health',min_value=1,max_value=30,step=1)
    st.write('Choose the physical health rating ')
    phys=st.number_input('Physcical Health',min_value=1,max_value=30,step=1)
    st.write('Do you have high blood pressure')
    high_bp=st.slider('high blood pressure',min_value=0,max_value=1,step=1)
    inp=pd.DataFrame([[gen[0],ment,phys,high_bp]],columns=columns)
    pred=out_knn.predict(sc.transform(inp))
    st.write('The prediction is:')
    st.write(pred)
    st.write('The probabilites matrix is')
    st.write(pd.DataFrame(out_knn.predict_proba(sc.transform(inp)),columns=[False,True]))
    st.ballons()


# In[ ]:





# In[ ]:




