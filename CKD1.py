import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

data = pd.read_csv(r"C:\Users\B Sravani\Desktop\internship project\kidney_disease.csv")

data.drop(["id"],axis=1,inplace=True)

data.columns=['age','blood_pressure','specific_gravity','albumin','sugar','red_blood_cells',
             'pus_cell','pus_cell_clumps','bacteria','blood_glucose_random','blood_urea',
             'serum_creatinine','sodium','potassium','hemoglobin','packed_cell_volume',
             'white_blood_cell_count','red_blood_cell_count','hypertension',
             'diabetesmellitus','coronary_artery_disease','appetite','pedal_edema','anemia','class']

data['class']=data['class'].replace('ckd\t','ckd')

catcols=set(data.dtypes[data.dtypes=='O'].index.values)

for i in catcols:
    print("columns:",i)
    print((data[i]))
    print('*'*120+ '\n')

catcols.remove('red_blood_cell_count')
catcols.remove('white_blood_cell_count')
catcols.remove('packed_cell_volume')

contcols=set(data.dtypes[data.dtypes!='O'].index.values)


for i in contcols:
    print("continous columns:",i)
    print((data[i]))
    print('*'*120+ '\n')

contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')  
print(contcols)

contcols.add('red_blood_cell_count')
contcols.add('white_blood_cell_count')
contcols.add('packed_cell_volume')
print(contcols)

catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')
print(catcols)

data['coronary_artery_disease']=data.coronary_artery_disease.replace('\tno','no')

data['diabetesmellitus']=data.diabetesmellitus.replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})

print(data.isnull().any())

print(data.isnull().sum())

data.packed_cell_volume=pd.to_numeric(data.packed_cell_volume,errors='coerce')
data.white_blood_cell_count=pd.to_numeric(data.white_blood_cell_count,errors='coerce')
data.red_blood_cell_count=pd.to_numeric(data.red_blood_cell_count,errors='coerce')

data['blood_glucose_random'].fillna(data['blood_glucose_random'].mean(),inplace=True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace=True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace=True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace=True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace=True)
data['potassium'].fillna(data['potassium'].mean(),inplace=True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace=True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace=True)
data['sodium'].fillna(data['sodium'].mean(),inplace=True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace=True)

data['age'].fillna(data['age'].mode()[0],inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0],inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0],inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0],inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0],inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0],inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0],inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0],inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0],inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0],inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0],inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0],inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0],inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0],inplace=True)

print(data.isnull().sum())

for i in catcols:
    print("LABEL ENCODIND OF:",i)
    LEi = LabelEncoder()
    print(data[i])
    data[i]=LEi.fit_transform(data[i])
    print(data[i])
    print("*"*100)

selcols=['red_blood_cells','pus_cell','diabetesmellitus','pedal_edema',
         'anemia','coronary_artery_disease','blood_glucose_random','blood_urea']
x=pd.DataFrame(data,columns=selcols)
y=pd.DataFrame(data,columns=['class'])
print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

lgr= LogisticRegression()
lgr.fit(x_train,y_train)

y_pred=lgr.predict(x_test)

y_pred1=lgr.predict([[129,99,1,0,0,1,0,1]])
print(y_pred)
y_pred

print(accuracy_score(y_test,y_pred))

conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)

pickle.dump(lgr, open('CKD.pkl','wb'))