#step1:import lib
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


#2.read data
dataset=pd.read_csv(r"E:\train.csv\train.csv")
dataset
from sklearn.utils import resample
df_1=dataset[dataset['toxic']==1]


df_0=(dataset[dataset['toxic']==0]).sample(n=2000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=2000,random_state=123)


dataset=pd.concat([df_0,df_1_upsample])
dataset.reset_index(inplace = True)




dataset.describe()

dataset.cov()


dataset.corr()

dataset.isnull().any()

import re#regular Expressions
import nltk#Natural lang Tool Kit
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

data=[]

for i in range(0,4000):
    #preprocessing techniques
    #step 4: replace regular expressions,!@#etc
    comment_text=re.sub('[^a-zA-Z]',' ',dataset['comment_text'][i])
    #step 5:Convert the sentence into lower case
    comment_text=comment_text.lower()
    #step 6:split the sentence to list 
    comment_text=comment_text.split()
    #step 7:remove the stop words and stem the words
    #stop words: are there is here where it this that
    comment_text=[ps.stem(word) for word in comment_text if not word in set(stopwords.words('english'))]
    #step 8:8.join the words in the list
    comment_text= ' '.join(comment_text)
    data.append(comment_text)

#step 9.vectorize the words
#vectorizer it is nothing butlike one hot encoding
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)

#step10.split data into x and y
x = cv.fit_transform(data).toarray()
with open('CountVectorizer','wb') as file:
    pickle.dump(cv,file)

y=dataset.iloc[:,3:4].values

#step 11:split data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

#step 12.ANN building 
#importlib
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(activation="relu",init="uniform",units=1000))#input layer
model.add(Dense(units=6000,activation="relu",init="uniform"))#hidden layer
model.add(Dense(units=4000,activation="relu",init="uniform"))
model.add(Dense(units=3000,activation="relu",init="uniform"))
model.add(Dense(units=1,activation="sigmoid",init="uniform"))
 
#configure learning process
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#train model
model.fit(x_train,y_train,epochs=3,batch_size=256)

model.save('comment_text1.h5')

#step 13: prediction
y_pred=model.predict(x_test)
y_pred=y_pred>0.5


x_intent="good"
x_intent=cv.transform([x_intent])
y_pred=model.predict(x_intent)
y_pre=(y_pred>0.6)
print(y_pre)