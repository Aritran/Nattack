#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


df = pd.read_csv("testno_strings_normalized_train.csv")


# In[30]:


df.head()


# In[31]:


len(df)


# In[32]:


training = df.iloc[:200000]


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(training, test_size=0.2, random_state=42)
x_train = x_train[x_train.notified == 0]
x_train = x_train.drop(['notified'], axis=1)
y_test = x_test['notified']
x_test = x_test.drop(['notified'], axis=1)
x_train = x_train.values
x_test = x_test.values


# In[34]:


from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

encoding_dim = 14
input_dim = x_train.shape[1]

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)


# In[35]:


autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

cp = ModelCheckpoint(filepath="classifiermodel.h5",verbose=0,save_best_only=True)

tb = TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=True)

history = autoencoder.fit(x_train, x_train,epochs=50,batch_size=32,shuffle=True,validation_data=(x_test, x_test),
                    verbose=1,callbacks=[cp, tb]).history


# In[36]:


from sklearn.metrics import (confusion_matrix, recall_score, classification_report, f1_score)


# In[37]:


predicted = autoencoder.predict(x_test)
mse = np.mean(np.power(x_test - predicted, 2), axis=1)
error = pd.DataFrame({'mse': mse,'y_test': y_test})


# In[38]:


threshold = 0.4
y_pred = [1 if e > threshold else 0 for e in error.mse.values]
conf_matrix = confusion_matrix(error.y_test, y_pred)


# In[39]:


csr = classification_report(error.y_test, y_pred)
print(csr)


# In[40]:


print(conf_matrix)


# In[41]:


print('Overall accuracy = ', (conf_matrix[0][0] + conf_matrix[1][1])/(conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[1][0]))


# In[42]:


import seaborn as sns
LABELS = ['Normal', 'Attack']
conf_matrix = confusion_matrix(error.y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




