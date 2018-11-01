
"""
Created on Wed Sep 12 14:51:05 2018

@author: utsav
"""

import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FastICA
from sklearn.naive_bayes import GaussianNB


#### Get dataset#1
data = np.genfromtxt("pendigitsdata1.csv", dtype=int, delimiter=',')

data_class = data[:,16]
data_attributes = data[:, :-1]
#########

#Get Testing Data
test_data = np.genfromtxt("pendigitsdata2.csv", dtype=int, delimiter=',')

test_data_class = test_data[:,16]
test_data_attributes = test_data[:, :-1]
########

x_axis = [10,20,30,40,50]
###Start of KNN

time_KNN_1 = []
time_KNN_2 = []
score_train_test_KNN = []
score_test_KNN = []

train_without_KNN = KNeighborsClassifier(n_neighbors=3)
split  = 750
   
time_KNN_1_pca = []
time_KNN_2_pca = []
score_train_test_KNN_pca = []
score_test_KNN_pca = []

pca = PCA(n_components=5)
pca_reduced_data=pca.fit(data_attributes).transform(data_attributes)
pca_reduced_data_test=pca.fit(test_data_attributes).transform(test_data_attributes)

time_KNN_1_ica = []
time_KNN_2_ica = []
score_train_test_KNN_ica = []
score_test_KNN_ica = []

ica = FastICA(n_components=5)
ica_reduced_data = ica.fit(data_attributes).transform(data_attributes)
ica_reduced_data_test = ica.fit(test_data_attributes).transform(test_data_attributes)


for i in range(5):
  if(i!= 4):
    ##Train, test for dataset-1 without data reduction 
    start_1 = time.time()
    train_without_KNN.fit(data_attributes[0:split],data_class[0:split])
    score_train_test_KNN.append(train_without_KNN.score(data_attributes[split:],data_class[split:]) *100)
    end_1 = time.time()
    time_KNN_1.append(end_1-start_1)
    
    ##Tests dataset-2 without data reduction
    start_2 = time.time()
    score_test_KNN.append(train_without_KNN.score(test_data_attributes,test_data_class) *100) 
    end_2 = time.time()
    time_KNN_2.append(end_2-start_2)
    
    ##Train, test for dataset-1 with PCA
    start_1 = time.time()
    train_without_KNN.fit(pca_reduced_data[0:split],data_class[0:split])
    score_train_test_KNN_pca.append(train_without_KNN.score(pca_reduced_data[split:],data_class[split:]) *100)
    end_1 = time.time()
    time_KNN_1_pca.append(end_1-start_1)
    
    ##Test dataset-2 with PCA
    start_2 = time.time()
    score_test_KNN_pca.append(train_without_KNN.score(pca_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_KNN_2_pca.append(end_2-start_2)
    
    ##Train, test for dataset-1 with ICA
    start_1 = time.time()
    train_without_KNN.fit(ica_reduced_data[0:split],data_class[0:split])
    score_train_test_KNN_ica.append(train_without_KNN.score(ica_reduced_data[split:],data_class[split:]) *100)
    end_1 = time.time()
    time_KNN_1_ica.append(end_1-start_1)
    
    ##Test dataset-2 with ICA
    start_2 = time.time()
    score_test_KNN_ica.append(train_without_KNN.score(ica_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_KNN_2_ica.append(end_2-start_2)
    
    split = split + 750

  else:
    ##Train, test for dataset-1 without data reduction
    start_1 = time.time()
    train_without_KNN.fit(data_attributes[0:3747],data_class[0:3747])
    score_train_test_KNN.append(train_without_KNN.score(data_attributes[3747:],data_class[3747:]) *100)
    end_1 = time.time()
    time_KNN_1.append(end_1-start_1)
  
  
 
    ##Test dataset-2 without data reduction
    start_2 = time.time()
    score_test_KNN.append(train_without_KNN.score(test_data_attributes,test_data_class) *100) 
    end_2 = time.time()
    time_KNN_2.append(end_2-start_2)
    
    ##Train, test for dataset-1 with PCA
    start_1 = time.time()
    train_without_KNN.fit(pca_reduced_data[0:3747],data_class[0:3747])
    score_train_test_KNN_pca.append(train_without_KNN.score(pca_reduced_data[3747:],data_class[3747:]) *100)
    end_1 = time.time()
    time_KNN_1_pca.append(end_1-start_1)
    
    ##Test dataset-2 with data reduction PCA
    start_2 = time.time()
    score_test_KNN_pca.append(train_without_KNN.score(pca_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_KNN_2_pca.append(end_2-start_2)
    
    ##Train, test for dataset-1 with ICA
    start_1 = time.time()
    train_without_KNN.fit(ica_reduced_data[0:3747],data_class[0:3747])
    score_train_test_KNN_ica.append(train_without_KNN.score(ica_reduced_data[3747:],data_class[3747:]) *100)
    end_1 = time.time()
    time_KNN_1_ica.append(end_1-start_1)
    
    ##Test dataset-2 with data reduction ICA
    start_2 = time.time()
    score_test_KNN_ica.append(train_without_KNN.score(ica_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_KNN_2_ica.append(end_2-start_2)
      
###End of KNN    

###Start of GNB
    
score_train_test_GNB = []
score_test_GNB = []
time_GNB_1 = []
time_GNB_2 = []


train_without_GNB = GaussianNB()
split = 750

score_train_test_GNB_pca = []
score_test_GNB_pca = []
time_GNB_1_pca = []
time_GNB_2_pca = []

score_train_test_GNB_ica = []
score_test_GNB_ica = []
time_GNB_1_ica = []
time_GNB_2_ica = []


for i in range(5):
  if(i!= 4):
    
    ##Train, test for dataset-1 without data reduction 
    start_1 = time.time()
    train_without_GNB.fit(data_attributes[0:split],data_class[0:split])
    score_train_test_GNB.append(train_without_GNB.score(data_attributes[split:],data_class[split:]) *100)
    end_1 = time.time()
    time_GNB_1.append(end_1-start_1)

    ##Tests dataset-2 without data reduction    
    start_2 = time.time()
    score_test_GNB.append(train_without_GNB.score(test_data_attributes,test_data_class) *100) 
    end_2 = time.time()
    time_GNB_2.append(end_2-start_2)
    
    ##Train, test for dataset-1 with PCA
    start_1 = time.time()
    train_without_GNB.fit(pca_reduced_data[0:split],data_class[0:split])
    score_train_test_GNB_pca.append(train_without_GNB.score(pca_reduced_data[split:],data_class[split:]) *100)
    end_1 = time.time()
    time_GNB_1_pca.append(end_1-start_1)
    
    
    ##Test dataset-2 with PCA
    start_2 = time.time()
    score_test_GNB_pca.append(train_without_GNB.score(pca_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_GNB_2_pca.append(end_2-start_2)    
    
    ##Train, test for dataset-1 with ICA
    start_1 = time.time()
    train_without_GNB.fit(ica_reduced_data[0:split],data_class[0:split])
    score_train_test_GNB_ica.append(train_without_GNB.score(ica_reduced_data[split:],data_class[split:]) *100)
    end_1 = time.time()
    time_GNB_1_ica.append(end_1-start_1)
    
    
    ##Test dataset-2 with ICA
    start_2 = time.time()
    score_test_GNB_ica.append(train_without_GNB.score(ica_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_GNB_2_ica.append(end_2-start_2)    
    
    split = split + 750
    
  else:
    ##Train, test for dataset-1 without data reduction
    start_1 = time.time()
    train_without_GNB.fit(data_attributes[0:3747],data_class[0:3747])
    score_train_test_GNB.append(train_without_GNB.score(data_attributes[3747:],data_class[3747:]) *100)
    end_1 = time.time()
    time_GNB_1.append(end_1-start_1)
    
    ##Test dataset-2 without data reduction
    start_2 = time.time()
    score_test_GNB.append(train_without_GNB.score(test_data_attributes,test_data_class) *100) 
    end_2 = time.time()
    time_GNB_2.append(end_2-start_2)
    
    ##Train, test for dataset-1 with PCA
    start_1 = time.time()
    train_without_GNB.fit(pca_reduced_data[0:3747],data_class[0:3747])
    score_train_test_GNB_pca.append(train_without_GNB.score(pca_reduced_data[3747:],data_class[3747:]) *100)
    end_1 = time.time()
    time_GNB_1_pca.append(end_1-start_1)
    
    ##Test dataset-2 with data reduction PCA    
    start_2 = time.time()
    score_test_GNB_pca.append(train_without_GNB.score(pca_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_GNB_2_pca.append(end_2-start_2)
    
    ##Train, test for dataset-1 with ICA
    start_1 = time.time()
    train_without_GNB.fit(ica_reduced_data[0:3747],data_class[0:3747])
    score_train_test_GNB_ica.append(train_without_GNB.score(ica_reduced_data[3747:],data_class[3747:]) *100)
    end_1 = time.time()
    time_GNB_1_ica.append(end_1-start_1)
    
    ##Test dataset-2 with data reduction ICA    
    start_2 = time.time()
    score_test_GNB_ica.append(train_without_GNB.score(ica_reduced_data_test,test_data_class) *100) 
    end_2 = time.time()
    time_GNB_2_ica.append(end_2-start_2)

###End of GNB

###Timing plot for without data reduction
plt.title("Timing without data reduction")
plt.plot(x_axis,time_KNN_2,label='KNN Dataset-2')
plt.scatter(x_axis,time_KNN_2)
plt.plot(x_axis,time_KNN_1, color='red',label='KNN Dataset-1')
plt.scatter(x_axis,time_KNN_1, color='red')
plt.plot(x_axis,time_GNB_2, color='yellow',label='GNB Dataset-2')
plt.scatter(x_axis,time_GNB_2,color='yellow')
plt.plot(x_axis,time_GNB_1, color='black',label='GNB Dataset-1')
plt.scatter(x_axis,time_GNB_1,color='black')
plt.legend()
plt.xlabel("% of training data")
plt.ylabel("Time")
plt.show() 

###Accuracy plot for without data reduction
plt.title("Accuracy without data reduction")
plt.plot(x_axis,score_test_KNN,label='KNN Dataset-2')
plt.scatter(x_axis,score_test_KNN)
plt.plot(x_axis,score_train_test_KNN, color='red',label='KNN Dataset-1')
plt.scatter(x_axis,score_train_test_KNN, color='red')
plt.plot(x_axis,score_test_GNB, color='yellow',label='GNB Dataset-2')
plt.scatter(x_axis,score_test_GNB,color='yellow')
plt.plot(x_axis,score_train_test_GNB, color='black',label='GNB Dataset-1')
plt.scatter(x_axis,score_train_test_GNB,color='black')
plt.legend()
plt.xlabel("% of training data")
plt.ylabel("Accuracy")
plt.show() 

###Accuracy plot with data reduction using PCA
plt.title("Accuracy using PCA")
plt.plot(x_axis,score_test_KNN_pca,label='KNN Dataset-2')
plt.scatter(x_axis,score_test_KNN_pca)
plt.plot(x_axis,score_train_test_KNN_pca, color='red',label='KNN Dataset-1')
plt.scatter(x_axis,score_train_test_KNN_pca, color='red')
plt.plot(x_axis,score_test_GNB_pca, color='yellow',label='GNB Dataset-2')
plt.scatter(x_axis,score_test_GNB_pca, color='yellow')
plt.plot(x_axis,score_train_test_GNB_pca, color='black',label='GNB Dataset-1')
plt.scatter(x_axis,score_train_test_GNB_pca, color='black')
plt.legend()
plt.xlabel("% of training data")
plt.ylabel("Accuracy")
plt.show() 


###Timing plot with data reduction using PCA
plt.title("Time using PCA")
plt.plot(x_axis,time_KNN_2_pca,label='KNN Dataset-2')
plt.scatter(x_axis,time_KNN_2_pca)
plt.plot(x_axis,time_KNN_1_pca, color='red',label='KNN Dataset-1')
plt.scatter(x_axis,time_KNN_1_pca, color='red')
plt.plot(x_axis,time_GNB_2_pca, color='yellow',label='GNB Dataset-2')
plt.scatter(x_axis,time_GNB_2_pca, color='yellow')
plt.plot(x_axis,time_GNB_1_pca, color='black',label='GNB Dataset-1')
plt.scatter(x_axis,time_GNB_1_pca, color='black')
plt.legend()
plt.xlabel("% of training data")
plt.ylabel("Time")
plt.show() 


###Accuracy plot with data reduction using ICA
plt.title("Accuracy using ICA")
plt.plot(x_axis,score_test_KNN_ica,label='KNN Dataset-2')
plt.scatter(x_axis,score_test_KNN_ica)
plt.plot(x_axis,score_train_test_KNN_ica, color='red',label='KNN Dataset-1')
plt.scatter(x_axis,score_train_test_KNN_ica, color='red')
plt.plot(x_axis,score_test_GNB_ica, color='yellow',label='GNB Dataset-2')
plt.scatter(x_axis,score_test_GNB_ica, color='yellow')
plt.plot(x_axis,score_train_test_GNB_ica, color='black',label='GNB Dataset-1')
plt.scatter(x_axis,score_train_test_GNB_ica, color='black')
plt.legend()
plt.xlabel("% of training data")
plt.ylabel("Accuracy")
plt.show() 

###Timing plot with data reduction using ICA
plt.title("Time using ICA")
plt.plot(x_axis,time_KNN_2_ica,label='KNN Dataset-2')
plt.scatter(x_axis,time_KNN_2_ica)
plt.plot(x_axis,time_KNN_1_ica, color='red',label='KNN Dataset-1')
plt.scatter(x_axis,time_KNN_1_ica, color='red')
plt.plot(x_axis,time_GNB_2_ica, color='yellow',label='GNB Dataset-2')
plt.scatter(x_axis,time_GNB_2_ica, color='yellow')
plt.plot(x_axis,time_GNB_1_ica, color='black',label='GNB Dataset-1')
plt.scatter(x_axis,time_GNB_1_ica, color='black')
plt.legend()
plt.xlabel("% of training data")
plt.ylabel("Time")
plt.show() 

###First 2 components of PCA
plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red', 'black', 'yellow', 'blue', 'pink', 'green', 'lightblue']
lw = 2

for color, i in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    plt.scatter(pca_reduced_data[data_class == i, 0], pca_reduced_data[data_class == i, 1], color=color, alpha=.8, lw=lw)
plt.title("2-components of PCA")  
plt.axis("off")  
plt.show()

###First 2 components of ICA
plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red', 'black', 'yellow', 'blue', 'pink', 'green', 'lightblue']
lw = 2

for color, i in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    plt.scatter(ica_reduced_data[data_class == i, 0], ica_reduced_data[data_class == i, 1], color=color, alpha=.8, lw=lw)
plt.title("2-components of ICA")
plt.axis("off")
plt.show()
