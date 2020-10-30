import pandas as pd
from keras.layers import Input, Dense 
from keras.models import Model, Sequential 
from keras import regularizers 
from sklearn.metrics import mean_squared_error
from statistics import mean, stdev
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

df = pd.read_csv('data_banknote_authentication.txt', names= ['Variance','Skewness','Curtosis', 'Entropy','Class'] , index_col=False)

# Scaling the Data
df = pd.DataFrame(MinMaxScaler().fit_transform(df))
df.columns = ['Variance','Skewness','Curtosis', 'Entropy','Class']

# Printing Scaled data
# print(df)


# Segregating Dataset into training and testing set where training set are notes which are geniune, and testing set are notes that are forged
train = df.loc[df['Class'] == 0]
y_train = train['Class']
x_train = train.drop(['Class'] , axis=1)

test = df.loc[df['Class'] == 1]
y_test = test['Class']
x_test = test.drop(['Class'] , axis=1)

print(df.head())

# Creating Layers for Encoder
input_layer = Input(shape =(x_test.shape[1], ))

encoded = Dense(4, activation ='relu')(input_layer)
encoded = Dense(3, activation ='relu')(encoded) 


# Bottleneck Layer
bottleneck = Dense(2, activation ='relu')(encoded)


# Creating Layer for Decoder
decoded = Dense(3, activation ='relu')(bottleneck) 
decoded = Dense(4, activation ='relu')(decoded)

output_layer = Dense(x_train.shape[1], activation ='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer) 
autoencoder.compile(optimizer ="adadelta", loss ="mse")

autoencoder.fit(x_train,x_train)
predicted = autoencoder.predict(x_train)

df_out = pd.DataFrame(predicted)
df_out.columns = ['Variance','Skewness','Curtosis', 'Entropy']

print(df_out.head())

errors = []


# Getting Errors by Calculating Euclidean Distance betwenn predicted values and original values 
for i in range(0,5):
    errors.append(distance.euclidean(df_out.iloc[i],x_train.iloc[i]))

errors_df = pd.DataFrame(errors)
print(errors_df)

mini = min(errors)
avg = mean(errors)
maxi = max(errors)
std = stdev(errors)

print(errors)
print("Minimum Error : {0}".format(mini))
print("Average Error : {0}".format(avg))
print("Max Error : {0}".format(maxi))
print("Std Deviation : {0}".format(std))


threshold = avg + std

print("Threshold : {0}".format(threshold))

accuracy = 0

# Calculating Euclidean Distance between predicted values of the test set and the original values of the test set
reproduced_test = autoencoder.predict(x_test)
for i in range(0,5):
    if distance.euclidean(reproduced_test[i],x_test.iloc[i]) > threshold:
        print("Anomalous")
        accuracy += 1
    else:
        print("Not Anomalous")  


# Calculating Accuracy as number of forged bank notes correctly identified as anomalous divided by total number of forged bank notes
print("----------------- SUMMARY REPORT ----------------------")
print("Accuracy : {0}".format(accuracy/len(x_test)))
print("Out of {0} forged bank notes, {1} got identified as Anomalous".format(len(x_test),accuracy))