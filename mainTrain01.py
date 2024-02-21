import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



data=np.load('./trainingDataTarget/data.npy')
target=np.load('./trainingDataTarget/target.npy')

# Assuming train_target and test_target are arrays of class labels (0 and 1)
num_classes = 2

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)
train_target_encoded = to_categorical(train_target, num_classes)
test_target_encoded = to_categorical(test_target, num_classes)


print(train_data.shape)

#CNN MODEL

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint




noOfFilters =64
sizeOfFilter1=(3,3)
sizeOfFilter2=(3,3)
sizeOfPool=(2,2)
noOfNode=64


model=Sequential()
model.add((Conv2D(32, sizeOfFilter1,input_shape=data.shape[1:],activation="relu")))
model.add((Conv2D(32, sizeOfFilter1, activation="relu")))
model.add(MaxPooling2D(pool_size=sizeOfPool))

model.add((Conv2D(64, sizeOfFilter2,input_shape=data.shape[1:], activation="relu")))
model.add((Conv2D(64, sizeOfFilter2, activation="relu")))
model.add(MaxPooling2D(pool_size=sizeOfPool))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(noOfNode, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())



#CNN MODEL
checkpoint= ModelCheckpoint('./trainingDataTarget/model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history = model.fit(train_data, train_target_encoded,
                    epochs=20,
                    callbacks=[checkpoint],
                    validation_split=0.2)