from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = loadtxt('diabetes.csv',skiprows=1, delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]
model = Sequential()
model.add(Dense(12,input_shape=(8,),activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=150,batch_size=10)

_,accuracy = model.evaluate(X,y)
print('Accuracy: %.2f' % (accuracy*100))
