import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import array as arr

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_dash(x):
    return 1-np.multiply(np.tanh(x),np.tanh(x))

def sigmoid(x):                                    
    return 1.0/(1.0+np.exp(-x))
    
def sigmoid_dash(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0,keepdims=True)

def ReLU(x):
    return np.maximum(0,x)

def ReLU_dash(x):
    return np.heaviside(x,0)
    

def cross_entropy_cost(a,y,lambd,WtMatrices):
    regularisation_cost=0
    for x in range (0,len(WtMatrices)):
        regularisation_cost=lambd*(np.sum(np.square(WtMatrices[x])))/2
    cost = -1*(np.sum(np.multiply(y,np.log(a)),keepdims=True))+regularisation_cost
    cost = np.squeeze(cost)
    return cost/y.shape[1]

def mean_square_cost(a,y,lamd,WtMatrices):
    regularisation_cost=0
    for x in range (0,len(WtMatrices)):
        regularisation_cost=lambd*(np.sum(np.square(WtMatrices[x])))/2
    cost=np.sum(np.square(y-a))+regularisation_cost
    cost = np.squeeze(cost)
    return cost/y.shape[1]

def accuracy(a, y):
    a1 = np.argmax(a, axis=0)
    y1 = np.argmax(y, axis=0)
    return np.sum(np.asarray((a1==y1), dtype=np.int))/y.shape[1]

def forwardProp(WtMatrices,BiasVectors,NumLayers,Input):
    Z=[]                                                         #List for storing Net vectors of different layers: [0] is Net vector of layer1
    A=[]                                                         #List for storing Activation vectors of different layers: [0] is Activation of layer1
    Net=np.dot(WtMatrices[0],Input)+BiasVectors[0]
    Z.insert(0,Net)
    V=ReLU(Net)
    A.insert(0,V)
    for x in range(1,NumLayers):
        if x== NumLayers-1:
            Net=np.dot(WtMatrices[x],V)+BiasVectors[x]
            Z.insert(x,Net)
            V=softmax(Net)                                      #Using SOFTMAX in the last layer
            A.insert(x,V)
        else:
            Net=np.dot(WtMatrices[x],V)+BiasVectors[x]
            Z.insert(x,Net)
            V=ReLU(Net)
            A.insert(x,V)
    return A,Z

def backProp(WtMatrices,BiasVectors,NumLayers,Z,A,Input,Label,learning_rate,lambd):
    m = Input.shape[1]                                          #Number of training examples sent in = Batch size
    dZ=A[NumLayers-1]- Label
    dW=(1/m)*np.dot(dZ,A[NumLayers-2].transpose())+(lambd*WtMatrices[NumLayers-1])/2
    dB=(1/m)*np.sum(dZ,axis=1,keepdims=True) 
    dA=np.dot(WtMatrices[NumLayers-1].transpose(),dZ)
    
    
    WtMatrices[NumLayers-1]=WtMatrices[NumLayers-1]-learning_rate*dW
    BiasVectors[NumLayers-1]=BiasVectors[NumLayers-1]-learning_rate*dB
    
    for x in range(NumLayers-2,0,-1):
        dZ=np.multiply(dA,ReLU_dash(Z[x]))
        dW=(1/m)*np.dot(dZ,A[x-1].transpose())+(lambd*WtMatrices[x])/2
        dB=(1/m)*np.sum(dZ,axis=1,keepdims=True)
        dA=np.dot(WtMatrices[x].transpose(),dZ)
            
        WtMatrices[x]=WtMatrices[x]-learning_rate*dW
        BiasVectors[x]=BiasVectors[x]-learning_rate*dB    
        
    dZ=np.multiply(dA,ReLU_dash(Z[0]))
    dW=(1/m)*np.dot(dZ,Input.transpose())+(lambd*WtMatrices[0])/2
    dB=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    
    WtMatrices[0]=WtMatrices[0]-learning_rate*dW
    BiasVectors[0]=BiasVectors[0]-learning_rate*dB


BatchSize=6500  #MUST BE A FACTOR OF THE NUMBER OF TRAINING DATA POINTS
NumOfBatches=int(6500/6500) #UPDATE THIS TOO
data=pd.read_csv('mnist_train.csv',header=None)     #Importing csv file   data=pd.read_csv('train.csv', header=None)
numpydata=data.values                               #Converting it to numpy array
train_label=[]                                      #List of training labels for different batches
train_data=[]                                       #List of training data for different batches

for x in range(0,NumOfBatches):                     #Creting batches of the given size
    train_label.insert(x,numpydata[x*BatchSize:(x+1)*BatchSize,:1]) 
    train_data.insert(x,numpydata[x*BatchSize:(x+1)*BatchSize,1:])
    
TRAIN_LABEL=numpydata[:6500,:1]                     #This is the entire training set which I have used to calculate accuracy and cost at the end of each EPOCH                  
TRAIN_DATA=numpydata[:6500,1:]

#test=pd.read_csv('mnist_test.csv',header=None)
#numpytest=test.values

test_label=numpydata[6500:,:1]                      #Here I have taken 500 data point for cross validation
test_data=numpydata[6500:,1:] 

NumLayers=3                                         #ENTER NUMBER OF LAYERS(HIDDEN LAYERS+1)
NumNeurons=[100,100,10]                             #ENTER THE DISTRIBUTION OF NEURONS IN THE LAYERS(ALWAYS KEEP LAST ENTRY AS 10)

WtMatrices=[]                                       #Declare a list to store Weight Matrices: [0] is wt matrix of layer 1
BiasVectors=[]                                      #Declare a list to store Bias Vectors: [0] is the bias vector of LAYER 1
InitFactor=0.01                                     #Standard deviation of the Normal Distribution with mean=0 used for initialization of WT matices
lambd=0.1                                           #Regularisation Parameter
for x in range(0,NumLayers):
    if x==0:
        WtMatrices.insert(x,np.random.randn(NumNeurons[x],784)*InitFactor)      #Multiplying with Initialisation Factor to ensure weights are close to 0
        BiasVectors.insert(x,np.zeros((NumNeurons[x],1)))                       #Bias Vectors are initialized to zero                     
    else:
        WtMatrices.insert(x,np.random.randn(NumNeurons[x],NumNeurons[x-1])*InitFactor)
        BiasVectors.insert(x,np.zeros((NumNeurons[x],1)))
        

ONE_HOT = np.eye(10)[TRAIN_LABEL[:, 0]].transpose() #One hot vector for the entire training data set


xAxis=arr.array('i',[])                             #Arrays for plotting graphs
yAxis=arr.array('f',[])

for y in range(0,250):                              #ENTER NUMBER OF EPOCHS HERE
    for z in range(0,NumOfBatches):
        one_hot = np.eye(10)[train_label[z][:, 0]].transpose()
        
        A,Z = forwardProp( WtMatrices, BiasVectors, NumLayers, train_data[z].T)
    
        learning_rate=0.0035                          #Learning Rate
    
        backProp(WtMatrices,BiasVectors,NumLayers,Z,A,train_data[z].T, one_hot,learning_rate,lambd)
    
    Atotal,Ztotal=forwardProp( WtMatrices, BiasVectors, NumLayers, TRAIN_DATA.T)
    xAxis.append(y)
    yAxis.append(cross_entropy_cost(Atotal[NumLayers-1], ONE_HOT,lambd,WtMatrices))
    
    print("Cost:", cross_entropy_cost(Atotal[NumLayers-1], ONE_HOT,lambd,WtMatrices))
    print("Accuracy:", accuracy(Atotal[NumLayers-1], ONE_HOT))
    

TestLabel = np.eye(10)[test_label[:, 0]].transpose()
TestA,TestZ = forwardProp( WtMatrices, BiasVectors, NumLayers, test_data.T)
print("Accuracy of Test Data:", accuracy(TestA[NumLayers-1], TestLabel))

plt.plot(xAxis,yAxis)                               #Plotting Graphs here
plt.title('[160:80:40:10];ReLU;softmax;CrossEntropyCost')
#plt.text(150, 2, r'lr=0.01, lambd=0.8')
#plt.text(150,1.7, r'iter=250, Bsize=full')
#plt.text(150,1.4, r'TrainAcc=0.94')
#plt.text(150,1.1, r'TestAcc=0.938')
plt.ylabel('Cross Entropy Cost')
plt.xlabel('No. of Iterations')
#plt.savefig('varying LAYERS 2.png')



#Final = np.argmax(TestA[2], axis=0)
#np.savetxt('data4.csv', Final, fmt="%i", delimiter=",")


