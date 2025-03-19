# 分布式恢复， 每个缺失的节点构成一个ego network， 并在子图上执行GT-NET

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import Input, Model
from keras import backend as K
from keras.activations import relu
from keras.models import clone_model
from keras.layers import Dense, Conv2D, subtract, multiply, Lambda, add, Concatenate, Flatten, Reshape
import numpy as np
import tensorflow as tf
import scipy.io as sio
import h5py

trainFile = './trainData/pems08_volume_train.mat'
testFile = './testData/pems08_volume_test.mat'
ConnectionFile= './pems08_connected_nodes.mat'
Lfile='./Lsym_PeMS08.mat'
connected_nodes = sio.loadmat(ConnectionFile)
connection = connected_nodes['idx']
Ldata = sio.loadmat(Lfile)
Lsym = Ldata['Lsym']


def transform (r):
    N=4
    # t0 = Conv2D(N, (1, 1), padding='same')(r)
    a=Flatten()(r)
    a1=Dense(a.shape[1])(a)
    b=Reshape((12,24,1))(a1)
    t1 = Conv2D(N, (3, 3), padding='same', activation='relu')(b)
    t3 = Conv2D(1, (3, 3), padding='same')(t1)
    return t3
# def invtransform (r):
#     N=2
#     o2 = Conv2D(N, (3, 3), padding='same')(r)
#     s2 = Conv2D(N, (3, 3), padding='same', activation='relu')(o2)
#     t2 = Conv2D(N, (3, 3), padding='same')(s2)
#     f2 = Conv2D(1, (3, 3), padding='same')(t2)
#     return f2
# Training Data Loading
def load_train_data(mat73=False):
    if mat73 == True:
        trainData = h5py.File(trainFile)
        trainLabel = np.transpose(trainData['sub_data_train'], [3, 2, 1, 0])
    else:
        trainData = sio.loadmat(trainFile)
        trainLabel = trainData['sub_data_train']


    if mat73 == True:
        valData = h5py.File(testFile)
        valLabel = np.transpose(valData['sub_data_test'], [3, 2, 1, 0])
    else:
        valData = sio.loadmat(testFile)
        valLabel = valData['sub_data_test']

    print(np.shape(trainLabel))

    del trainData
    del valData
    return trainLabel, valLabel

trainLabel, testLabel = load_train_data(mat73=False)
##接下来构造0-1张量

L= trainLabel.shape[3]
number = 40

for i in range(number):   #随机次数
    trainPhi = np.ones(trainLabel.shape)
    missing_rate=0.1*np.random.randint(1,10)
    num_missing = round(missing_rate * L)  # missing rate
    # indexvex_0_1=np.ones([trainLabel.shape[0], L])
    index = np.arange(L, dtype=int)
    np.random.shuffle(index)
    missing_index = (index[:num_missing])
    for index_x in missing_index:
        trainPhi[:, :, :, index_x] = 0
        # indexvex_0_1[:, index_x]=0
    trainPhi = trainPhi.astype('float32')
    # indexvex_0_1 = indexvex_0_1.astype('float32')


    trainInput = np.multiply(trainPhi, trainLabel)  # 得到带缺失的张量
    if i==0:
        trainInput1=trainInput
        trainLabel1=trainLabel
        trainPhi1=trainPhi
    else:
        trainInput1=np.concatenate([trainInput1,trainInput], axis=0)
        trainLabel1=np.concatenate([trainLabel1,trainLabel], axis=0)
        trainPhi1=np.concatenate([trainPhi1,trainPhi], axis=0)

# PhiC=1-trainPhi1


######## generate testing data
repeattimes=30
for i in range(9):
    testmissing_rate=0.1*(i+1) #缺失率
    testnum_missing = round(testmissing_rate * L)  # missing rate
    for j in range(repeattimes):
        index1 = np.arange(L, dtype=int)
        testPhi = np.ones(testLabel.shape)
        # testindexvex_0_1 = np.ones([testLabel.shape[0], L])
        np.random.shuffle(index1)
        missing_index = (index1[:testnum_missing])
        for index_x in missing_index:
            testPhi[:, :, :, index_x] = 0
            # testindexvex_0_1[:, index_x] = 0
        testPhi = testPhi.astype('float32')
        # testindexvex_0_1 = testindexvex_0_1.astype('float32')
        testInput = np.multiply(testPhi, testLabel)  # 得到带缺失的张量
        if i == 0 and j==0:
            testInput1 = testInput
            testLabel1 = testLabel
            testPhi1 = testPhi
        else:
            testInput1 = np.concatenate([testInput1, testInput], axis=0)
            testLabel1 = np.concatenate([testLabel1, testLabel], axis=0)
            testPhi1 = np.concatenate([testPhi1, testPhi], axis=0)
testPhiC=1-testPhi1

#定义训练输入输出
Input_Phi=Input(shape=(12,24,L), name='input_2')
# Input_PhiC=Input(shape=(12,24,L), name='input_2')
InputData=Input(shape=(12,24,L), name='input_1')

L = InputData.shape[3]
tempvalue=np.ones([1,1,1,L])
vrho=tempvalue*0.1
vsoftThr=tempvalue*0.01

rho = K.variable(value=vrho, dtype=tf.float32)
softThr = K.variable(value=vsoftThr, dtype=tf.float32)
t = K.variable(value=tempvalue, dtype=tf.float32)
z = K.constant(np.zeros([1, 12, 24, L]), dtype=tf.float32)
r = subtract([z, rho * multiply([Input_Phi, (subtract([z, InputData]))])])

List_slices=[]
List_transformslice=[]
for j in range(L):
    List_slices.append(      K.expand_dims(   r[:,:,:,j], axis=-1)       )

for i in range(L):
    nei_nodes = connection[i, :]
    a = nei_nodes[0].shape[0]
    List1 = [Lsym[i,i]*List_slices[i]]
    for j in range(a):
        b = int(nei_nodes[0][j][0]) - 1
        List1.append(Lsym[i,b]*List_slices[b])
    List_transformslice.append(transform(add(List1)))

for k in range(L):
    List_transformslice[k] = multiply([K.sign(List_transformslice[k]), relu(K.abs(List_transformslice[k]) - softThr[:,:,:,k])])
    # List_transformslice[k]=invtransform(List_transformslice[k])

List_invtransformslice=[]
for i in range(L):
    nei_nodes = connection[i, :]
    a = nei_nodes[0].shape[0]
    List1 = [Lsym[i,i]*List_transformslice[i]]
    for j in range(a):
        b = int(nei_nodes[0][j][0]) - 1
        List1.append(Lsym[i,b]*List_transformslice[b])
    List_invtransformslice.append(transform(add(List1)))
Output = Concatenate(axis=-1)(List_invtransformslice)

# Output=add([Output0,r])

multi_phase_model = Model(inputs=[InputData, Input_Phi], outputs=Output)

multi_phase_model.compile(optimizer='adam',
              loss={'concatenate': 'mean_squared_error'},
              loss_weights={'concatenate': 1})
multi_phase_model.fit({'input_1': trainInput1,  'input_2': trainPhi1},{'concatenate': trainLabel1}, epochs=20, batch_size=1, validation_split=0.1)

test_y = multi_phase_model.predict(x=[testInput1,testPhi1])

def Myrse(img11, img22, testPhiC):
    myrse=[]
    num=img11.shape[0]
    D1=img11-img22
    D=np.multiply(D1,testPhiC)
    for j in range(num):
        img1 = D[j, :, :, :]
        img2 = img22[j, :, :, :]
        img1.astype(np.float32)
        img2.astype(np.float32)
        mse = np.sqrt(np.sum(img1 ** 2))
        mse2 = np.sqrt(np.sum(img2 ** 2))
        rse = mse / mse2
        myrse.append(rse)
    return myrse


# rse = Myrse(test_y, testLabel1, testPhiC)
# rt=repeattimes*testLabel.shape[0]
# meanrse=[]
# for k in range(9):
#     meanrse.append(np.mean(rse[k*rt:k*rt+rt]))
#
# print(meanrse)
def Mymae(img11, img22, testPhiC):
    mymae=[]
    num=img11.shape[0]
    D1=img11-img22
    D=np.multiply(D1,testPhiC)
    for j in range(num):
        img1 = D[j, :, :, :]
        img1.astype(np.float32)

        temp1=testPhiC[j, :, :, :]
        num=np.count_nonzero(temp1)
        mae=np.sum(np.abs(img1))/num
        mymae.append(mae)
    return mymae

rse = Myrse(test_y, testLabel1, testPhiC)
mae = Mymae(test_y, testLabel1, testPhiC)

rt=repeattimes*testLabel.shape[0]
meanrse=[]
meanmae=[]
for k in range(9):
    meanrse.append(np.mean(rse[k*rt:k*rt+rt]))
    meanmae.append(np.mean(mae[k * rt:k * rt + rt]))


print('RSE:')
print(meanrse)
print('MAE:')
print(meanmae)

