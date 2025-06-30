import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import math

#ReLu激活函数
def relu(z):
    return np.maximum(0,z)

def de_relu(h):
    h[h<0]=0
    h[h>0]=1.0
    return h

#leakRelu激活函数
def leakyrelu(z):
    z[z<0] = 0.01*z[z<0]
    z[z>0] = z[z>=0]
    return z

def de_leakrelu(h):
    h[h<0]=0.01
    h[h>0]=1.0
    return h

#sigmod激活函数的前向反向传递
def sigmoid(z):
    h = 1.0/(1+np.exp(-z))
    return h

def de_sigmoid(h):
    return h*(1-h)

#tanh激活函数
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def de_tanh(h):
    return 1- h**2

#无激活函数（全1矩阵）前反向传递函数
def no_active(z):
    h = z
    return h

def de_no_active(h):
    return np.ones(h.shape)


#L2损失函数的前向反向函数（均方差和MSE）
def loss_L2(o,lab):
    diff = lab-o
    sqrdiff = diff**2
    return 0.5*np.sum(sqrdiff)

def de_loss_L2(o,lab):
    return o-lab

#softmax交叉熵的前向反向函数
def loss_CE(o,lab):
    p = np.exp(o)/np.sum(np.exp(o),axis = 1,keepdims = True)
    loss_ce = np.sum(-lab*np.log(p))
    return loss_ce

def de_loss_CE(o,lab):
    p = np.exp(o)/np.sum(np.exp(o),axis = 1,keepdims=True)
    return p-lab

#构建网络初始化
def build_net(dim_in,list_num_hidden,list_act_funs,list_de_act_funs):
    layers = []

    for i in range(len(list_num_hidden)):
        layer = {}

        if i == 0:
            layer["w"] = np.random.randn(dim_in,list_num_hidden[i])
        else:
            layer["w"] = np.random.randn(list_num_hidden[i-1],list_num_hidden[i])

        layer["b"] = np.ones([1,list_num_hidden[i]])
        layer["act_fun"] = list_act_funs[i]
        layer["de_act_fun"] = list_de_act_funs[i]
        layers.append(layer)

    return layers

#前向传递函数
def fead_forward(datas,layers):
    input_layers = []

    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs,layer["w"])+layer["b"]
            h = layer["act_fun"](z)
            input_layers.append(inputs)
        else:
            inputs = h
            z = np.dot(inputs,layer["w"])+layer["b"]
            h = layer["act_fun"](z)
            input_layers.append(inputs)
    return input_layers,h

#参数更新
def updata_wb(datas,labs,layers,loss_fun,de_loss_fun,lr = 0.01):
    N,D = np.shape(datas)
    inputs,output = fead_forward(datas,layers)
    loss = loss_fun(output,labs)
    deltas0 = de_loss_fun(output,labs)
    deltas = []
    for i in range(len(layers)):
        index = -i-1
        if i == 0:
            delta = loss*layers[index]["de_act_fun"](output)
        else:
            h = inputs[index+1]
            delta = np.dot(delta,layers[index+1]["w"].T)*layers[index]["de_act_fun"](h)
        deltas.insert(0,delta)
    #利用误差进行修正
    for i in range(len(layers)):
        dw = np.dot(inputs[i].T,deltas[i])
        db = np.sum(deltas[i],axis=0,keepdims=True)
        #梯度下降
        layers[i]["w"] = layers[i]["w"]-lr*dw
        layers[i]["b"] = layers[i]["b"]-lr*db

    return layers,loss


def test_accuracy(datas,labs_true,layers):
    #print(labs_true)
    _,output = fead_forward(datas,layers)
    #print(output)
    lab_det = np.argmax(output,axis=1)
    #print(lab_det)
    labs_true = np.argmax(labs_true,axis=1)
    #print(labs_true)
    N_error = np.where(np.abs(labs_true-lab_det)>0)[0].shape[0]

    error_rate = N_error/np.shape(datas)[0]
    return error_rate

def test_accuracy_on_test(datas,labs_true,layers):
    _,output = fead_forward(datas,layers)
    lab_det = np.argmax(output,axis=1)
    labs_true = np.argmax(labs_true,axis=1)
    N_error = np.where(np.abs(labs_true-lab_det)>0)[0].shape[0]

    num = range(len(labs_true))
    plt.plot(num, np.array(labs_true), 'b', label='test')
    plt.plot(num,np.array(lab_det),'r',label='predict')
    plt.title('Fit')
    plt.legend(loc='lower right')
    plt.show()

    error_rate = N_error/np.shape(datas)[0]
    return error_rate

def load_dataset(N_tr):
    datas = np.array(np.loadtxt(r'D:\Working park\total_version_second.csv',dtype=float,delimiter=',',skiprows=1,usecols=[3,4,5],encoding='utf-8'))
    label = np.array(np.loadtxt(r'D:\Working park\total_version_second.csv',dtype=float,delimiter=',',skiprows=1,usecols=6,encoding='utf-8'))
    #data_train = np.array(np.loadtxt(r'D:\Working park\train.csv',dtype=float,delimiter=',',skiprows=1,usecols=[2,3,4],encoding='utf-8'))
    #label_train = np.loadtxt(r'D:\Working park\train.csv',dtype=float,delimiter=',',skiprows=1,usecols=6,encoding='utf-8')
    #data_test= np.loadtxt(r'D:\Working park\test.csv',dtype=float,delimiter=',',skiprows=1,usecols=[2,3,4],encoding='utf-8')
    #label_test = np.loadtxt(r'D:\Working park\test.csv',dtype=float,delimiter=',',skiprows=1,usecols=6,encoding='utf-8')
    #N_train,D_train = np.shape(data_train)
    #N_test,D_test = np.shape(data_test)
    N,D = np.shape(datas)
    #print("数据集总数量:",N,D)
    N_te = N-N_tr
    unique_labs = np.unique(label).tolist()
    #unique_train_labs = np.unique(label_train).tolist()
    #unique_test_labs = np.unique(label_test).tolist()

    dic_str2index={}
    dic_index2str={}
    for i in range(len(unique_labs)):
        lab_str = unique_labs[i]
        dic_str2index[lab_str] = i
        dic_index2str[i] = lab_str
    labs_onehot = np.zeros([N,len(unique_labs)])
    for i in range(N):
        labs_onehot[i,dic_str2index[label[i]]] = 1

    perm = np.random.permutation(N)
    index_train = perm[:N_tr]
    index_test = perm[N_tr:]

    train_data = datas[index_train,:]
    label_train_onehot = labs_onehot[index_train,:]

    data_test = datas[index_test,:]
    label_test_onehot = labs_onehot[index_test]


    return train_data,label_train_onehot,data_test,label_test_onehot,dic_index2str

'''
    dic_train_str2index = {}
    dic_train_index2str = {}
    dic_test_str2index = {}
    dic_test_index2str = {}
    for i in range(len(unique_train_labs)):
        lab_str = unique_train_labs[i]
        dic_train_str2index[lab_str]=i
        dic_train_index2str[i] = lab_str
    for i in range(len(unique_test_labs)):
        lab_str = unique_test_labs[i]
        dic_test_str2index[lab_str]=i
        dic_test_index2str[i] = lab_str
    lab_train_onehot = np.zeros([N_train,len(unique_train_labs)])
    lab_test_onehot = np.zeros([N_test,len(unique_test_labs)])
    for i in range(N_train):
        lab_train_onehot[i,dic_train_str2index[label_train[i]]]=1
    for i in range(N_test):
        lab_test_onehot[i,dic_test_str2index[label_test[i]]]=1


    return data_train,lab_train_onehot,data_test,lab_test_onehot
'''

if __name__ == "__main__":
    data_train,label_train_onehot,data_test,label_test_onehot,dic_index2str = load_dataset(30000)
    print("训练集数量:", len(label_train_onehot))
    print("测试集数量:", len(label_test_onehot))
    N,dim_in = np.shape(data_train)

    list_num_hidden = [9,9,9]
    list_act_funs = [sigmoid,sigmoid,no_active]
    list_de_act_funs = [de_sigmoid,de_sigmoid,de_no_active]

    loss_fun = loss_CE
    de_loss_fun = de_loss_CE

    layers = build_net(dim_in,list_num_hidden,list_act_funs,list_de_act_funs)

    #开始训练
    error_list = []
    acc_list = []
    loss_list = []
    n_epoch = 300
    batchsize = 8
    N_batch = N//batchsize
    for i in range(n_epoch):
        rand_index = np.random.permutation(N).tolist()
        loss_sum = 0
        for j in range(N_batch):
            index = rand_index[j*batchsize:(j+1)*batchsize]
            batch_datas = data_train[index]
            batch_labs = label_train_onehot[index]
            layers,loss = updata_wb(batch_datas,batch_labs,layers,loss_fun,de_loss_fun,lr=0.1)
            loss_sum = loss_sum + loss

        error = test_accuracy(data_train,label_train_onehot,layers)
        print("epoch %d error %.3f  loss_all %.3f   accuracy: %.3f "%(i,error,loss_sum,(1-error)*100))
        acc_list.append(((1 - error) * 100))
        loss_list.append(loss_sum)
    #print(layers)
    error = test_accuracy_on_test(data_test,label_test_onehot,layers)
    print("验证集上的误差为：",error)
    print("验证集上的准确率为:", (1-error)*100,end=' ')
    print("%")

    epoch = range(len(acc_list))
    plt.plot(epoch,np.array(acc_list),'b',label='accuracy')
    plt.title('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(epoch,np.array(loss_list),'r',label='loss')
    plt.title('loss')
    plt.legend(loc='upper right')
    plt.show()

    while(True):
        print("第一层w:")
        print(layers[0]['w'])
        np.savetxt(r"C:\Users\11374\Desktop\nn\first_flood_w.txt",layers[0]['w'],fmt = '%s',delimiter=',')
        print("第一层b:")
        print(layers[0]['b'])
        np.savetxt(r"C:\Users\11374\Desktop\nn\first_flood_b.txt", layers[0]['b'], fmt='%s', delimiter=',')
        print("第二层w:")
        print(layers[1]['w'])
        np.savetxt(r"C:\Users\11374\Desktop\nn\second_flood_w.txt", layers[1]['w'], fmt='%s', delimiter=',')
        print("第二层b:")
        print(layers[1]['b'])
        np.savetxt(r"C:\Users\11374\Desktop\nn\second_flood_b.txt", layers[1]['b'], fmt='%s', delimiter=',')
        choice = input("是否保存该模型:(Y/N)")
        if choice == 'Y':

            np.save(r"D:\Working park\model1.npy",layers)
            np.savetxt(r"D:\Working park\model1.txt", layers, fmt='%s')

            break
        elif choice == 'N':
            break
        else:
            print("输入错误请重新输入")













