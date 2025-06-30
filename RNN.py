import copy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) #随机种子，固定的话每个人得到的结果都一致

# sigmoid函数
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# sigmoid导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# 训练数据生成
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim) #2的8次方，共256个数
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
# unpackbits函数可以把整数转化成2进制数
for i in range(largest_number):
    int2binary[i] = binary[i] #形成一个映射，将0-255的数映射到对应的二进制

# 初始化一些变量
alpha = 0.1 #学习率
input_dim = 2   #输入的大小，因为是2个数相加，所以是2
hidden_dim = 16  #隐含层的大小，代表记忆的维度，这里可以设置任意值
output_dim = 1  #输出层的大小，输出一个结果，所以是1
EPOCHS = 10000 #迭代次数，即共训练多少次
Eval = 100 #每迭代多少次验证一次

# 随机初始化权重
w_i = 2 * np.random.random((hidden_dim, input_dim)) - 1   #输入的权值矩阵，维度为(16, 2)
w_o = 2 * np.random.random((output_dim, hidden_dim)) - 1  #输出的权值矩阵，维度为(1, 16)
w_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1  #循环rnn的关键，隐藏层与隐藏层之间的权值矩阵，维度为(16, 16)

w_i_update = np.zeros_like(w_i) #w_i的梯度，维度(16, 2)
w_o_update = np.zeros_like(w_o) #w_o的梯度，维度(1, 16)
w_h_update = np.zeros_like(w_h) #w_h的梯度，维度(16, 16)

# 开始训练
error_num_list = []
for j in range(EPOCHS):
    #每次随机生成两个128以内的数进行相加（防止溢出），并将相加的结果作为标签
    # 二进制相加
    a_int = np.random.randint(largest_number / 2)  # 随机生成相加的数
    a = int2binary[a_int]  # 映射成二进制值

    b_int = np.random.randint(largest_number / 2)  # 随机生成相加的数
    b = int2binary[b_int]  # 映射成二进制值

    # 真实的答案
    label_int = a_int + b_int   #结果
    label = int2binary[label_int]   #映射成二进制值

    # 待存放预测值，这里我们要输出8位二进制，所以维度是8，即rnn输出8次
    prediction = np.zeros_like(label)

    overallError = 0 # rnn输出的8个值错了几个

    layer_2_deltas = list() #输出层的误差
    layer_2_values = list() #第二层的值（输出的结果）
    layer_1_values = list() #第一层的值（隐含状态）
    layer_1_values.append(copy.deepcopy(np.zeros((hidden_dim, 1)))) #第一个隐含状态需要0作为它的上一个隐含状态

    #前向传播
    for i in range(binary_dim):
        X = np.array([[a[binary_dim - i - 1], b[binary_dim - i - 1]]]).T    #将两个输入并起来变为矩阵，维度(2,1)
        y = np.array([[label[binary_dim - i - 1]]]).T   #将y也变为矩阵，维度(1,1)
        layer_1 = sigmoid(np.dot(w_h, layer_1_values[-1]) + np.dot(w_i, X)) #先算第一层，算到ah，维度(1,1)
        layer_1_values.append(copy.deepcopy(layer_1))   #将第一层的值存储起来，方便反向传播用
        layer_2 = sigmoid(np.dot(w_o, layer_1))   #算输出层的值，维度(1,1)
        #loss = 1/2(y-pred)^2 没有必要写了，直接写梯度就行
        error = -(y-layer_2)    #损失对pred求导，记得这里pred = layer_2
        layer_delta2 = error * sigmoid_output_to_derivative(layer_2)    # 这里是输出层求导至zo的那一段，(1,1)
        layer_2_deltas.append(copy.deepcopy(layer_delta2)) #存储起来反向传播用
        prediction[binary_dim - i - 1] = np.round(layer_2[0][0]) #预测值,[0][0]是为了把1*1矩阵变成数
    future_layer_1_delta = np.zeros((hidden_dim, 1)) #这个是未来一个RNN单元求导至zh的δ(next)，存起来求wh用的
    #反向传播

    for i in range(binary_dim): #对于8位数，之前是从右往左前向传播，所以现在是从左往右反向传播
        X = np.array([[a[i], b[i]]]).T
        prev_layer_1 = layer_1_values[-i-2] #前一个RNN单元的值,因为包含初始化的0值层（第一个RNN单元的pre是0），所以-2
        layer_1 = layer_1_values[-i-1]  #当前的隐藏层值
        layer_delta2 = layer_2_deltas[-i-1] #将之前存着的输出层的误差求导取出来
        layer_delta1 = np.multiply(np.add(np.dot(w_h.T, future_layer_1_delta),np.dot(w_o.T, layer_delta2)), sigmoid_output_to_derivative(layer_1)) #根据当前的误差以及未来一层的误差求当前的wh
        w_i_update += np.dot(layer_delta1, X.T) #根据公式还要再乘上一项才是梯度
        w_h_update += np.dot(layer_delta1, prev_layer_1.T)
        w_o_update += np.dot(layer_delta2, layer_1.T)
        future_layer_1_delta = layer_delta1
    w_i -= alpha * w_i_update
    w_h -= alpha * w_h_update
    w_o -= alpha * w_o_update
    w_i_update *= 0
    w_o_update *= 0
    w_h_update *= 0
    # 验证结果

    if (j % Eval == 0): #每100次验证一次
        overallError = sum([1 if prediction[i] != label[i] else 0 for i in range(binary_dim)])#错了几个
        error_num_list.append(overallError)
        print("Error:" + str(overallError))
        print("Pred:" + str(prediction))
        print("True:" + str(label))
        out = 0
        for index, x in enumerate(reversed(prediction)): #二进制还原为10进制
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

plt.plot(np.arange(len(error_num_list))*Eval, error_num_list)
plt.ylabel('Error numbers')
plt.xlabel('Epochs')
plt.show()
