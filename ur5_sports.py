import numpy as np
import matplotlib.pyplot as plt
import struct
import socket
import time
import math

import xlrd
import xlwt

import rtde_control
import rtde_receive
import rtde_io

def cos(angle):
    return np.cos(angle)
def sin(angle):
    return np.sin(angle)
def atan2_math(y,x):
    return math.atan2(y,x)
def atan2(y,x):
    return np.arctan2(y,x)
def acos1(res):
    return np.arccos(res)
def sqrt(n):
    return np.sqrt(n)
def modify_S(x):
    if x >= 0:
        x = (x - (round(x)-1))*np.pi
    if x < 0:
        x = (round(x) - x)*np.pi
    return x
def modify(x):
    if x >= np.pi:
        x = x % np.pi
    if x < -np.pi:
        x = x % -np.pi
    return x
def array_modify(array):
    pass
def data_recv(recv):
    data = recv.recv(1000)
    fx = struct.unpack('f', data[6:10])[0]
    fy = struct.unpack('f', data[10:14])[0]
    fz = struct.unpack('f', data[14:18])[0]
    mx = struct.unpack('f', data[18:22])[0]
    my = struct.unpack('f', data[22:26])[0]
    mz = struct.unpack('f', data[26:30])[0]
    F = np.array([fx, fy, fz, mx, my, mz])
    return F
def sigmoid(z):
    h = 1. / (1 + np.exp(-z))
    return h
def de_sigmoid(h):
    return h * (1 - h)
# 无激活函数（全1矩阵）前反向传递函数
def no_active(z):
    h = z
    return h
def de_no_active(h):
    return np.ones(h.shape)
# 推演函数
def fead_forward(datas, layers):
    input_layers = []

    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer["act_fun"](z)
            input_layers.append(inputs)
        else:
            inputs = h
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer["act_fun"](z)
            input_layers.append(inputs)
    output = np.argmax(h)+1
    return output
def predict(F,model):
    trait = np.array([F[3] , F[4] , F[5] ])
    res = fead_forward(trait, model)
    return res

a = [0,-425.00,-392.25,0,0,0]#mm制
d = [89.159,0,0,109.15,94.65,82.30]

#a = [0,-0.42500,-0.39225,0,0,0]#m制
#d = [0.089159,0,0,0.10915,0.09465,0.08230]
alpha = [np.pi/2,0,0,np.pi/2,-np.pi/2,0]

#阻抗控制MBK
def MBK(x,dx,ddx,M,B,K,dt,F):
    ddx1 = (F - B * dx - K * x) / M
    dx1 = dt * (ddx1 + ddx) / 2 + dx  # v1=dt*(a0+a1)/2+v0  加速度一次积分得速度
    x1 = dt * (dx1 + dx) / 2 + x  # 同理速度一次积分得位移
    return x1, dx1, ddx1
#正逆运动学:正
#输入D-H参数，输出转换矩阵
def Transfer_Matrix(theta,d,a,alpha):
    #用于获取坐标系i ii到坐标系i − 1 i-1i−1的通用变换矩阵
    T = np.array([[cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta)],
                  [sin(theta), cos(theta)*cos(alpha) , -cos(theta)*sin(alpha), a*sin(theta)],
                  [0, sin(alpha), cos(alpha), d],
                  [0, 0, 0, 1]])
    return T
def Forward_Kinematics(theta,d,a,alpha):
    #正运动学直接建模
    T01 = Transfer_Matrix(theta[0], d[0], a[0], alpha[0])
    T12 = Transfer_Matrix(theta[1], d[1], a[1], alpha[1])
    T23 = Transfer_Matrix(theta[2], d[2], a[2], alpha[2])
    T34 = Transfer_Matrix(theta[3], d[3], a[3], alpha[3])
    T45 = Transfer_Matrix(theta[4], d[4], a[4], alpha[4])
    T56 = Transfer_Matrix(theta[5], d[5], a[5], alpha[5])
    T = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56)
    #print(T00)
    return T
#正逆运动学:逆
def Backward_Kinematics(G):
    nx = G[0][0]
    ny = G[1][0]
    nz = G[2][0]
    ox = G[0][1]
    oy = G[1][1]
    oz = G[2][1]
    ax = G[0][2]
    ay = G[1][2]
    az = G[2][2]
    px = G[0][3]
    py = G[1][3]
    pz = G[2][3]

    #关节1
    theta1 = np.zeros((1,2))
    m = d[5]*ay-py
    n = ax*d[5]-px
    theta1[0][0] = atan2(m,n)-atan2(d[3],(sqrt(m**2+n**2-d[3]*d[3])))
    theta1[0][1] = atan2(m,n)-atan2(d[3],-(sqrt(m**2+n**2-d[3]*d[3])))
    #关节5
    theta5 = np.zeros((2,2))
    theta5[0][0] = acos1(ax*sin(theta1[0][0])-ay*cos(theta1[0][0]))
    theta5[0][1] = acos1(ax*sin(theta1[0][1])-ay*cos(theta1[0][1]))
    theta5[1][0] = -acos1(ax*sin(theta1[0][0])-ay*cos(theta1[0][0]))
    theta5[1][1] = -acos1(ax*sin(theta1[0][1])-ay*cos(theta1[0][1]))
    #关节6
    mm = np.zeros((1,2))
    nn = np.zeros((1,2))
    theta6 = np.zeros((2,2))
    haha = np.zeros((6,1))
    mm[0][0] = nx * sin(theta1[0][0]) - ny * cos(theta1[0][0])
    nn[0][0] = ox * sin(theta1[0][0]) - oy * cos(theta1[0][0])
    mm[0][1] = nx * sin(theta1[0][1]) - ny * cos(theta1[0][1])
    nn[0][1] = ox * sin(theta1[0][1]) - oy * cos(theta1[0][1])
    haha[0][0] = mm[0][0]
    haha[1][0] = nn[0][0]
    haha[2][0] = mm[0][1]
    haha[3][0] = nn[0][1]
    haha[4][0] = nx
    haha[5][0] = ny
    #theta6 = atan2(mm, nn) - atan2(sin(theta5), 0);
    theta6[0][0] = atan2(mm[0][0], nn[0][0]) - atan2(sin(theta5[0][0]), 0)
    theta6[0][1] = atan2(mm[0][1], nn[0][1]) - atan2(sin(theta5[0][1]), 0)
    theta6[1][0] = atan2(mm[0][0], nn[0][0]) - atan2(sin(theta5[1][0]), 0)
    theta6[1][1]= atan2(mm[0][1], nn[0][1]) - atan2(sin(theta5[1][1]), 0)
    #关节3
    mmm = np.zeros((2,2))
    nnn = np.zeros((2,2))
    theta3 = np.zeros((4,2))
    mmm[0][0] = d[4] * (sin(theta6[0][0])* (nx * cos(theta1[0][0]) + ny * sin(theta1[0][0])) + cos(theta6[0][0])* (
                ox * cos(theta1[0][0]) + oy * sin(theta1[0][0]))) - d[5] * (
                            ax * cos(theta1[0][0]) + ay * sin(theta1[0][0])) + px * cos(theta1[0][0]) + py * sin(theta1[0][0])
    mmm[0][1] = d[4] * (sin(theta6[0][1]) * (nx * cos(theta1[0][1]) + ny * sin(theta1[0][1])) + cos(theta6[0][1]) * (
                ox * cos(theta1[0][1]) + oy * sin(theta1[0][1]))) - d[5] * (
                            ax * cos(theta1[0][1]) + ay * sin(theta1[0][1])) + px * cos(theta1[0][1]) + py * sin(theta1[0][1])

    nnn[0][0] = pz - d[0] - az * d[5] + d[4] * (oz * cos(theta6[0][0]) + nz * sin(theta6[0][0]))
    nnn[0][1] = pz - d[0] - az * d[5] + d[4] * (oz * cos(theta6[0][1]) + nz * sin(theta6[0][1]))
    mmm[1][0] = d[4] * (sin(theta6[1][0])* (nx * cos(theta1[0][0]) + ny * sin(theta1[0][0])) + cos(theta6[1][0]) * (
                ox * cos(theta1[0][0]) + oy * sin(theta1[0][0]))) - d[5] * (
                            ax * cos(theta1[0][0]) + ay * sin(theta1[0][0])) + px * cos(theta1[0][0]) + py * sin(theta1[0][0])
    mmm[1][1] = d[4] * (sin(theta6[1][1]) * (nx * cos(theta1[0][1]) + ny * sin(theta1[0][1])) + cos(theta6[1][1]) * (
                ox * cos(theta1[0][1]) + oy * sin(theta1[0][1]))) - d[5] * (
                            ax * cos(theta1[0][1]) + ay * sin(theta1[0][1])) + px * cos(theta1[0][1]) + py * sin(theta1[0][1])

    nnn[1][0] = pz - d[0] - az * d[5] + d[4] * (oz * cos(theta6[1][0]) + nz * sin(theta6[1][0]))

    nnn[1][1] = pz-d[0]-az*d[5]+d[4]*(oz*cos(theta6[1][1])+nz*sin(theta6[1][1]))

    theta3[0] = acos1((mmm[0]*mmm[0]+nnn[0]*nnn[0]-a[1]*a[1]-a[2]*a[2])/(2*a[1]*a[2]))
    theta3[1]=acos1((mmm[1]*mmm[1] + nnn[1]*nnn[1] - a[1]*a[1]-a[2]*a[2]) / (2 * a[1] * a[2]))
    theta3[2]=-acos1((mmm[0]*mmm[0] + nnn[0]*nnn[0] - a[1]*a[1]-a[2]*a[2]) / (2 * a[1] * a[2]))
    theta3[3]=-acos1((mmm[1]*mmm[1] + nnn[1]*nnn[1] - a[1]*a[1]-a[2]*a[2]) / (2 * a[1] * a[2]))
#关节2
    mmm_s2 = np.zeros((4,2))
    nnn_s2 = np.zeros((4,2))
    # mmm_s2(1: 2,:)=mmm(1: 2,:)
    mmm_s2[0] = mmm[0]
    mmm_s2[1] = mmm[1]
    # mmm_s2(3: 4,:)=mmm(1: 2,:)
    mmm_s2[2] = mmm[0]
    mmm_s2[3] = mmm[1]
    #nnn_s2(1: 2,:)=nnn(1: 2,:)
    nnn_s2[0] = nnn[0]
    nnn_s2[1] = nnn[1]
    # nnn_s2(3: 4,:)=nnn(1: 2,:)
    nnn_s2[2] = nnn[0]
    nnn_s2[3] = nnn[1]

    s2 = ((a[2] * cos(theta3) + a[1])*nnn_s2 - a[2] * sin(theta3) * mmm_s2) / (a[1]*a[1] + a[2]*a[2] + 2 * a[1] * a[2] * cos(theta3))
    c2 = (mmm_s2 + a[2] * sin(theta3)* s2) / (a[2] * cos(theta3) + a[1])
    theta2 = atan2(s2, c2)#有问题

#整理关节角
    theta = np.zeros((8,6))
    #theta[0:3][0] = theta1[0][0]
    theta[0][0] = theta1[0][0]
    theta[1][0] = theta1[0][0]
    theta[2][0] = theta1[0][0]
    theta[3][0] = theta1[0][0]
    #theta[4:7][0] = theta1[0][1]
    theta[4][0] = theta1[0][1]
    theta[5][0] = theta1[0][1]
    theta[6][0] = theta1[0][1]
    theta[7][0] = theta1[0][1]
    #theta[:][1]=[theta2[0][0], theta2[2][0], theta2[1][0], theta2[3][0], theta2[0][1], theta2[2][1], theta2[1][1],theta2[3][1]]
    theta[0][1] = theta2[0][0]
    theta[1][1] = theta2[2][0]
    theta[2][1] = theta2[1][0]
    theta[3][1] = theta2[3][0]
    theta[4][1] = theta2[0][1]
    theta[5][1] = theta2[2][1]
    theta[6][1] = theta2[1][1]
    theta[7][1] = theta2[3][1]
    #theta[:][2]=[theta3[0][0], theta3[2][0], theta3[1][0], theta3[3][0], theta3[0][1], theta3[2][1], theta3[1][1],theta3[3][1]]
    theta[0][2] = theta3[0][0]
    theta[1][2] = theta3[2][0]
    theta[2][2] = theta3[1][0]
    theta[3][2] = theta3[3][0]
    theta[4][2] = theta3[0][1]
    theta[5][2] = theta3[2][1]
    theta[6][2] = theta3[1][1]
    theta[7][2] = theta3[3][1]
    #theta[0:1][4]=theta5[0][0]
    theta[0][4] = theta5[0][0]
    theta[1][4] = theta5[0][0]
    #theta[2:3][4]=theta5[1][0]
    theta[2][4] = theta5[1][0]
    theta[3][4] = theta5[1][0]
    #theta[4:5][4]=theta5[0][1]
    theta[4][4] = theta5[0][1]
    theta[5][4] = theta5[0][1]
    #theta[6:7][4]=theta5[1][1]
    theta[6][4] = theta5[1][1]
    theta[7][4] = theta5[1][1]
    #theta[0:1][5]=theta6[0][0]
    theta[0][5] = theta6[0][0]
    theta[1][5] = theta6[0][0]
    #theta[2:3][5]=theta6[1][0]
    theta[2][5] = theta6[1][0]
    theta[3][5] = theta6[1][0]
    #theta[4:5][5]=theta6[0][1]
    theta[4][5] = theta6[0][1]
    theta[5][5] = theta6[0][1]
    #theta[6:7][5]=theta6[1][1]
    theta[6][5] = theta6[1][1]
    theta[7][5] = theta6[1][1]

#关节4
    theta[0][3] = atan2(-sin(theta[0][5]) * (nx * cos(theta[0][0]) + ny * sin(theta[0][0])) - cos(theta[0][5]) * (
                ox * cos(theta[0][0]) + oy * sin(theta[0][0])), oz * cos(theta[0][5]) + nz * sin(theta[0][5])) - \
                  theta[0][1] - theta[0][2]
    theta[1][3] = atan2(-sin(theta[1][5]) * (nx * cos(theta[1][0]) + ny * sin(theta[1][0])) - cos(theta[1][5]) * (
                ox * cos(theta[1][0]) + oy * sin(theta[1][0])), oz * cos(theta[1][5]) + nz * sin(theta[1][5])) - \
                  theta[1][1] - theta[1][2]
    theta[2][3] = atan2(-sin(theta[2][5]) * (nx * cos(theta[2][0]) + ny * sin(theta[2][0])) - cos(theta[2][5]) * (
                ox * cos(theta[2][0]) + oy * sin(theta[2][0])), oz * cos(theta[2][5]) + nz * sin(theta[2][5])) - \
                  theta[2][1] - theta[2][2]
    theta[3][3] = atan2(-sin(theta[3][5]) * (nx * cos(theta[3][0]) + ny * sin(theta[3][0])) - cos(theta[3][5]) * (
                ox * cos(theta[3][0]) + oy * sin(theta[3][0])), oz * cos(theta[3][5]) + nz * sin(theta[3][5])) - \
                  theta[3][1] - theta[3][2]
    theta[4][3] = atan2(-sin(theta[4][5]) * (nx * cos(theta[4][0]) + ny * sin(theta[4][0])) - cos(theta[4][5]) * (
                ox * cos(theta[4][0]) + oy * sin(theta[4][0])), oz * cos(theta[4][5]) + nz * sin(theta[4][5])) - \
                  theta[4][1] - theta[4][2]
    theta[5][3] = atan2(-sin(theta[5][5]) * (nx * cos(theta[5][0]) + ny * sin(theta[5][0])) - cos(theta[5][5]) * (
                ox * cos(theta[5][0]) + oy * sin(theta[5][0])), oz * cos(theta[5][5]) + nz * sin(theta[5][5])) - \
                  theta[5][1] - theta[5][2]
    theta[6][3] = atan2(-sin(theta[6][5]) * (nx * cos(theta[6][0]) + ny * sin(theta[6][0])) - cos(theta[6][5]) * (
                ox * cos(theta[6][0]) + oy * sin(theta[6][0])), oz * cos(theta[6][5]) + nz * sin(theta[6][5])) - \
                  theta[6][1] - theta[6][2]
    theta[7][3] = atan2(-sin(theta[7][5]) * (nx * cos(theta[7][0]) + ny * sin(theta[7][0])) - cos(theta[7][5]) * (
                ox * cos(theta[7][0]) + oy * sin(theta[7][0])), oz * cos(theta[7][5]) + nz * sin(theta[7][5])) - \
                  theta[7][1] - theta[7][2]
    #最后调整（限制角度大小）
    for i in range(8):
        for x in range(6):
            theta[i][x] = modify(theta[i][x])
    return theta

def Force_Compensation(joint_q,Fz):#用传感器Z轴力减掉该力
    #FZ0 = 2.349710013  #紧固力#待标定
    FZ1 = 5.362874011  #物件重力
    #TIEBAN = 1.564675237
    #MUBAN = 3.749816782
    #mt = 5.314492019
    #ding = 0.048381992
    angle_z = np.arccos(Forward_Kinematics(joint_q, d, a, alpha)[2][2])
    if angle_z <= np.pi / 2:
        fz = -FZ1 * np.cos(angle_z)   # Fz-fz
    if angle_z > np.pi / 2:
        fz = FZ1 * np.cos(np.pi - angle_z)   # Fz+fz
    #print(angle_z, angle_z * 180 / np.pi, F[2] - fz)
    FZ = Fz - fz
    return FZ

def Go_ahead(FK,velocity):
    #FK = Forward_Kinematics(joint_q, d, a, alpha)
    angle_z = np.arccos(FK[2][2])
    angle_y = np.arccos(FK[0][1])+np.pi/4
    angle_x = np.arccos(-FK[2][0])
    angle_z = np.pi/2 - angle_z
    TT = np.array([[1, 0, 0, velocity*sin(angle_x)],
                   [0, 1, 0, velocity*cos(angle_y)],
                   [0, 0, 1, velocity*cos(angle_z)],
                   [0, 0, 0, 1]])
    return TT

def Up_ahead(FK,velocity):
    angle_z = np.arccos(FK[2][2])
    angle_y = np.arccos(FK[2][1])
    angle_x = np.arccos(FK[2][0])
    TT = np.array([[1, 0, 0, -velocity*cos(angle_x)],
                   [0, 1, 0, velocity*cos(angle_y)],
                   [0, 0, 1, velocity*cos(angle_z)],
                   [0, 0, 0, 1]])
    return TT

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")#接收消息
rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")#控制消息

#joint_q = [-0.74036,-1.51756,-2.2404,0.59917,1.53746,1.55544]
#FK_T = Forward_Kinematics(joint_q, d, a, alpha)

IP_ADDR = '192.168.0.109'
PORT1 = 30005
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((IP_ADDR,PORT1))
#查看链接地址
sendData = "AT+EIP=?\r\n"
s.send(sendData.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#查看解耦矩阵
sendData = "AT+DCPMA6X8=?\r\n"
s.send(sendData.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#计算单位
sendData = "AT+DCPCU=?\r\n"
s.send(sendData.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#设置采样频率
set_update_rate = "AT+SMPF=100\r\n"
s.send(set_update_rate.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#矩阵运算单位
set_compute_unit = "AT+DCPCU=MVPV\r\n"
s.send(set_compute_unit.encode())
#上传数据格式
set_recieve_format = "AT+SGDM=(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)\r\n"
s.send(set_recieve_format.encode())
recvData = bytearray(s.recv(1000))
print(recvData)
#连续上传数据包
get_data_stream = "AT+GSD\r\n"
s.send(get_data_stream.encode())
print("开始接收\r\n")


wb = xlwt.Workbook(encoding='ascii')
ws = wb.add_sheet("first")
total =0
model = np.load(r"D:\Working park\model1.npy",allow_pickle=True)
velocityJ = 0.08
acceleration = 0.01
accelerationL = 0.005
dt = 1.0/1000
lookahead_time = 0.1
gain = 2000
velocityL = 0.005
#df = 0.0005
'''
joint_q = rtde_r.getActualQ()
FK_T = Forward_Kinematics(joint_q, d, a, alpha)
joint_c = [0,0,0,0,0,0]
for i in range(6):
    joint_c[i] = joint_q[i]*180/np.pi
print(joint_q)
print(FK_T)
TT = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,1],
               [0,0,0,1]])
FK_T = FK_T.dot(TT)
print(FK_T)
BK_T = Backward_Kinematics(FK_T)
joint_b = [0,0,0,0,0,0]
for i in range(8):
    right = 0
    for j in range(6):
        if abs(BK_T[i][j]-joint_q[j])<0.5:
            right += 1
    if right == 6:
        joint_b = BK_T[i]
print(joint_b)
'''



list = []
for i in range(20):
    f = data_recv(s)
    if i >= 10:
        list.append(Force_Compensation(rtde_r.getActualQ(),f[2]))
print(list)
Fastening_Force = sum(list)/len(list)

Mz = -0.245

while False:
    F = data_recv(s)
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    F_modify = Force_Compensation(joint_q, F[2]) - Fastening_Force
    Kp = 0.018
    Ki = 0.000
    Kd = 0.015
    F_expert = -3.0
    F_error = round(F_modify - F_expert, 1)
    print(F_error, F_modify)
    if F_error == 0.0:
        print("标准碰撞")
        std += 1
        if std == 50:
            print("结束")
            break
    else:
        std = 0
        FK_T = FK_T.dot(Go_ahead(FK_T, Kp * F_error + Ki * F_error_integ + Kd * (F_error - F_error_last)))

    BK_T = Backward_Kinematics(FK_T)
    for i in range(8):
        right = 0
        for j in range(6):
            if abs(BK_T[i][j] - joint_q[j]) < 0.6:
                right += 1
        if right == 5:
            joint_q = [BK_T[i][0], BK_T[i][1], BK_T[i][2], BK_T[i][3], BK_T[i][4], joint_q[5]]
            # print("新矩阵已产生")
            break
    F_error_last = F_error
    F_error_integ += F_error
    rtde_c.servoJ(joint_q, velocityJ, acceleration, dt, lookahead_time, gain)

    #print(np.arccos(Forward_Kinematics(joint_q, d, a, alpha)[2][2]))
    #print(F[2],Force_Compensation(joint_q, F[2])-Fastening_Force)


"""    #传感器补偿:(待实验+标定） 10.25已完成
    F = data_recv(s)
    joint_q = rtde_r.getActualQ()
    angle_z = np.arccos(Forward_Kinematics(joint_q, d, a, alpha)[2][2])
    print(angle_z,angle_z*180/np.pi,F[2])
    ws.write(total, 0, angle_z)
    ws.write(total, 1, angle_z*180/np.pi)
    ws.write(total, 2, F[2])
    total+=1
    if total >= 1000:
        wb.save(r"D:\Working park\fz4.xls")
        break
        pass"""

TT = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0.3],
                   [0, 0, 0, 1]])

joint_q = rtde_r.getActualQ()
Forward_Flag = True
Search_Flag = False
PID_Flag = False
Putin_Flag = False
smash = 0
dj3 = joint_q[3]
dj4 = joint_q[4]
while Forward_Flag:
    F = data_recv(s)
    FK = Forward_Kinematics(rtde_r.getActualQ(),d,a,alpha)

    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    print(FK[2][2])
    print(F[2],Force_Compensation(joint_q,F[2]) - Fastening_Force,-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
    kp = 0.00018
    if Force_Compensation(joint_q,F[2]) - Fastening_Force<-1.5:
        joint_e = rtde_r.getActualQ()
        res = predict(F, model)
        print("结果为：", res)
        if res == 1:  # My
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            joint_q[3] += df
            joint_q[4] += df
            smash = 0
        if res == 4:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[3] -= df
            joint_q[4] -= df
        if res == 3:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[3] -= df
            joint_q[4] += df
        if res == 2:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[3] += df
            joint_q[4] -= df
        if res == 5:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[3] += df
        if res == 6:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[3] -= df
        if res == 7:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[4] += df
        if res == 8:
            df = kp * (-1.5 - (Force_Compensation(joint_q, F[2]) - Fastening_Force))
            smash = 0
            joint_q[4] -= df
        if res == 9:
            smash += 1
            print("正碰撞!")
        if smash >= 100:
            Forward_Flag = False
            PID_Flag = True

    else:
        #print("结果为：未碰撞")
        FK_T = FK_T.dot(Go_ahead(FK_T,0.05))#获取往前推的新T矩阵
        #print(FK_T[0][3],FK_T[1][3],FK_T[2][3])
        BK_T = Backward_Kinematics(FK_T)
        #print(BK_T)
        #print(joint_q)
        for i in range(8):
            right = 0
            for j in range(6):
                if abs(BK_T[i][j]-joint_q[j])<0.6:
                    right += 1
            if right == 5:
                joint_q = [BK_T[i][0],BK_T[i][1],BK_T[i][2],BK_T[i][3],BK_T[i][4],joint_q[5]]
                #print("新矩阵已产生")
                break
    rtde_c.servoJ(joint_q, velocityJ, acceleration, dt, lookahead_time, gain)

std = 0
F_error_last = 0
F_error_integ = 0
while PID_Flag:
    F = data_recv(s)
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    F_modify = Force_Compensation(joint_q, F[2]) - Fastening_Force
    Kp = 0.018
    Ki = 0.000
    Kd = 0.015
    F_expert = -0.6
    F_error = round(F_modify - F_expert, 1)
    print(F_error,F_modify)
    if F_error == 0.0:
        print("标准碰撞")
        std += 1
        if std == 50:
            print("结束")
            PID_Flag = False
            Search_Flag = True
    else:
        std = 0
        FK_T = FK_T.dot(Go_ahead(FK_T,Kp*F_error+Ki*F_error_integ+Kd*(F_error-F_error_last)))

    BK_T = Backward_Kinematics(FK_T)
    for i in range(8):
        right = 0
        for j in range(6):
            if abs(BK_T[i][j] - joint_q[j]) < 0.6:
                right += 1
        if right == 5:
            joint_q = [BK_T[i][0], BK_T[i][1], BK_T[i][2], BK_T[i][3], BK_T[i][4], joint_q[5]]
            # print("新矩阵已产生")
            break
    F_error_last = F_error
    F_error_integ += F_error
    rtde_c.servoJ(joint_q, velocityJ, acceleration, dt, lookahead_time, gain)


yes = 0
while Search_Flag:
    F = data_recv(s)
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    F_modify = Force_Compensation(joint_q, F[2]) - Fastening_Force
    print(F[2], F_modify)
    # print(np.arccos(FK_T[2][1])*180/np.pi)
    if F_modify <= -0.1:
        FK_T = FK_T.dot(Up_ahead(FK_T, -0.015))
    if F_modify >-0.1:
        yes += 1
        FK_T = FK_T.dot(Up_ahead(FK_T, -0.01))
        print("找到空位")
    if yes >= 150:
        print("完成")
        Putin_Flag = True
        Search_Flag = False
        break
    BK_T = Backward_Kinematics(FK_T)
    for i in range(8):
        right = 0
        for j in range(6):
            if abs(BK_T[i][j] - joint_q[j]) < 0.6:
                right += 1
        if right == 5:
            joint_q = [BK_T[i][0], BK_T[i][1], BK_T[i][2], BK_T[i][3], BK_T[i][4], joint_q[5]]
            # print("新矩阵已产生")
            break

    rtde_c.servoJ(joint_q, velocityJ, acceleration, dt, lookahead_time, gain)

ok = 0
while Putin_Flag:
    F = data_recv(s)
    FK_T = Forward_Kinematics(joint_q, d, a, alpha)
    F_modify = Force_Compensation(joint_q, F[2]) - Fastening_Force
    print(F[2], F_modify)
    if F_modify > -0.4:
        FK_T = FK_T.dot(Go_ahead(FK_T, 0.02))
    else:
        ok+=1
        print("产生碰撞！")
    if ok >= 100:
        print("完成")
        break
    BK_T = Backward_Kinematics(FK_T)
    for i in range(8):
        right = 0
        for j in range(6):
            if abs(BK_T[i][j] - joint_q[j]) < 0.6:
                right += 1
        if right == 5:
            joint_q = [BK_T[i][0], BK_T[i][1], BK_T[i][2], BK_T[i][3], BK_T[i][4], joint_q[5]]
            # print("新矩阵已产生")
            break

    rtde_c.servoJ(joint_q, velocityJ, acceleration, dt, lookahead_time, gain)