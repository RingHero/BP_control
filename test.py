import time

import numpy as np
import matplotlib.pyplot as plt
import struct
import socket
import rtde_control
import rtde_receive

import random
import xlrd
import xlwt
from xlutils.copy import copy
from random import randint

def cos(angle):
    return np.cos(angle)
def sin(angle):
    return np.sin(angle)
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

# 左上角点1  右上角2  左下角3  右下角4  上边5  下边6  左边7  右边8  正碰撞9


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
    output = np.argmax(h) + 1
    return output


def predict(F, model):
    trait = np.array([F[3]/F[2],F[4]/F[2],F[5]/F[2]])
    res = fead_forward(trait, model)
    return res


def write_exl(matrix):
    wb = xlwt.Workbook(encoding='ascii')
    ws = wb.add_sheet("first")

    for i in range(100):
        ws.write(i, 0, random.randint(75, 85))

    wb.save("D:\Working park\ew.xls")


class NET:
    def __init__(self, modelpath):
        self.model_path = modelpath
        self.model = np.load(self.model_path, allow_pickle=True)

    def print_model(self):
        print("模型如下")
        print(self.model)


class CSqQueue:

    def __init__(self):
        self.__MaxSize = 10
        self.data = [None] * self.__MaxSize
        self.front = self.rear = 0

    def empty(self):
        if self.front == self.rear:
            return True
        return False

    def enough(self):
        if (self.rear + 1) % self.__MaxSize == self.front:
            return True
        return False

    def push(self, e):
        assert (self.rear + 1) % self.__MaxSize != self.front
        self.rear = (self.rear + 1) % self.__MaxSize
        self.data[self.rear] = e

    def pop(self):
        assert not self.empty()
        self.front = (self.front + 1) % self.__MaxSize
        return self.data[self.front]

    def gethead(self):
        assert not self.empty()
        return self.data[(self.front + 1) % self.__MaxSize]
'''
IP_ADDR = '192.168.0.109'
PORT1 = 30005
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((IP_ADDR,PORT1))
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
'''
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

#加载模型
model = np.load(r"D:\Working park\model2.npy",allow_pickle=True)

#rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")#接收消息
#rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")#控制消息

velocity = 0.001
accelerationL = 0.005
acceleration = 0.001
dt = 0.1
lookahead_time = 0.1
gain = 1500
speed = [0.01,-0.01,0,0,0,0]
double_time = 0.0
blend0 = 0.02

M = Forward_Kinematics([-0.680206601,-1.488890473,-2.14605457,0.776520848,1.327429771,1.436538935],[89.159,0,0,109.15,94.65,82.30],[0,-425.00,-392.25,0,0,0],[np.pi/2,0,0,np.pi/2,-np.pi/2,0])
RT = np.array([[M[0][0],M[0][1],M[0][2]],
               [M[1][0],M[1][1],M[1][2]],
               [M[2][0],M[2][1],M[2][2]]])
print(RT)

#[0.30012767251710276, -0.46379931093045756, 0.3517919996035615, 0.6130314041605509, 1.4705270567231168, -0.6344948306571077]

#[0.3162100409464825, -0.4799698089510645, 0.3520365695703271, 0.6131265610919395, 1.4705759779767826, -0.634473202736846]