from numpy import *
import numpy as np



import rtde_receive

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


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


'''
Made by 水木皆Ming
重力补偿计算
'''
class GravityCompensation:
    M = np.empty((0, 0))
    F = np.empty((0, 0))
    f = np.empty((0, 0))
    R = np.empty((0, 0))

    x = 0
    y = 0
    z = 0
    k1 = 0
    k2 = 0
    k3 = 0

    U = 0
    V = 0
    g = 0

    F_x0 = 0
    F_y0 = 0
    F_z0 = 0

    M_x0 = 0
    M_y0 = 0
    M_z0 = 0

    F_ex = 0
    F_ey = 0
    F_ez = 0

    M_ex = 0
    M_ey = 0
    M_ez = 0

    def Update_M(self, torque_data):
        M_x = torque_data[0]
        M_y = torque_data[1]
        M_z = torque_data[2]

        if (any(self.M)):
            M_1 = matrix([M_x, M_y, M_z]).transpose()
            self.M = vstack((self.M, M_1))
        else:
            self.M = matrix([M_x, M_y, M_z]).transpose()

    def Update_F(self, force_data):
        F_x = force_data[0]
        F_y = force_data[1]
        F_z = force_data[2]

        if (any(self.F)):
            F_1 = matrix([[0, F_z, -F_y, 1, 0, 0],
                          [-F_z, 0, F_x, 0, 1, 0],
                          [F_y, -F_x, 0, 0, 0, 1]])
            self.F = vstack((self.F, F_1))
        else:
            self.F = matrix([[0, F_z, -F_y, 1, 0, 0],
                             [-F_z, 0, F_x, 0, 1, 0],
                             [F_y, -F_x, 0, 0, 0, 1]])

    def Solve_A(self):
        A = dot(dot(linalg.inv(dot(self.F.transpose(), self.F)), self.F.transpose()), self.M)

        self.x = A[0, 0]
        self.y = A[1, 0]
        self.z = A[2, 0]
        self.k1 = A[3, 0]
        self.k2 = A[4, 0]
        self.k3 = A[5, 0]
        # print("A= \n" , A)
        print("x= ", self.x)
        print("y= ", self.y)
        print("z= ", self.z)
        print("k1= ", self.k1)
        print("k2= ", self.k2)
        print("k3= ", self.k3)

    def Update_f(self, force_data):
        F_x = force_data[0]
        F_y = force_data[1]
        F_z = force_data[2]

        if (any(self.f)):
            f_1 = matrix([F_x, F_y, F_z]).transpose()
            self.f = vstack((self.f, f_1))
        else:
            self.f = matrix([F_x, F_y, F_z]).transpose()

    def Update_R(self, euler_data):
        # 机械臂末端到基坐标的旋转矩阵
        R_array = self.eulerAngles2rotationMat(euler_data)

        alpha = (0) * 180 / np.pi

        # 力传感器到末端的旋转矩阵
        R_alpha = np.array([[math.cos(alpha), -math.sin(alpha), 0],
                            [math.sin(alpha), math.cos(alpha), 0],
                            [0, 0, 1]
                            ])

        R_array = np.dot(R_alpha, R_array.transpose())

        if (any(self.R)):
            R_1 = hstack((R_array, np.eye(3)))
            self.R = vstack((self.R, R_1))
        else:
            self.R = hstack((R_array, np.eye(3)))

    def Solve_B(self):
        B = dot(dot(linalg.inv(dot(self.R.transpose(), self.R)), self.R.transpose()), self.f)

        self.g = math.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
        self.U = math.asin(-B[1] / self.g)
        self.V = math.atan(-B[0] / B[2])

        self.F_x0 = B[3, 0]
        self.F_y0 = B[4, 0]
        self.F_z0 = B[5, 0]

        # print("B= \n" , B)
        print("g= ", self.g / 9.81)
        print("U= ", self.U * 180 / math.pi)
        print("V= ", self.V * 180 / math.pi)
        print("F_x0= ", self.F_x0)
        print("F_y0= ", self.F_y0)
        print("F_z0= ", self.F_z0)

    def Solve_Force(self, force_data, euler_data):
        Force_input = matrix([force_data[0], force_data[1], force_data[2]]).transpose()

        my_f = matrix([cos(self.U)*sin(self.V)*self.g, -sin(self.U)*self.g, -cos(self.U)*cos(self.V)*self.g, self.F_x0, self.F_y0, self.F_z0]).transpose()

        R_array = self.eulerAngles2rotationMat(euler_data)
        R_array = R_array.transpose()
        R_1 = hstack((R_array, np.eye(3)))

        Force_ex = Force_input - dot(R_1, my_f)
        print('接触力：\n', Force_ex)

    def Solve_Torque(self, torque_data, euler_data):
        Torque_input = matrix([torque_data[0], torque_data[1], torque_data[2]]).transpose()
        M_x0 = self.k1 - self.F_y0 * self.z + self.F_z0 * self.y
        M_y0 = self.k2 - self.F_z0 * self.x + self.F_x0 * self.z
        M_z0 = self.k3 - self.F_x0 * self.y + self.F_y0 * self.x

        Torque_zero = matrix([M_x0, M_y0, M_z0]).transpose()

        Gravity_param = matrix([[0, -self.z, self.y],
                                [self.z, 0, -self.x],
                                [-self.y, self.x, 0]])

        Gravity_input = matrix([cos(self.U)*sin(self.V)*self.g, -sin(self.U)*self.g, -cos(self.U)*cos(self.V)*self.g]).transpose()

        R_array = self.eulerAngles2rotationMat(euler_data)
        R_array = R_array.transpose()

        Torque_ex = Torque_input - Torque_zero - dot(dot(Gravity_param, R_array), Gravity_input)

        print('接触力矩：\n', Torque_ex)

    def eulerAngles2rotationMat(self, theta):
        theta = [i * math.pi / 180.0 for i in theta]  # 角度转弧度

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        # 第一个角为绕X轴旋转，第二个角为绕Y轴旋转，第三个角为绕Z轴旋转
        R = np.dot(R_x, np.dot(R_y, R_z))
        return R


def main():
    '''
    force_data = [-6.349214527290314e-05, 0.0016341784503310919, -24.31537437438965]
    torque_data = [-0.25042885541915894, 0.32582423090934753, 2.255179606436286e-05]
    euler_data = [-80.50866918099089, 77.83705434751874, -9.294185889510375 + 12]

    force_data1 = [-7.469202995300293, 2.3709897994995117, -23.0179500579834]
    torque_data1 = [-0.2169264256954193, 0.3719269931316376, 0.10870222747325897]
    euler_data1 = [-105.99038376663763, 60.89987226261212, -10.733422007074305 + 12]

    force_data2 = [-14.45930004119873, 0.995974063873291, -19.523677825927734]
    torque_data2 = [-0.19262456893920898, 0.3845194876194, 0.1622740775346756]
    euler_data2 = [-114.24258417090118, 43.78913507089547, -19.384088817327235 + 12]
    '''
    #[ 9.78120736e-01 -3.81116509e+00 -4.00626141e+00  9.81646341e-02 -7.43259582e-02 -2.85761283e-03]
    force_data = [9.78120736e-01, -3.81116509e+00, -4.00626141e+00]
    torque_data = [9.81646341e-02, -7.43259582e-02, -2.85761283e-03]
    #[37.929806037660496420440378486938,85.485303033518822748182846982645,-37.356848242529673211672396938797]
    euler_data = [-157.17561553,   88.37627645,  159.62509846]


    force_data1 = [0.6758556,  -3.07151178, -3.38905632]
    #[ 0.6758556  -3.07151178 -3.38905632  0.06719826 -0.09859436  0.01290188]
    torque_data1= [0.06719826, -0.09859436,  0.01290188]
    #[54.545582096454369474711843383028,33.346143676613910750296526101809,-56.894709056490744630660567730406]
    euler_data1 = [131.68478225,  50.98125058, 115.34919594]

    force_data2 = [6.67888594e-01, -3.42838083e+00, -4.30829616e+00]
    #[6.67888594e-01 - 3.42838083e+00 - 4.30829616e+00  1.04986415e-01,- 6.63832613e-02 - 7.95524178e-04]
    torque_data2= [1.04986415e-01, -6.63832613e-02, -7.95524178e-04]
    #[39.763270982079130688497919440989,99.121698557632415116860807828402,-5.0420285971512442371582376236413]
    euler_data2 = [61.46229426,  70.27313339, -39.10179626]

    compensation = GravityCompensation()

    compensation.Update_F(force_data)
    compensation.Update_F(force_data1)
    compensation.Update_F(force_data2)

    compensation.Update_M(torque_data)
    compensation.Update_M(torque_data1)
    compensation.Update_M(torque_data2)

    compensation.Solve_A()

    compensation.Update_f(force_data)
    compensation.Update_f(force_data1)
    compensation.Update_f(force_data2)

    compensation.Update_R(euler_data)
    compensation.Update_R(euler_data1)
    compensation.Update_R(euler_data2)

    compensation.Solve_B()
    '''
    compensation.Solve_Force(force_data, euler_data)
    compensation.Solve_Force(force_data1, euler_data1)
    compensation.Solve_Force(force_data2, euler_data2)
    #
    compensation.Solve_Torque(torque_data, euler_data)
    compensation.Solve_Torque(torque_data1, euler_data1)
    compensation.Solve_Torque(torque_data2, euler_data2)
    '''

    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")
    joint_q = rtde_r.getActualQ()
    FK = Forward_Kinematics(joint_q,d=[89.159,0,0,109.15,94.65,82.30],a=[0,-425.00,-392.25,0,0,0],alpha=[np.pi/2,0,0,np.pi/2,-np.pi/2,0])
    print(FK)
    R = np.array([[FK[0][0],FK[0][1],FK[0][2]],
                 [FK[1][0],FK[1][1],FK[1][2]],
                 [FK[2][0],FK[2][1],FK[2][2]]])
    print(rotationMatrixToEulerAngles(R)*180/np.pi)
if __name__ == "__main__":
    main()

