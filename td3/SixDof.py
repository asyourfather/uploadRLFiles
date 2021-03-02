# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:34:44 2020

@author: 13668
"""
import numpy as np
import scipy.linalg as linalg
import math
class SixDofModelEuler():
    #基于欧拉角的六自由度积分
    def __init__(self, xyze0, uvw0, phithetapsi0, pqr0, mass, inertia):
        #默认初始参数为行向量
        #统一按行向量的形式存，矩阵乘法时再转
        self.xe = xyze0.flatten()
        self.vb = uvw0.flatten()
        
        self.euler = np.array([phithetapsi0[0], phithetapsi0[1], phithetapsi0[2]])
        self.quaterion = Euler2Quaterion(phithetapsi0)
        self.pqr = pqr0.flatten()
        self.mass = mass
        self.inertia = inertia
        self.invI = linalg.inv(self.inertia)
        self.w2b=Quaterion2DCM(self.quaterion)
        self.b2w = self.w2b.transpose()
        self.ve = np.matmul(self.b2w,self.vb.reshape(-1,1))
        
    def getstate(self):
         return self.xe, self.ve, self.euler, self.pqr
     
    def step(self, Fb, Mb, dt):
        #perform RK45
        #计算k1
        
        Ab_k1, pqrdot_k1, quateriondot_k1, ve_k1 = calDerivative(self.mass, self.inertia, self.invI, Fb, Mb, self.vb,
                                                                 self.pqr, self.quaterion)
        
        # 计算k2
        vb_k1 = self.vb + dt / 2 * Ab_k1
        pqr_k1 = self.pqr + dt / 2 * pqrdot_k1
        quaterion_k1 = self.quaterion + dt / 2 * quateriondot_k1
        Ab_k2, pqrdot_k2, quateriondot_k2, ve_k2 = calDerivative(self.mass, self.inertia, self.invI, Fb, Mb, vb_k1, pqr_k1,
                                                             quaterion_k1)
        # 计算k3
        vb_k2 = self.vb + dt / 2 * Ab_k2
        pqr_k2 = self.pqr + dt / 2 * pqrdot_k2
        quaterion_k2 = self.quaterion + dt / 2 * quateriondot_k2
        Ab_k3, pqrdot_k3, quateriondot_k3, ve_k3 = calDerivative(self.mass, self.inertia, self.invI, Fb, Mb, vb_k2, pqr_k2,
                                                             quaterion_k2)
        
        # 计算k4
        vb_k3 = self.vb + dt * Ab_k3
        pqr_k3 = self.pqr + dt * pqrdot_k3
        quaterion_k3 = self.quaterion + dt * quateriondot_k3
        Ab_k4, pqrdot_k4, quateriondot_k4, ve_k4 = calDerivative(self.mass, self.inertia, self.invI, Fb, Mb, vb_k3, pqr_k3,
                                                             quaterion_k3)
        
        self.vb += dt / 6 * (Ab_k1 + 2 * Ab_k2 + 2 * Ab_k3 + Ab_k4)
        self.pqr += dt / 6 * (pqrdot_k1 + 2 * pqrdot_k2 + 2 * pqrdot_k3 + pqrdot_k4)
        self.quaterion += dt / 6 * (quateriondot_k1 + 2 * quateriondot_k2 + 2 * quateriondot_k3 + quateriondot_k4)
        self.xe += dt / 6 * (ve_k1 + 2 * ve_k2 + 2 * ve_k3 + ve_k4)
        
        self.w2b = Quaterion2DCM(self.quaterion)
        self.b2w = self.w2b.transpose()
        self.ve = np.matmul(self.b2w, self.vb.reshape(-1, 1)).flatten()
        self.euler=Quaterion2Euler(self.quaterion)
        return self.xe, self.ve, self.euler, self.pqr
    
    
    def setIntialState(self, xyze0, uvw0, phithetapsi0, pqr0):
        self.xe = xyze0.flatten()
        self.vb = uvw0.flatten()
        
        self.euler = np.array([phithetapsi0[0], phithetapsi0[1], phithetapsi0[2]])
        self.quaterion = Euler2Quaterion(phithetapsi0)
        self.pqr = pqr0.flatten()
        self.invI = linalg.inv(self.inertia)
        self.w2b=Quaterion2DCM(self.quaterion)
        self.b2w = self.w2b.transpose()
        self.ve = np.matmul(self.b2w,self.vb.reshape(-1,1))
        

def calDerivative(mass, inertia, invI, Fb, Mb, vb, pqr, quaterion):
    # 给定状态，计算动力学方程的导数
    # 质心平动动力学
    Ab = Fb.flatten() / mass + np.cross(vb, pqr).flatten()

    # 姿态动力学
    I_w = np.matmul(inertia, pqr.reshape(-1, 1)).flatten()
    M = Mb.flatten() - np.cross(pqr, I_w).flatten()
    pqrdot = np.matmul(M, invI).flatten()

    # 姿态运动学
    w2b = Quaterion2DCM(quaterion)
    b2w = w2b.transpose()

    quateriondot = np.array([0.0, 0.0, 0.0, 0.0])
    p = pqr[0]
    q = pqr[1]
    r = pqr[2]
    epsilon = 1 - quaterion[0] * quaterion[0] - quaterion[1] * quaterion[1] - \
              quaterion[2] * quaterion[2] - quaterion[3] * quaterion[3]
    quateriondot[0] = (-p * quaterion[1] - q * quaterion[2] - r * quaterion[3]) / 2 + 1.0 * epsilon * quaterion[0]
    quateriondot[1] = (p * quaterion[0] + r * quaterion[2] - q * quaterion[3]) / 2 + 1.0 * epsilon * quaterion[1]
    quateriondot[2] = (q * quaterion[0] - r * quaterion[1] + p * quaterion[3]) / 2 + 1.0 * epsilon * quaterion[2]
    quateriondot[3] = (r * quaterion[0] + q * quaterion[1] - p * quaterion[2]) / 2 + 1.0 * epsilon * quaterion[3]
    ve = np.matmul(b2w, vb.reshape(-1, 1)).flatten()
    
    return Ab, pqrdot, quateriondot, ve

def Euler2Quaterion(euler):
    spsi = math.sin(euler[0] / 2)
    stheta = math.sin(euler[1] / 2)
    sphi = math.sin(euler[2] / 2)
    cpsi = math.cos(euler[0] / 2)
    ctheta = math.cos(euler[1] / 2)
    cphi = math.cos(euler[2] / 2)
    Quaterion = np.array([0.0, 0.0, 0.0, 0.0])
    Quaterion[0] = cpsi * ctheta * cphi + spsi * stheta * sphi
    Quaterion[1] = cpsi * ctheta * sphi - spsi * stheta * cphi
    Quaterion[2] = cpsi * stheta * cphi + spsi * ctheta * sphi
    Quaterion[3] = spsi * ctheta * cphi - cpsi * stheta * sphi
    return Quaterion

def Quaterion2Euler(Q):
    euler = np.array([0.0, 0.0, 0.0])
    q = math.sqrt(Q[0] * Q[0] + Q[1] * Q[1] + Q[2] * Q[2] + Q[3] * Q[3])
    Q_ = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(4):
        Q_[i] = Q[i] / q
    r11 = 2 * (Q_[1] * Q_[2] + Q_[0] * Q_[3])
    r12 = Q_[0] * Q_[0] + Q_[1] * Q_[1] - Q_[2] * Q_[2] - Q_[3] * Q_[3]
    r21 = -2 * (Q_[1] * Q_[3] - Q_[0] * Q_[2])
    r31 = 2 * (Q_[2] * Q_[3] + Q_[0] * Q_[1])
    r32 = Q_[0] * Q_[0] - Q_[1] * Q_[1] - Q_[2] * Q_[2] + Q_[3] * Q_[3]
    euler[0] = math.atan2(r11, r12)
    if r21 > 1.0:
        r21 = 1.0
    elif r21 < -1.0:
        r21 = -1.0
    euler[1] =math.asin(r21)
    euler[2] = math.atan2(r31, r32)
    return euler



def RotationAngle2DCM(euler,RO='ZYX'):
    if RO == 'ZYX':
        phi = euler[2]
        theta = euler[1]
        psi = euler[0]
        A = np.zeros([3, 3])
        A[0, 0] = np.cos(theta) * np.cos(psi)
        A[0, 1] = np.cos(theta) * np.sin(psi)
        A[0, 2] = -np.sin(theta)
        A[1, 0] = np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)
        A[1, 1] = np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)
        A[1, 2] = np.sin(phi) * np.cos(theta)
        A[2, 0] = np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)
        A[2, 1] = np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)
        A[2, 2] = np.cos(theta) * np.cos(phi)
        return A
    else:
        print("Not Supported Rotation Order!")


def Quaterion2DCM(Q):
    DCM = np.eye(3,dtype=np.float)
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    DCM[0][0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    DCM[1][1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    DCM[2][2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    DCM[0][1] = 2 * (q1 * q2 + q3 * q0)
    DCM[0][2] = 2 * (q1 * q3 - q2 * q0)
    DCM[1][0] = 2 * (q1 * q2 - q3 * q0)
    DCM[1][2] = 2 * (q2 * q3 + q1 * q0)
    DCM[2][0] = 2 * (q1 * q3 + q2 * q0)
    DCM[2][1] = 2 * (q2 * q3 - q1 * q0)
    return DCM
def unwrap(a):
    b = np.zeros_like(a)
    for i in range(len(a)):
        b[i] = np.mod(a[i], np.pi * 2)
        if b[i] > np.pi:
            b[i] -= 2 * np.pi
    return b
