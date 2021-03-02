# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:09:48 2021

@author: 13668
"""

import numpy as np
from SixDof import SixDofModelEuler
class GymAPIobservation():
    def __init__(self,shape):
        self.shape=shape

class GymAPIaction():
    def __init__(self,shape):
        self.shape=shape
        self.high=np.array([1.0,1.0,1.0,1.0,1.0,1.0],dtype=np.float32)
    def sample(self):
        return np.random.uniform(-self.high[0],self.high[0],self.shape)

class LanderEnv():
    def __init__(self):
        mass0=1700.0
        Inertia=np.eye(3,dtype=float)
        a=3.0
        b=3.0
        c=1.0
        Inertia[0][0]=(b*b+c*c)*mass0/12
        Inertia[1][1]=(a*a+c*c)*mass0/12
        Inertia[2][2]=(a*a+b*b)*mass0/12
        
        self.Model=SixDofModelEuler(np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), mass0, Inertia)
        self.dt=0.05
        self.Tf=50.0
        self.t=0.0
        #self.constantReward = 50.0*self.dt/self.Tf
        self.constantReward = 70.0*self.dt/self.Tf
        #推力范围
        Tmax=3600.0
        ThrustUpperLimit=0.8*Tmax
        ThrustLowerLimit=0.3*Tmax
        self.delta_Thrust=(ThrustUpperLimit-ThrustLowerLimit)/2
        self.med_thrust=(ThrustUpperLimit+ThrustLowerLimit)/2
        
        
        #添加OpenAI Gym 环境的API以DDPG被调用
        obs_shape=(16,)
        act_shape=(6,)
        self.observation_space=GymAPIobservation(obs_shape)
        self.action_space=GymAPIaction(act_shape)

        #动力学模型
        L=2.0
        delta=27.0/180.0*np.pi
        temp1=np.array([[-np.cos(0.0),-np.cos(np.pi/3),-np.cos(np.pi*2/3),-np.cos(np.pi),-np.cos(np.pi*4/3),-np.cos(np.pi*5/3)],
                         [-np.sin(0.0),-np.sin(np.pi/3),-np.sin(np.pi*2/3),-np.sin(np.pi),-np.sin(np.pi*4/3),-np.sin(np.pi*5/3)],
                         [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]])
        self.Force_matrix=np.matmul(np.diag([np.sin(delta),np.sin(delta),np.cos(delta)]),temp1)
        
        
        temp2=np.array([[np.sin(0.0),np.sin(np.pi/3),np.sin(np.pi*2/3),np.sin(np.pi),np.sin(np.pi*4/3),np.sin(np.pi*5/3)],
                        [-np.cos(0.0),-np.cos(np.pi/3),-np.cos(np.pi*2/3),-np.cos(np.pi),-np.cos(np.pi*4/3),-np.cos(np.pi*5/3)],
                        [0.0,0.0,0.0,0.0,0.0,0.0]])
        self.Torch_matrix=-np.matmul(np.diag([L*np.cos(delta),L*np.cos(delta),0.0]),temp2)
        self.gmars=3.7114
        self.Isp=225.0
        self.ge=9.8
        self.eulerInitBound=np.array([np.pi*10.0/180.0,np.pi*10.0/180.0,np.pi*10.0/180.0])
        self.pqrInitBound=np.array([np.pi*5.0/180.0,np.pi*5.0/180.0,np.pi*5.0/180.0])
        self.velInitBoundUpper=np.array([3.0,3.0,10.0])
        self.velInitBoundLower=np.array([-3.0,-3.0,9.0])
        self.posInitBoundUpper=np.array([500.0,500.0,-1500.0])
        self.posInitBoundLower=np.array([-500.0,-500.0,-1700.0])
        
        self.v_error_obsgain=0.1
        
    def step(self, a):
        self.t+=self.dt
        thrust=np.array(np.clip(a,-1,1)*self.delta_Thrust+self.med_thrust)
        
        Fb_thrust=np.matmul(self.Force_matrix,thrust).flatten()
        Fb_G=np.matmul(self.Model.w2b,np.array([[0.0],[0.0],[self.Model.mass*self.gmars]])).flatten()
        Fb=Fb_G+Fb_thrust
        Mb=np.matmul(self.Torch_matrix,thrust)
        
        self.Model.step(Fb,Mb,self.dt)
        
        ob=self._get_obs()
        v=np.sqrt(ob[0]*ob[0]+ob[1]*ob[1]+ob[2]*ob[2])/self.v_error_obsgain
        if np.abs(self.Model.euler[0])>np.pi/3 or np.abs(self.Model.euler[1])>np.pi/3 or v>20:
            done=True
        else:
            done=False
            
        #计算奖励值
        reward=0.0
        if done:
            #reward+=-20.0
            reward+=-20.0
        # if v<2:
        #     reward+=self.constantReward
        if v<1.0:
            reward+=2.0*self.constantReward
        if v<0.3:
            reward+=3.0*self.constantReward
        reward+=self.constantReward
        reward+=-0.05*np.square(v/5.0)
        if v>10.0:
            reward+=0.0
        elif (v<=10.0 and v>2.0):
            reward+=-8.0*self.constantReward/8.0*(v-10.0)
        else:
            reward+=8.0*self.constantReward-4.0*self.constantReward/2.0*(v-2.0)
        
        return ob, reward,done,{}

    def _get_obs(self):
        v_error_w=-self.Model.ve.reshape(-1,1)
        v_error_b=np.matmul(self.Model.w2b,v_error_w).flatten()
        obs=np.array([v_error_b[0]*self.v_error_obsgain,
                      v_error_b[1]*self.v_error_obsgain,
                      v_error_b[2]*self.v_error_obsgain,
                      np.sin(self.Model.euler[0]),
                      np.cos(self.Model.euler[0]),
                      np.sin(self.Model.euler[1]),
                      np.cos(self.Model.euler[1]),
                      np.sin(self.Model.euler[2]),
                      np.cos(self.Model.euler[2]),
                      self.Model.pqr[0],
                      self.Model.pqr[1],
                      self.Model.pqr[2],
                      self.Model.quaterion[0],
                      self.Model.quaterion[1],
                      self.Model.quaterion[2],
                      self.Model.quaterion[3]])
        
        return obs

    def reset(self):
        
        euler = np.array([np.random.uniform(-self.eulerInitBound[0],self.eulerInitBound[0]),\
                                   np.random.uniform(-self.eulerInitBound[1],self.eulerInitBound[1]),\
                                   np.random.uniform(-self.eulerInitBound[2],self.eulerInitBound[2])])
    
        pqr = np.array([np.random.uniform(-self.pqrInitBound[0],self.pqrInitBound[0]),\
                                   np.random.uniform(-self.pqrInitBound[1],self.pqrInitBound[1]),\
                                   np.random.uniform(-self.pqrInitBound[2],self.pqrInitBound[2])])
        
        xe = np.array([np.random.uniform(self.posInitBoundLower[0],self.posInitBoundUpper[0]),\
                                   np.random.uniform(self.posInitBoundLower[1],self.posInitBoundUpper[1]),\
                                   np.random.uniform(self.posInitBoundLower[2],self.posInitBoundUpper[2])])
    
        vb = np.array([np.random.uniform(self.velInitBoundLower[0],self.velInitBoundUpper[0]),\
                                   np.random.uniform(self.velInitBoundLower[1],self.velInitBoundUpper[1]),\
                                   np.random.uniform(self.velInitBoundLower[2],self.velInitBoundUpper[2])])
        self.Model.setIntialState(xe,vb,euler,pqr)
        return self._get_obs()

