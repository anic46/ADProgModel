import gym
import numpy as np
import random

from garage.envs.step import Step
from garage.envs import EnvSpec

class BrainEnv(gym.Env):     
    # deterioration associated with metabolism and pathology

    def get_new_structure(self, D_old, Y_V, X_V):
        # evolution of amyloid deposition over time
        #D_new = D_old - self.beta * self.H@D_old + self.alpha_2*Y_V
        
        D_new = D_old - self.beta * self.H@D_old
        
        # new health of brain regions
        if self.degrade_model == 'new':
            delta_X = -self.alpha_1*D_new - self.alpha_2*Y_V/X_V
        else:
            delta_X = -self.alpha_1*D_new - self.alpha_2*Y_V
        #delta_X = -self.alpha_2*Y_V*np.exp(self.alpha_1*D_new)
        
        X_V_new = X_V + delta_X

        return X_V_new, D_new

    # compute energy consumed at frontal and mtl nodes (Yv)
    def calc_node_energy(self, Iv, Xv):
        """compute node energy"""
        if self.degrade_model == 'inverse':
            return self.gamma_v*Iv/Xv
        elif self.degrade_model == 'inverse-squared':
            return self.gamma_v*Iv/(Xv**2)
        else:
            raise NotImplementedError("relationship between Iv,Xv and Yv not defined")
    
    def __init__(self, network_size=2, num_edges=1, max_time_steps=7, alpha1_init=None, alpha2_init=None, beta_init=None, gamma_init=None, 
                 X_V_init=None, D_init=None, cog_type='fixed', cog_init=None, adj=None, action_limit=1.0, w_lambda=1.0, patient_idx=-1, gamma_type='fixed', action_type='delta', 
                 scale=False, use_gamma=False, degrade_model='old'):
        self.patient_idx = patient_idx
        self.gamma_type = gamma_type
        self.action_type = action_type
        self.normalize_ = scale
        self.lambda_ = w_lambda
        self.degrade_model = degrade_model
        self.cog_type = cog_type
        
        self.C_task = 10
        self.normalize_factor = 1/self.C_task if self.normalize_ else 1.0
        
        self.state_limits = np.array([[0,10],[0,5]])
        self.action_limit = action_limit
        self.reward_bound = 2000.0
        self.env_name = 'brain_env'
        
        self.cog_init_ = cog_init
        
        self.beta_init_ = beta_init
        self.alpha1_init_ = alpha1_init 
        self.alpha2_init_ = alpha2_init 
        self.gamma_init_ = gamma_init 
        self.X_V_init_ = X_V_init 
        self.D_init_ = D_init
        
        self.use_gamma_ = use_gamma
        
        self.adj_ = adj # adjacency matrix
        self.H = np.diag(np.sum(self.adj_, axis=1)) - self.adj_
        
        self.patient_count = D_init.shape[0]
        self.max_time_steps = max_time_steps
        
        if use_gamma:
            self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0,0]), high=np.array([self.C_task,self.C_task,5,5,2,2,self.C_task*5,1]), shape=(8, ), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=np.array([0.0,0.0,0,0,0,0]), high=np.array([self.C_task,self.C_task,5,5,2,2]), shape=(6, ), dtype=np.float64)
            
        self.action_space = gym.spaces.Box(low=-action_limit, high=action_limit, shape=(2, ), dtype=np.float64)
        self.reset()
        
    def reset(self, state_type='healthy', randomize=True):
        """Reset the environment."""
        self.t = 0
        patient_idx = self.patient_idx
        self.gamma_e = np.random.normal(3,1)

        if patient_idx == -1:
            patient_idx = np.random.randint(self.patient_count)

        self.beta = self.beta_init_[patient_idx]
        self.alpha_1 = self.alpha1_init_[patient_idx]
        self.alpha_2 = self.alpha2_init_[patient_idx]
        prod = self.alpha2_init_[patient_idx] * self.gamma_init_[patient_idx]
        
        if self.gamma_type == 'variable':
            self.gamma_v = np.random.choice([0.5,1.0,1.5,2.1,2.5,3.0])
            #self.gamma_v = np.random.choice([0.125,0.25,0.5,0.625,0.75])
            self.alpha_2 = prod/self.gamma_v
        else:
            self.gamma_v = self.gamma_init_[patient_idx]
        
        self.X_V = self.X_V_init_[patient_idx]
        self.D = self.D_init_[patient_idx]
        cog = self.cog_init_[patient_idx]
        
        #print(self.cog_type)
        if self.cog_type == 'variable':
            base = cog.sum()//2
            mtl_init = base + base*random.random()
            cog = np.array([mtl_init,base*2-mtl_init])
        
        self.cog=cog
        
        # initial state of the patient
        if self.use_gamma_:
            self.state = np.array([cog[0]*self.normalize_factor, cog[1]*self.normalize_factor, self.X_V[0], self.X_V[1], self.D[0], self.D[1], 1.0/self.X_V[0]*self.gamma_v, 0.0])
        else:
            self.state = np.array([cog[0]*self.normalize_factor, cog[1]*self.normalize_factor, self.X_V[0]/5.0, self.X_V[1]/5.0, self.D[0], self.D[1]])
        #Y_V = self.calc_node_energy(self.state[:2], self.X_V)
        self.reward = None
        return self.state
    
    def observe(self):
        return self.state
    
    def is_done(self):
        ##Check if the episode is complete
        return True if self.t >= self.max_time_steps else False
 
    # reward model for RL
    def calc_reward(self, state, M, Ct_1, C_first=None):    
        Ct = state.sum()
        C_task = self.C_task if C_first is None else C_first
        #print(C_task, C_first)
        power_factor = np.clip(-C_task + Ct,a_min=0,a_max=None)
        factor = 100**power_factor
        reward = -(np.abs(C_task - Ct)*factor*self.lambda_ + np.sum(M)) # + max(0,np.abs(Ct-Ct_1)/Ct_1 - 0.1))
        
        #t = 2.0 if t>5 else 1.0
        #print(np.abs(Ct-Ct_1), np.abs(C_task - Ct))
        
        # Constrain reward to be within specified range
        if np.isnan(reward):
            reward = -self.reward_bound
        elif reward > self.reward_bound:
            reward = self.reward_bound
        elif reward < -self.reward_bound:
            reward = -self.reward_bound
        return reward
 
    def step(self, action):
        self.t += 1
        next_state = self.state.copy()
        if self.normalize_:
            next_state[:2] = next_state[:2]*self.C_task
        
        a = action.copy()
        a = np.clip(a, self.action_space.low, self.action_space.high)
        
        if self.action_type == 'delta':
            if self.t == 1:
                next_state[:2] += a
            else:
                next_state[:2] += a
        else:
            next_state[:2] = a
        next_state = np.clip(next_state, a_min=self.observation_space.low, a_max=self.observation_space.high)
        
        Y_V = self.calc_node_energy(next_state[:2], self.X_V)
        health=self.X_V.copy()
        
        Ct_1 = self.state[:2].sum()*self.C_task
        
        # if self.t==1:
        #     reward = self.calc_reward(next_state[:2], Y_V, Ct_1, self.cog.sum())
        # else:
        reward = self.calc_reward(next_state[:2], Y_V, Ct_1)
        
        D_old = self.D.copy()
        X_V_old =  self.X_V.copy()
        self.X_V, self.D = self.get_new_structure(D_old.copy(), Y_V.copy(), X_V_old.copy())
        self.X_V = np.clip(self.X_V,a_min=0.0001, a_max=None)
        
        self.reward = reward
        if self.use_gamma_:
            self.state = np.array([next_state[0]*self.normalize_factor, next_state[1]*self.normalize_factor, self.X_V[0], self.X_V[1], self.D[0], self.D[1], Y_V.sum()*self.normalize_factor, self.t/11.]) 
        else:
            self.state = np.array([next_state[0]*self.normalize_factor, next_state[1]*self.normalize_factor, self.X_V[0]/5.0, self.X_V[1]/5.0, self.D[0], self.D[1]]) 
  
        done = self.is_done()
        return Step(observation=self.state, reward=reward, done=done, y=Y_V, health=health, D=D_old)
    
    def render(self, mode='human'):
        print(self.state, self.reward)

    def log_diagnostics(self, paths):
        pass
    
   
