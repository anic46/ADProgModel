import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from envs import BrainEnv

import tensorflow as tf
import pandas as pd
import numpy as np
# import stable_baselines
# from stable_baselines import TRPO, PPO2
# from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

import garage
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.experiment import Snapshotter


def error_plot(mat, ylab):
    x = np.arange(1,mat.T.shape[0]+1)
    mtl_mean = mat.mean(axis=0)
    mtl_max = mat.max(axis=0)
    mtl_min = mat.min(axis=0)
    plt.plot(x,mat.T,alpha=0.6)
    plt.plot(x,mtl_mean,'k-',color='black', linewidth='2')
    plt.fill_between(x,mtl_min,mtl_max,color='gray', alpha=0.2)
    plt.ylabel(ylab)
    plt.xlabel("Time(years)")
    plt.tight_layout()
    plt.xticks(range(1,mat.T.shape[0]+1))
    
# individual plot function for synthetic data
def plot_synthetic(df, filepath, type=''):
    font = {'family' : 'arial',
            'weight' : 'normal',
            'size'   : 15}
    
    df['reward'] =  (-np.abs(df['reg1_info'+type] + df['reg2_info'+type] - 5) - (df['reg1_fdg'+type] + df['reg2_fdg'])).values
    
    palette = sns.color_palette("Spectral", as_cmap=True)
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(30,20))
    plt.subplot(4, 4, 1)
    sns.lineplot(data=df, x="Years", y="cogsc"+type, hue='RID', palette=palette,marker="o")
    sns.lineplot(data=df, x="Years", y="cogsc"+type, marker="o", color="black", linewidth="2.5")
    field = "cogsc"+type
    plt.title("Mean Total Cognition:" + str(np.round(df[field].mean(),2)))
    plt.legend([],[], frameon=False)
    
    
    plt.subplot(4, 4, 2)
    sns.lineplot(data=df, x="Years", y="reg1_info"+type, hue='RID', palette=palette,marker="o")
    sns.lineplot(data=df, x="Years", y="reg1_info"+type, marker="o", color="black", linewidth="2.5")
    plt.title("Mean Total MTL Load:" + str(np.round(df['reg1_info' + type].mean(),2)))
    plt.legend([],[], frameon=False)

    #plt.ylim([0,11])

    plt.subplot(4, 4, 3)
    #plt.plot(ftl_load.T)
    sns.lineplot(data=df, x="Years", y="reg2_info"+type, hue='RID', palette=palette,marker="o")
    sns.lineplot(data=df, x="Years", y="reg2_info"+type, marker="o", color="black", linewidth="2.5")
    plt.title("Mean Total Frontal Load:" + str(np.round(df['reg2_info' + type].mean(),2)))
    plt.legend([],[], frameon=False)
    #plt.ylim([0,11])

    df['total_energy'] = df['reg1_fdg'+type] + df['reg2_fdg'+type]
    plt.subplot(4, 4, 4)
    sns.lineplot(data=df, x="Years", y="total_energy", hue='RID', palette=palette,marker="o")
    sns.lineplot(data=df, x="Years", y="total_energy", marker="o", color="black", linewidth="2.5")
    plt.title(f"Mean Total Metabolic Cost:{np.round(df['total_energy'].mean(),2)}")
    plt.legend([],[], frameon=False)
    #plt.ylim([3,20])

    #plt.tight_layout()
    plt.savefig(filepath)
    
# individual plot function for adni data
def plot_real(df, filepath):
    font = {'family' : 'arial',
            'weight' : 'normal',
            'size'   : 15}
    n = len(np.unique(df['RID']))
    palette = sns.color_palette("Paired", n)
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(40,30))
    plt.subplot(2, 2, 1)
    sns.lineplot(data=df, x="Years", y="cogsc", hue='RID', palette=palette,marker="o")
    plt.title(f"Mean Total Cognition:{np.round(df['cogsc'].mean(),2)}")
    plt.legend([],[], frameon=False)
    plt.ylim([0,11])
    
    plt.subplot(2, 2, 2)
    sns.lineplot(data=df, x="Years", y="cogsc", marker="o")
    plt.title(f"Mean Total Cognition:{np.round(df['cogsc'].mean(),2)}")
    plt.legend([],[], frameon=False)
    plt.ylim([0,11])
   
    plt.savefig(filepath)
    
#plot function for generating output png files
def plot_curves(mtl_load, ftl_load, mtl_energy, ftl_energy, mtl_h, ftl_h, filepath):
    font = {'family' : 'arial',
            'weight' : 'normal',
            'size'   : 15}

    cognition = mtl_load + ftl_load

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(20,15))

    plt.subplot(4, 4, 1)
    error_plot(cognition[:,:],"Cognition (C)")
    plt.title(f"Mean Total Cognition:{np.round(cognition.sum(axis=1).mean(),2)}")
    #plt.legend(RID_decline)
    #plt.ylim([0,10])

    plt.subplot(4, 4, 2)
    error_plot(mtl_load,"MTL Information Load ($I_v)$")
    plt.title(f"Mean Total MTL Load:{np.round(mtl_load.sum(axis=1).mean(),2)}")

    plt.ylim([0,11])

    plt.subplot(4, 4, 3)
    #plt.plot(ftl_load.T)
    error_plot(ftl_load,"Frontal Information Load ($I_v)$")
    plt.title(f"Mean Total Frontal Load:{np.round(ftl_load.sum(axis=1).mean(),2)}")
    plt.ylim([0,11])

    total_energy = mtl_energy+ftl_energy
    plt.subplot(4, 4, 4)
    error_plot(total_energy,"Total Energy ($Y_v$)")
    plt.title(f"Mean Total Metabolic Cost:{np.round(total_energy.sum(axis=1).mean(),2)}")
    #plt.ylim([3,20])

    #plt.tight_layout()
    plt.savefig(filepath)
    
# policy evaluation class
class EvalPolicy():
    def __init__(self, T=11, snapshot_dir=None, log_dir=None, gamma=2.1, gamma_type='fixed', cog_init=None, adj=None, action_type='delta', action_limit=1.0, w_lambda=1.0, energy_model='inverse'):
        self.T = T
        self.gamma = gamma
        self.snapshot_dir = snapshot_dir
        self.log_dir = log_dir
        self.cog_init = cog_init
        self.adj = adj
        self.gamma_type = gamma_type
        self.action_type = action_type
        self.action_limit = action_limit
        self.w_lambda=w_lambda
        self.energy_model=energy_model


    def simulate(self, data=None, data_type='test', scale_state=True, normalize_state=False):
        gamma_init = np.ones(len(data[0])) * self.gamma
        gamma_val = data[2][0]
        alpha2_init_new = gamma_val*data[1]/self.gamma
        scale_factor = 10.0 if scale_state else 1.0
        
        snapshotter = Snapshotter()
        tf.compat.v1.reset_default_graph()
        
        RIDs = data[-2]
        n_sim = len(RIDs)
        self.mtl_load = np.zeros((n_sim, self.T))
        self.ftl_load = np.zeros((n_sim, self.T))

        self.mtl_energy = np.zeros((n_sim, self.T))
        self.ftl_energy = np.zeros((n_sim, self.T))
        self.mtl_h = np.zeros((n_sim, self.T))
        self.ftl_h = np.zeros((n_sim, self.T))

        self.mtl_d = np.zeros((n_sim, self.T))
        self.ftl_d = np.zeros((n_sim, self.T))

        out_data = []
        self.cognition_vec_rl = []
        self.reward_vec_rl = []
        
        with tf.compat.v1.Session(): # optional, only for TensorFlow
            trained_data = snapshotter.load(self.log_dir)
            policy = trained_data['algo'].policy
            
            for i,j in enumerate(RIDs):
                done = False
                if normalize_state:
                    env = BrainEnv(max_time_steps=self.T+1, alpha1_init=data[0], alpha2_init=alpha2_init_new, beta_init=data[3], gamma_init=gamma_init, X_V_init=data[5], \
                                            D_init=data[4], cog_init=data[-1], adj=self.adj, action_limit=self.action_limit, w_lambda=self.w_lambda, patient_idx=i, gamma_type=self.gamma_type, action_type=self.action_type, scale=scale_state, energy_model=self.energy_model)
                    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
                else:
                    env = BrainEnv(max_time_steps=self.T+1, alpha1_init=data[0], alpha2_init=alpha2_init_new, beta_init=data[3], gamma_init=gamma_init, X_V_init=data[5], \
                                        D_init=data[4], cog_init=data[-1], adj=self.adj, action_limit=self.action_limit, w_lambda=self.w_lambda, patient_idx=i, gamma_type=self.gamma_type, action_type=self.action_type, scale=scale_state, energy_model=self.energy_model)
                
                obs = env.reset()  # The initial observation
                policy.reset()
                steps, max_steps = 0, self.T #data[-2][i]
                
                while steps < max_steps:
                    # Getting policy action (due to stochastic nature, we select the mean)
                    action = policy.get_action(obs)[1]['mean']
                    obs, rew, done, eg = env.step(action)
                    
                    self.mtl_energy[i,steps], self.ftl_energy[i,steps] = eg['y']
                    self.mtl_h[i,steps], self.ftl_h[i,steps] = eg['health']
                    self.mtl_d[i,steps], self.ftl_d[i,steps] = eg['D']
                    self.mtl_load[i,steps], self.ftl_load[i,steps] = obs[:2]*scale_factor
                    
                    out_data.append([j, steps, obs[0]*scale_factor, obs[1]*scale_factor, eg['y'][0], eg['y'][1], eg['health'][0], eg['health'][1], eg['D'][0], eg['D'][1], data[3][i], data[0][i], alpha2_init_new[i], gamma_init[i]])
                    self.cognition_vec_rl.append(obs[0]+obs[1])
                    self.reward_vec_rl.append(rew)
                    
                    steps += 1

                env.close()
                
        try:
            plot_curves(self.mtl_load, self.ftl_load, self.mtl_energy, self.ftl_energy, self.mtl_h, self.ftl_h, f'{self.snapshot_dir}/rl_traj_full_{data_type}.png')   
        except:
            print("##################################################error####################################")
        
        columns = ['RID','Years','reg1_info','reg2_info','reg1_fdg','reg2_fdg','reg1_mri','reg2_mri','reg1_D','reg2_D','beta','alpha1','alpha2','gamma']
        new_columns = []
        
        self.output = pd.DataFrame(out_data, columns=columns)
        
        for j,c in enumerate(columns):
            if j >= 2:
                new_columns.append(c + "_rl")
            else:
                new_columns.append(c)
                
        self.output.columns = new_columns
        self.output['cogsc_rl'] = self.output['reg1_info_rl'] + self.output['reg2_info_rl']
    
    # computes the output cognition values and frontal/mtl degradation and energy-related values which are stored in xlsx file. Also, calls the plotting function.
    def eval(self, df, data_type='test', exp_type='synthetic', score='MMSE'):
        if exp_type == 'adni':
            #cognition_vec = df['MMSE_norm'].values
            df_join = pd.merge(df, self.output,  how='right', left_on=['RID','Years'], right_on = ['RID','Years'])
            
            df_join['cogsc'] = df_join[f'{score}_norm']
            
            if data_type == 'train':
                mode = 'w'
            else:
                mode = 'a'
            
            outfile = self.snapshot_dir.split("/")[-1].replace("synthetic","rl")
            df_join['cog_diff'] = df_join['cogsc_rl'] - df_join['cogsc']
            with pd.ExcelWriter(f'{self.snapshot_dir}/{outfile}.xlsx', engine="openpyxl", mode=mode) as writer:  
                df_join.to_excel(writer, sheet_name=data_type, index=False)  
            df_join = pd.merge(df, self.output,  how='left', left_on=['RID','Years'], right_on = ['RID','Years'])
            df_join['cogsc'] = df_join[f'{score}_norm']
            
            plot_real(df_join, f'{self.snapshot_dir}/ground_truth_traj_{data_type}.png')
            df_join['cogsc'] = df_join['reg1_info_rl'] + df_join['reg2_info_rl']
            plot_real(df_join, f'{self.snapshot_dir}/rl_traj_common_{data_type}.png')
            
            cog_mae = np.abs(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - df_join[f'{score}_norm']).values.mean()
            cog_mse = np.square(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - df_join[f'{score}_norm']).values.mean()
            
            #reward_vec_rl =  (-np.abs(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - 10)*self.w_lambda - (df_join['reg1_fdg_rl'] + df_join['reg2_fdg_rl'])).values
            
            categories = ['EMCI','CN','LMCI','SMC']
            mae_cat = []
            mse_cat = []
            
            for cat in categories:
                cat_mae = np.abs((df_join[df_join['DX_bl']==cat]['reg1_info_rl'] + df_join[df_join['DX_bl']==cat]['reg2_info_rl'] - df_join[df_join['DX_bl']==cat][f'{score}_norm']).values)
                cat_mse = np.square((df_join[df_join['DX_bl']==cat]['reg1_info_rl'] + df_join[df_join['DX_bl']==cat]['reg2_info_rl'] - df_join[df_join['DX_bl']==cat][f'{score}_norm']).values)
                
                mae_cat.append(cat_mae.mean())
                mse_cat.append(cat_mse.mean())
                
            return cog_mae, mae_cat[0], mae_cat[1], mae_cat[2], mae_cat[3], cog_mse, mse_cat[0], mse_cat[1], mse_cat[2], mse_cat[3], 0, np.mean(self.reward_vec_rl), 0
        else:
            df_join = pd.merge(df, self.output,  how='left', left_on=['RID','Years'], right_on = ['RID','Years'])
            
            cognition_vec = df_join['cogsc'].values
            reward_vec =  (-np.abs(df_join['reg1_info'] + df_join['reg2_info'] - 10)*self.w_lambda - (df_join['reg1_fdg'] + df_join['reg2_fdg'])).values
        
            df_6 = df.groupby('RID').head(7).reset_index(drop=True)
        
            mtl_load = df_6['reg1_info'].values
            ftl_load = df_6['reg2_info'].values
            mtl_load = mtl_load.reshape(-1,7)[:,1:]
            ftl_load = ftl_load.reshape(-1,7)[:,1:]
            cognition = mtl_load + ftl_load

            mtl_energy = df_6['reg1_fdg'].values
            ftl_energy = df_6['reg2_fdg'].values
            mtl_energy = mtl_energy.reshape(-1,7)[:,1:]
            ftl_energy = ftl_energy.reshape(-1,7)[:,1:]
            total_energy = mtl_energy + ftl_energy
            
            mtl_h = None
            ftl_h = None
            
            cognition_vec_rl = df_join['cogsc_rl'].values
            reward_vec_rl =  (-np.abs(df_join['reg1_info_rl'] + df_join['reg2_info_rl'] - 10)*self.w_lambda - (df_join['reg1_fdg_rl'] + df_join['reg2_fdg_rl'])).values
            
            try:
                plot_synthetic(df_join, f'{self.snapshot_dir}/ground_truth_traj_{data_type}.png')
                plot_synthetic(df_join, f'{self.snapshot_dir}/rl_traj_common_{data_type}.png', '_rl')
            except:
                pass
            
            cog_mae = np.abs(cognition_vec_rl - cognition_vec).mean()
            cog_mse = (np.square(cognition_vec_rl - cognition_vec)).mean()
            
            reward_diff = np.mean(reward_vec_rl - reward_vec)
            return cog_mae, 0, 0, 0, 0, cog_mse, 0, 0, 0, 0, reward_diff, np.mean(self.reward_vec_rl), np.mean(reward_vec)
        
        
        
            
        




