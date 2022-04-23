import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import garage
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO, VPG, DDPG, PPO
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy, GaussianMLPPolicy, GaussianLSTMPolicy
from garage.experiment import LocalTFRunner
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.experiment import Snapshotter
from garage.sampler.utils import rollout

import numpy as np
import scipy
import pandas as pd

from envs import BrainEnv
from eval import EvalPolicy

def main(args):
    global adj, cog_init, log_dir

    gamma = args.gamma
    max_time_steps = args.trainsteps
    epochs = args.epochs
    batch_size=args.batch_size
    gamma_type = args.gammatype
    action_type = args.actiontype
    scale_state = args.scale
    score = args.score

    name = args.filename.split(".xls")[0]

    adj = np.array([[0,1],[1,0]]) 
    H = np.diag(np.sum(adj, axis=1)) - adj
        
    if args.cog_init == 'full':
        cog_init = np.array([args.cog_mtl,10.0 - args.cog_mtl])
    else:
        cog_init = [None, None]
    
                
    name = name.replace("_frontal_hippo_atleast3","")
    
    log_dir = f'../models/{name}_{args.algo}_{max_time_steps}_{cog_init[0]}_{action_type}_{gamma_type}_{gamma}_{epochs}_{batch_size}_{args.action_limit}_{args.cog_type}_{args.cog_init}_{args.discount}_{args.w_lambda}_{args.trainsteps}_{args.energy_model}_{args.score}_{args.network}'
    output_dir = log_dir.replace("../models","../output") 

    if args.cog_type == 'variable':
        log_dir = f'../models/{name}_{args.algo}_{max_time_steps}_{10.0}_{action_type}_{gamma_type}_{gamma}_{epochs}_{batch_size}_{args.action_limit}_{args.cog_type}_{args.cog_init}_{args.discount}_{args.w_lambda}_{args.trainsteps}_{args.energy_model}_{args.score}_{args.network}'


    if args.eval:
        cog_init = np.array([args.cog_mtl,10.0 - args.cog_mtl])
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        

    @wrap_experiment(log_dir=log_dir, archive_launch_repo = False, use_existing_dir=True)
    def train_policy(ctxt = None, seed=1, n_epochs=50, batch_size=1000, action_type='delta', gamma_type='variable', gamma=2.1, max_time_steps=10, algo_name='TRPO'):
        """Train policy with Brain environment.
        """
        set_seed(seed)
        
        with LocalTFRunner(snapshot_config=ctxt) as trainer:
            gamma_val = train_data[2]
            if gamma_type == 'variable':
                gamma_init = train_data[2]
                alpha2_new_init = train_data[1]
            else:    
                gamma_init = np.ones(len(train_data[0]))*gamma
                alpha2_new_init = gamma_val*train_data[1]/gamma
            
            if args.normalize:
                env = normalize(GarageEnv(BrainEnv(max_time_steps=max_time_steps+1, alpha1_init=train_data[0], alpha2_init=alpha2_new_init, beta_init=train_data[3], 
                                                   gamma_init=gamma_init, X_V_init=train_data[5], 
                                                   D_init=train_data[4], cog_type=args.cog_type, cog_init=train_data[-1], adj=adj, action_limit=args.action_limit, 
                                                   w_lambda=args.w_lambda, gamma_type=gamma_type, action_type=action_type, scale=False, energy_model=args.energy_model)), 
                                                   normalize_obs=True)
            else:
                env = GarageEnv(BrainEnv(max_time_steps=max_time_steps+1, alpha1_init=train_data[0], alpha2_init=alpha2_new_init, beta_init=train_data[3], gamma_init=gamma_init, X_V_init=train_data[5], \
                                            D_init=train_data[4], cog_type=args.cog_type, cog_init=train_data[-1], adj=adj, action_limit=args.action_limit, 
                                            w_lambda=args.w_lambda, gamma_type=gamma_type, action_type=action_type, scale=scale_state, energy_model = args.energy_model))
                
            #policy = GaussianLSTMPolicy(name='policy', learn_std=False, env_spec=env.spec)
            policy = GaussianMLPPolicy(name='policy',env_spec=env.spec, hidden_sizes=[args.network, args.network], max_std=None, adaptive_std=False, std_share_network=False, output_nonlinearity=None)
            
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            
            print(algo_name)
            if algo_name == 'TRPO':
                algo = TRPO(env_spec=env.spec,
                            policy=policy,
                            baseline=baseline,
                            discount=args.discount,
                            gae_lambda=0.97,
                            lr_clip_range=0.2,
                            policy_ent_coeff=0.0,
                            max_path_length=max_time_steps)
            elif algo_name == 'PPO':
                algo = PPO(
                        env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=max_time_steps,
                        discount=1.00,
                        gae_lambda=0.95,
                        lr_clip_range=0.2,
                        optimizer_args=dict(
                            batch_size=10,
                            max_epochs=4,
                        ),
                        stop_entropy_gradient=True,
                        entropy_method='max',
                        policy_ent_coeff=0.02,
                        center_adv=False,
                    )
            elif algo_name == 'VPG':
                algo = VPG(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    discount=0.99,
                    optimizer_args=dict(learning_rate=0.005, ))
            
            trainer.setup(algo, env)
            trainer.train(n_epochs=n_epochs, batch_size=batch_size)
            
            return policy
            
    def load_data_syn(filename):
        df = {}
        for i in ['train','valid','test']:
            df[i] = pd.read_excel(filename, sheet_name=i)
        return df['train'], df['valid'], df['test']


    def load_data(filename):
        df = {}
        for i in ['train','valid','test']:
            df_temp = pd.read_excel(filename, sheet_name=i)
            df_params = pd.read_excel(filename.replace(".xls","_parameters.xls"), sheet_name=f'{score}_norm_PTGENDER_APOEPOS')
            df[i] = df_temp.set_index('RID').join(df_params.set_index('RID'), on='RID', how='inner', rsuffix='_right')
            df[i].reset_index(level=0, inplace=True)
        return df['train'], df['valid'], df['test']


    def extract_params(df):
        RIDs = np.unique(df['RID'].values)
        df['D1_1'] = -df.groupby('RID')['reg1_av45'].diff(-1)
        df['D1_2'] = -df.groupby('RID')['reg2_av45'].diff(-1)
        df_first = df.groupby('RID', as_index=False).first()

        df_first = df_first[df_first.RID.isin(RIDs[:])]
        
        alpha1_init = df_first['alpha1'].values
        alpha2_init = df_first['alpha2'].values
        beta_init = df_first['beta'].values
        
        gamma_init = df_first['gamma'].values
        D1_init = df_first[['D1_1','D1_2']].values
        
        X_V_init = df_first[['reg1_mri','reg2_mri']].values
        FDG_init = df_first[['reg1_fdg','reg2_fdg']].values
        
        info_init = np.array([cog_init]*len(RIDs)) if args.cog_init != 'baseline' else np.array([df_first[f'{score}_norm'].values/2, df_first[f'{score}_norm'].values]/2).T#df_first[['reg1_info','reg2_info']].values    
        print("Info_init",info_init)
        end_times = df.groupby('RID', as_index=False).last()['Years'].values
        t_array = df_first['tpo'].values
        
        return [alpha1_init, alpha2_init, gamma_init, beta_init, D1_init, X_V_init, end_times, RIDs, info_init], FDG_init, t_array

    def extract_data(df, H, gamma):
        RIDs = np.unique(df['RID'].values)
        df_first = df.groupby('RID', as_index=False).first()
        beta_init = df_first['beta_estm'].values
        X_V_init = df_first[['mri_FRONT_norm','mri_HIPPO_norm']].values
        end_times = df.groupby('RID', as_index=False).last()['Years'].values
        gamma_init = np.ones(len(RIDs))*gamma
        alpha1_init = df_first['alpha1_estm'].values
        alpha2_init = df_first['alpha2_gamma_estm'].values/gamma
        t_array = df_first['tpo_estm'].values
        D1_init = df_first[['FRONTAL_SUVR','HIPPOCAMPAL_SUVR']].values
        info_init = np.array([cog_init]*len(RIDs)) if args.cog_init != 'baseline' else np.array([df_first[f'{score}_norm'].values, 0*df_first[f'{score}_norm'].values/2]).T 
        lamb, U = np.linalg.eig(H)

        D0_init = X_V_init.copy()
        
        for i,beta in enumerate(beta_init):
            t = t_array[i]        
            diag_array = np.diag([lamb[0]*np.exp(-lamb[0]*beta*t)/(1-np.exp(-lamb[0]*beta*t)), 1/(beta*t)])
            phi_t = D1_init[i]
            mult = U.dot(diag_array).dot(U.T)
            mult = U.dot(diag_array).dot(U.T)
            D0_init[i,:] = beta*(mult).dot(phi_t.T)   

        return [alpha1_init, alpha2_init, gamma_init, beta_init, D0_init, X_V_init, end_times, RIDs, info_init]
        
    def calculate_D(D1_init, beta_init, alpha1_init, alpha2_init, H, fdg, tpo):
        D_init = D1_init.copy()
        
        for i in range(D1_init.shape[0]):        
            D_old = np.linalg.inv(np.exp(-beta_init[i]*H)).dot(D1_init[i,:])
            D_init[i,:] = D_old

        return D_init

    if args.datatype == 'synthetic':
        filename = f'../dataset/processed/{args.filename}'
        train_df, valid_df, test_df = load_data_syn(filename=filename)
        global train_data
        train_data, train_fdg, train_tpo = extract_params(train_df)
        valid_data, valid_fdg, valid_tpo = extract_params(valid_df)
        test_data, test_fdg, test_tpo = extract_params(test_df)
        train_data[4] = calculate_D(train_data[4].copy(), train_data[3], train_data[0], train_data[1], H, train_fdg, train_tpo)
        test_data[4] = calculate_D(test_data[4].copy(), test_data[3], test_data[0], test_data[1], H, test_fdg, test_tpo)
        valid_data[4] = calculate_D(valid_data[4].copy(), valid_data[3], valid_data[0], valid_data[1], H, valid_fdg, valid_tpo)
        
    else:
        filename = f'../dataset/processed/{args.filename}'
        train_df, valid_df, test_df = load_data(filename=filename)
        
        train_data = extract_data(train_df, H, gamma)
        valid_data = extract_data(valid_df, H, gamma)
        test_data = extract_data(test_df, H, gamma)

    tf.compat.v1.reset_default_graph()

    if not args.eval:
        policy = train_policy(seed=args.seed, n_epochs=epochs, batch_size=batch_size, action_type=action_type, gamma_type=gamma_type, gamma=gamma, max_time_steps=max_time_steps, algo_name=args.algo)

    #test accuracy
    evaluator = EvalPolicy(T=11, snapshot_dir=output_dir, log_dir=log_dir, gamma=gamma, gamma_type=gamma_type, cog_init=cog_init, adj=adj, action_type=action_type, action_limit=args.action_limit, w_lambda=args.w_lambda, energy_model=args.energy_model)
    evaluator.simulate(train_data, 'train', scale_state, args.normalize)
    train_mae, train_mae_emci, train_mae_cn, train_mae_lmci, train_mae_smc, train_mse, train_mse_emci, train_mse_cn, train_mse_lmci, train_mse_smc, train_reward_gain, train_reward_rl, train_reward = evaluator.eval(train_df, 'train', args.datatype, score)

    evaluator.simulate(valid_data, 'valid', scale_state, args.normalize)
    valid_mae, valid_mae_emci, valid_mae_cn, valid_mae_lmci, valid_mae_smc, valid_mse, valid_mse_emci, valid_mse_cn, valid_mse_lmci, valid_mse_smc, valid_reward_gain, valid_reward_rl, valid_reward = evaluator.eval(valid_df, 'valid', args.datatype, score)

    evaluator.simulate(test_data, 'test', scale_state, args.normalize)
    test_mae, test_mae_emci, test_mae_cn, test_mae_lmci, test_mae_smc, test_mse, test_mse_emci, test_mse_cn, test_mse_lmci, test_mse_smc, test_reward_gain, test_reward_rl, test_reward = evaluator.eval(test_df, 'test', args.datatype, score)

    print(train_mae, valid_mae, test_mae)
        
    results_df = pd.DataFrame({'category':'APOE', 'name':name, 'gamma':gamma,'gamma_type':gamma_type, 'epochs':epochs, 'batch_size':batch_size, 'cog_mtl':args.cog_mtl, 
                                'discount':args.discount, 'max_time_steps':args.trainsteps, 'w_lambda':args.w_lambda, 'action_lim':args.action_limit, 'cog_init':args.cog_init, 'cog_type':args.cog_type,
                                'model':args.energy_model, 'score':args.score, 'network':args.network, 'algo':args.algo,
                                'train_mae':train_mae, 'valid_mae':valid_mae, 'test_mae':test_mae,
                                'train_mse':train_mse, 'valid_mse':valid_mse, 'test_mse':test_mse,
                                'train_mae_emci':train_mae_emci, 'valid_mae_emci':valid_mae_emci, 'test_mae_emci':test_mae_emci,
                                'train_mae_cn':train_mae_cn, 'valid_mae_cn':valid_mae_cn, 'test_mae_cn':test_mae_cn,
                                'train_mae_lmci':train_mae_lmci, 'valid_mae_lmci':valid_mae_lmci, 'test_mae_lmci':test_mae_lmci,
                                'train_mae_smc':train_mae_smc, 'valid_mae_smc':valid_mae_smc, 'test_mae_smc':test_mae_smc,
                                'train_mse_emci':train_mse_emci, 'valid_mse_emci':valid_mse_emci, 'test_mse_emci':test_mse_emci,
                                'train_mse_cn':train_mse_cn, 'valid_mse_cn':valid_mse_cn, 'test_mse_cn':test_mse_cn,
                                'train_mse_lmci':train_mse_lmci, 'valid_mse_lmci':valid_mse_lmci, 'test_mse_lmci':test_mse_lmci,
                                'train_mse_smc':train_mse_smc, 'valid_mse_smc':valid_mse_smc, 'test_mse_smc':test_mse_smc,
                                'train_reward_rl':train_reward_rl, 'valid_reward_rl':valid_reward_rl, 'test_reward_rl':test_reward_rl,
                                'train_reward':train_reward, 'valid_reward':valid_reward, 'test_reward':test_reward
                            }, index=[0])
    
    # saving results to output file
    output_filename = f'../output/results_{args.datatype}.csv'
    if os.path.isfile(output_filename):
        results_df.to_csv(output_filename, mode='a', header=False, index=False)
    else:    
        results_df.to_csv(output_filename, mode='w', header=True, index=False)
   
