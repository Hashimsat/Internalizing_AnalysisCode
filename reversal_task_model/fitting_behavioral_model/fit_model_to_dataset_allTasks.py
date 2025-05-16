
import sys
from pathlib import Path
import os

#Pymc
import pymc3 as pm
import theano
import theano.tensor as T

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'

import numpy as np
import pickle
import imp
import sys
import pandas as pd
import datetime
from reversal_task_model.model_code import stats as stat_func

# import model_base
from reversal_task_model.model_code.model_base import *

def invlogit(p):
    return 1 / (1 + np.exp(-p))

# set up the directory structure
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / 'model_code'))

def fit_model(args,params, data):

    B_max = 10
    nonlinear_indicator = 0 # mag diff scaled

    # ------------
    # Initialize models
    # ------------
    if (args.task_type=='MagVersion'):
        if args.modelname=='1':
            import models_1 as model_specific

        if any(digit in args.modelname for digit in ['2', '3', '4', '5', '6', '7','9']):
            import models_2thr9_11 as model_specific

        if any(digit in args.modelname for digit in ['8','10']):
            import models_9_12 as model_specific

    elif (args.task_type=='MagVersionBoth'):
        if args.modelname=='1':
            import models_1 as model_specific

        if any(digit in args.modelname for digit in ['2', '3', '4', '5', '6', '7','8','11']):
            import models_2thr9_11 as model_specific

        if any(digit in args.modelname for digit in ['9','12']):
            import models_9_12 as model_specific

        if any(digit in args.modelname for digit in ['10','13']):
            import models_10_13 as model_specific

    elif args.task_type=='NoMagVersion':
        if any(digit in args.modelname for digit in ['1', '2', '3', '4', '5', '6']):
            import models_1thr6_ReducedTask as model_specific

        if any(digit in args.modelname for digit in ['7','8']):
            import models_7_8_reduced as model_specific


    u_covariate_mask = None
    mask_name=''

    # --------------
    # Extract data
    # --------------
    # extract relevant data from the dataset to be used by the model
    subj_indices = slice(0, data['participants_choice'].shape[1])

    Y = {}
    Y['participants_choice'] = data['participants_choice'][:, subj_indices]

    C = {}
    for trait in ['Bi1item_w_j_scaled', 'Bi2item_w_j_scaled', 'Bi3item_w_j_scaled',
                  'PSWQ_scaled', 'MASQ_scaled',
                  'BDI_scaled', 'STAI_scaled']:
        C[trait + '_both'] = np.array(list(data[trait]))

    Subjects = data['subjectID']

    if args.exp == 1: # dataset without reward mags
        includes_subjs_with_one_task = False

        # task specific data
        X = {}
        for var in ['outcomes_c_flipped', 'stabvol', 'rewpain']:
            X[var] = data[var][:, subj_indices]
        X['NN'] = X[var].shape[1]
        X['Nboth'] = X['NN']

    if args.exp==2:  # Dataset with reward magnitudes
        includes_subjs_with_one_task = False

        # task specific data
        X = {}
        for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
            X[var]=data[var][:,subj_indices]
        X['NN']=X[var].shape[1]
        X['Nboth']=data['Nboth']

    # ----------------
    # Create the base and likelihood model
    # -----------------
    # Create base model (i.e. prior), embedding factors into the priors
    print('compiling base model')

    idx_first_reward_pain = np.min([pi for (pi, p) in enumerate(params) if 'rew' in p]) if any(
        'rew' in p for p in params) else 0

    model = create_model_base(X,Y,C, # Changed here
                              params=params,
                              K=len(params),
                              Konetask=idx_first_reward_pain,
                              rew_slice=slice(0,idx_first_reward_pain),
                              pain_slice=slice(0,idx_first_reward_pain),
                              split_by_reward=False,
                              includes_subjs_with_one_task=includes_subjs_with_one_task,
                              covariate=args.covariate,
                              hierarchical=args.hierarchical,
                              covv='diag',
                              coding='deviance',
                              u_covariate_mask=u_covariate_mask,
                              one_task_only=args.one_task_only,
                              task_type=args.task_type,)

    # Create likelihood model
    print('compiling specific model')
    model = model_specific.combined_prior_model_to_choice_model(X,Y,
                                                                param_names=params,
                                                                model=model,
                                                             save_state_variables=False,
                                                                B_max=B_max,nonlinear_indicator=nonlinear_indicator)

    # Save name
    print('saving')
    now = datetime.datetime.now()
    filename='prl2_' +args.task_type +'_model='+args.modelname+'try_one_task_'+str(args.one_task_only)+'_covariate='+args.covariate+'_date='+str(now.year)+ \
             '_'+str(now.month)+'_'+str(now.day)+'_samples='+str(args.steps)+'_seed='+str(args.seed)+'_exp='+str(args.exp)

    #     # Save empty placeholder
    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:

        pickle.dump({}, buff)
    #

    # ---------------
    # Fit the model
    # ---------------
    with model:

        MAP = {}

        step=pm.HamiltonianMC(target_accept=.95)
        # step=pm.HamiltonianMC(target_accept=.90)

        print('sampling ...')
        trace = pm.sample(args.steps,step=step,chains=3,tune=args.steps_tune,random_seed=args.seed) # cores = 4

        ppc = pm.sample_ppc(trace,500)

        loo= stat_func.loo(trace,model=model,progressbar=True)

        print(loo)

    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        pickle.dump({'model': model,'trace':trace,'ppc':ppc,'MAP':MAP, 'subjectID':Subjects}, buff)

    return loo
    #

