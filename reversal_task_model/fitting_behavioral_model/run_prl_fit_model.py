# Main code that fits models to the data, based on what task type and model is selected
# Python env: 'reversal_learning_env

import sys
import sys
import os
from pathlib import Path

#Pymc
import pymc3 as pm
import theano
import theano.tensor as T
from fit_model_to_dataset_allTasks import fit_model

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'

from reversal_task_model.model_code.model_base import *
import ModelParams

import argparse

def main():
    # This python function is a wrapper used to fit behavioral models to data.

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--modelname', '-m', type=str, default='11')
    # parser.add_argument('--steps', '-st', type=int, default=2500)
    # parser.add_argument('--steps_tune', '-stt', type=int, default=1500)

    parser.add_argument('--steps', '-st', type=int, default=2000)
    parser.add_argument('--steps_tune', '-stt', type=int, default=1200)

    parser.add_argument('--covariate', '-c', type=str, default='Bi3itemCDM')
    parser.add_argument('--hierarchical', '-hh', type=str, default='True')
    parser.add_argument('--task', '-tt', type=str, default='both')
    parser.add_argument('--subset', '-sub', type=str, default='all')
    parser.add_argument('--covariatemask', '-cm', type=str, default='None')
    parser.add_argument('--exp', '-e', type=int, default=2)
    parser.add_argument('--task_type', '-ttype', type=str, default='MagVersion') #MagVersion, MagVersionBoth or NoMagVersion
    parser.add_argument('--iterate_models', '-itm', type=bool, default=True)

    args = parser.parse_args()

    # as a redundancy, we set the experiment number here too
    if args.task_type == 'NoMagVersion':
        args.exp = 1
    elif args.task_type == 'MagVersion' or args.task_type == 'MagVersionBoth':
        args.exp = 2

    # extract data
    if args.exp == 1: # Experiment with no reward magnitudes
        pkl_path = '../../data/reversal_task/prl_nomag_data_model_alligned.pkl'

    elif args.exp==2: # reversal learning task with reward magnitudes, either withor without loss domain
        if args.task_type == 'MagVersion': # task that only had a reward domain
            pkl_path = '../../data/reversal_task/prl_rewardmag_data_model_alligned.pkl'

        elif args.task_type == 'MagVersionBoth': # task that had both reward and loss domain
            pkl_path = '../../data/reversal_task/prl_rewardloss_data_model_alligned.pkl'

    with open(pkl_path,'rb') as f:
            data = pickle.load(f)

    # if we want to run all models one by one or a specific model on data
    print(args)

    if (args.iterate_models == False):
        args.one_task_only = True if args.task_type == 'NoMagVersion' else False
        model_params = ModelParams.ModelParams(args)
        params = model_params.get_params()
        fit_model(args, params, data)

    # First extract model types we want to run
    elif (args.iterate_models == True):

        if args.task_type == 'MagVersion':
            args.one_task_only = True
            # models = [str(i) for i in range(1, 10)]  # for model comparison, run model 1 to model 10
            models = ['9']   # winning model for this experiment
        elif args.task_type == 'MagVersionBoth':
            args.one_task_only = False
            # models = [str(i) for i in range(1, 13)]
            models = [ '12',]
        elif args.task_type == 'NoMagVersion':
            args.one_task_only = True
            # models = [str(i) for i in range(1, 8)]
            models = [ '6', ]

        # Fit all the models one by one on the data
        for i,model in enumerate(models):
            args.modelname = model
            print('Running Model: ',args.modelname)

            model_params = ModelParams.ModelParams(args)
            params = model_params.get_params()
            print(params)
            fit_model(args,params,data)


if __name__=='__main__':
        main()





