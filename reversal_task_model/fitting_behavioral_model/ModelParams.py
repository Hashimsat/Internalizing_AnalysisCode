# class that defines parameters for all the models, given the type of task


class ModelParams:

    def __init__(self,model_vars):
        self.task_type = model_vars.task_type
        self.modelname = model_vars.modelname

    def get_params(self):
        # Define model parameters for each task and model
        task_params = {
            'MagVersion': {
                '1':['lr_baseline','lr_stabvol','Gamma_baseline','Gamma_stabvol','Binv_baseline','Binv_stabvol'],
                '2': ['lr_baseline','lr_stabvol','Amix_baseline','Amix_stabvol','Binv_baseline','Binv_stabvol',],
                '3': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_stabvol',
                      'Binv_baseline','Binv_stabvol'],
                '4': ['lr_baseline','lr_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',],
                '5': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol'],

                '6': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'mag_baseline'],
                '7': ['lr_baseline', 'lr_goodbad', 'lr_stabvol', 'lr_goodbad_stabvol',
                      'Amix_baseline', 'Amix_goodbad', 'Amix_stabvol', 'Amix_goodbad_stabvol',
                      'Binv_baseline', 'Binv_goodbad', 'Binv_stabvol', 'Binv_goodbad_stabvol',
                      'mag_baseline',
                      'eps_baseline'],
                '8': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'mag_baseline',
                      'decay_baseline','decay_stabvol'],
                '9': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'lr_c_baseline',
                       'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'Bc_baseline',
                       'mag_baseline'],
                '10': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'lr_c_baseline',
                       'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'Bc_baseline',
                       'mag_baseline',
                       'decay_baseline','decay_stabvol'],
            },


            'MagVersionBoth': {
                '1': ['lr_baseline','lr_stabvol',
                      'Gamma_baseline','Gamma_stabvol',
                      'Binv_baseline','Binv_stabvol',
                      'lr_rewpain','lr_rewpain_stabvol',
                      'Gamma_rewpain','Gamma_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_stabvol',
                      ],
                '2': ['lr_baseline','lr_stabvol',
                      'Amix_baseline','Amix_stabvol',
                      'Binv_baseline','Binv_stabvol',
                      'lr_rewpain','lr_rewpain_stabvol',
                      'Amix_rewpain','Amix_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_stabvol',
                      ],
                '3':  ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'Amix_baseline','Amix_stabvol',
                       'Binv_baseline','Binv_stabvol',
                       'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                       'Amix_rewpain','Amix_rewpain_stabvol',
                       'Binv_rewpain','Binv_rewpain_stabvol',
                       ],
                '4': ['lr_baseline','lr_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'lr_rewpain','lr_rewpain_stabvol',
                      'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol'
                      ],
                '5': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                      'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol'
                      ],
                '6': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol','lr_rewpain_goodbad_stabvol',
                      'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol','Amix_rewpain_goodbad_stabvol',
                      'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol','Binv_rewpain_goodbad_stabvol'
                      ] ,

                '7': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'mag_baseline',
                      'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                      'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                      'mag_rewpain'
                      ],
                '8': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'mag_baseline',
                      'eps_baseline',
                      'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                      'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                      'mag_rewpain',
                      'eps_rewpain'
                      ],
                '9': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'mag_baseline',
                      'decay_baseline','decay_stabvol',
                      'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                      'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                      'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                      'mag_rewpain',
                      'decay_rewpain','decay_rewpain_stabvol',
                      ],
                '10': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'mag_baseline',
                       'decay_baseline','decay_stabvol',
                       'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                       'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                       'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                       'mag_rewpain',
                       'decay_rewpain','decay_rewpain_stabvol',
                       ],
                '11': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'lr_c_baseline',
                       'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'Bc_baseline',
                       'mag_baseline',
                       'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                       'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                       'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                       'Bc_rewpain',
                       'mag_rewpain'
                       ],

                '12': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'lr_c_baseline',
                       'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'Bc_baseline',
                       'mag_baseline',
                       'decay_baseline','decay_stabvol',
                       'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                       'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                       'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                       'Bc_rewpain',
                       'mag_rewpain',
                       'decay_rewpain','decay_rewpain_stabvol',
                       ],
                '13': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'lr_c_baseline',
                       'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'Bc_baseline',
                       'mag_baseline',
                       'decay_baseline','decay_stabvol',
                       'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
                       'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
                       'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
                       'Bc_rewpain',
                       'mag_rewpain',
                       'decay_rewpain','decay_rewpain_stabvol',
                       ],
            },


            'NoMagVersion': {
                '1': ['lr_baseline', 'lr_stabvol',
                      'Binv_baseline', 'Binv_stabvol'
                      ],
                '2': ['lr_baseline','lr_goodbad','lr_stabvol',
                      'Binv_baseline','Binv_stabvol'
                      ],
                '3': ['lr_baseline','lr_goodbad','lr_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol'
                      ],
                '4': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol'
                      ],
                '5': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol'
                      ],
                '6': ['lr_baseline', 'lr_goodbad', 'lr_stabvol', 'lr_goodbad_stabvol',
                      'lr_c_baseline',
                      'Binv_baseline', 'Binv_goodbad', 'Binv_stabvol', 'Binv_goodbad_stabvol',
                      'Bc_baseline',
                      ],
                '7': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                      'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                      'decay_baseline','decay_stabvol'
                      ],
                '8': ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                       'lr_c_baseline',
                       'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
                       'Bc_baseline',
                       'decay_baseline','decay_stabvol',
                       ],
            }
        }
        return task_params[self.task_type][self.modelname]


def extract_model_numbers(args):
    if (args.task_type == 'MagVersion'):
        args.one_task_only = True
        models = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # models from 1 to 10

    elif (args.task_type == 'MagVersionBoth'):
        args.one_task_only = False
        models = [str(i) for i in range(1, 13)]

    elif (args.task_type == 'NoMagVersion'):
        args.one_task_only = True
        models = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

    return args, models