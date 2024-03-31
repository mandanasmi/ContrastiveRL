import numpy as np
import os
import wandb


def slurm_infos():
    return {
        'slurm/job_id': os.getenv('SLURM_JOB_ID'),
        'slurm/job_user': os.getenv('SLURM_JOB_USER'),
        'slurm/job_partition': os.getenv('SLURM_JOB_PARTITION'),
        'slurm/cpus_per_node': os.getenv('SLURM_JOB_CPUS_PER_NODE'),
        'slurm/num_nodes': os.getenv('SLURM_JOB_NUM_NODES'),
        'slurm/nodelist': os.getenv('SLURM_JOB_NODELIST'),
        'slurm/cluster_name': os.getenv('SLURM_CLUSTER_NAME'),
        'slurm/array_task_id': os.getenv('SLURM_ARRAY_TASK_ID')
    }


def table_from_array(array):
    data = [(value,) + index for (index, value) in np.ndenumerate(array)]
    columns = ['data'] + [f'x_{i}' for i in range(array.ndim)]
    return wandb.Table(data=data, columns=columns)

def table_from_dict(dictionary):
    data = list(dictionary.items())
    return wandb.Table(data=data, columns=['key', 'value'])

def scatter_from_dicts(x, dictx, y, dicty, transform=lambda x: x, title=None):
    data = []
    for key in (dictx.keys() & dicty.keys()):
        data.append([transform(dictx[key]), transform(dicty[key])])
    table = wandb.Table(data=data, columns=[x, y])
    return wandb.plot.scatter(table, x, y, title=title)