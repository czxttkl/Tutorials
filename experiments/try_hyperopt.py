from hyperopt import fmin, tpe, space_eval
from hyperopt import hp, Trials, STATUS_OK, STATUS_FAIL, STATUS_RUNNING, JOB_STATE_DONE
from hyperopt.fmin import generate_trial
from functools import partial
import time
import json


def objective_function(params):
    print("sleep 5 seconds")
    time.sleep(5)
    return 0

# space = {
#     "executor_memory": hp.choice("executor_memory", list(range(1, 17))), # unit: 4 * g, range: 4 ~ 64
#     "offheap_size": hp.choice("offheap_size", list(range(1, 17))),  # unit: 4 * g, range: 4 ~ 64
#     "shuffle_partitions": hp.choice("shuffle_partitions", list(range(1, 17))),   # unit: 1000, range: 1k ~ 16k
#     "split_size": hp.choice("split_size", list(range(1, 21))) # unit: 1024 * 1024 * 32 (32mb), range: 32 ~ 640mb
# }

space = {
    "executor_memory": hp.randint("executor_memory", 17), # unit: 4 * g, range: 4 ~ 64
    "offheap_size": hp.randint("offheap_size", 17),  # unit: 4 * g, range: 4 ~ 64
    "shuffle_partitions": hp.randint("shuffle_partitions", 17),   # unit: 1000, range: 1k ~ 16k
    "split_size": hp.randint("split_size", 20)  # unit: 1024 * 1024 * 32 (32mb), range: 32 ~ 640mb
}



num_random_startup_jobs = 1
num_evals = 4
trials = Trials()
new_trials = [{'state': JOB_STATE_DONE, 'tid': 0, 'spec': None, 'result': {'status': 'ok', 'loss': 11.859982881944445}, 'misc': {'tid': 0, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'workdir': None, 'idxs': {'executor_memory': [0], 'offheap_size': [0], 'shuffle_partitions': [0], 'split_size': [0]}, 'vals': {'executor_memory': [7], 'offheap_size': [1], 'shuffle_partitions': [16], 'split_size': [15]}}, 'exp_key': None, 'owner': None, 'version': 0, 'book_time': None, 'refresh_time': None}]
trials.insert_trial_docs(new_trials)
trials.refresh()


tpe_suggest = partial(tpe.suggest, n_startup_jobs=num_random_startup_jobs)
best = fmin(
    objective_function,
    space=space,
    algo=tpe_suggest,
    max_evals=num_evals,
    trials=trials
)
trials.best_trial['result']['loss']



# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}



