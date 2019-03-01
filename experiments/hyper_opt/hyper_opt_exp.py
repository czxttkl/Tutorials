import unittest
import pickle
import time
import numpy as np
import math
from functools import partial
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, STATUS_FAIL, Trials, base
import os
# os.environ['HYPEROPT_FMIN_SEED'] = '1234'


count = 0
space = {
    "x": hp.uniform('x', -10, 10),
    "y": hp.randint('y', 10),
}


def objective(x, y):
    return - x ** 2 + 2 * x - y ** 2 + y * 5 + math.cos(y) - 3 * math.sin(x)


class TestHyperOpt(unittest.TestCase):

    def method1(self, suggest, num_evals):
        trials = Trials()
        def objective_wrapper(params):
            # global count
            # count += 1
            # if -2 < x < 2:
            #     return {'status': STATUS_FAIL}
            # else:
            # print("evaluate {}-th time: x={}, y={}, loss={}, trial={}".format(
            #     count, x, y, objective(x, y),
            #     len(trials.trials))
            # )
            x, y = params["x"], params["y"]
            return {
                'loss': objective(x, y),
                'status': STATUS_OK,
            }

        best = fmin(
            objective_wrapper,
            space=space,
            algo=suggest,
            max_evals=num_evals,
            trials=trials
        )
        # print("best trial:")
        # print(trials.best_trial)
        return trials.best_trial['result']['loss']
        # print(trials.best_trial)
        # print()
        # print()
        # for i, t in enumerate(trials.trials):
        #     print("{}: {}".format(i, t))


    def method2(self, suggest, num_evals):
        trials = Trials()
        def objective_wrapper(params):
            return {
                'loss': 0,
                'status': STATUS_OK,
            }
        for ne in range(num_evals):
            best = fmin(
                objective_wrapper,
                space=space,
                algo=suggest,
                max_evals=ne + 1,
                trials=trials
            )
            x, y = trials.trials[-1]['misc']['vals']['x'][0], trials.trials[-1]['misc']['vals']['y'][0]
            trials.trials[-1]['result'] = {"status": STATUS_OK, "loss": objective(x, y)}
            # print(trials.trials[-1])
        # print("best trial:")
        # print(trials.best_trial)
        return trials.best_trial['result']['loss']

    def test(self):
        num_evals = 16
        test_times = 200

        tpe_suggest = partial(tpe.suggest, n_startup_jobs=10)
        method1_tpe_losses = [TestHyperOpt().method1(tpe_suggest, num_evals)
                              for _ in range(test_times)]
        method1_tpe_loss = np.mean(method1_tpe_losses)
        method1_tpe_std = np.std(method1_tpe_losses)

        rand_suggest = rand.suggest
        method1_rand_losses = [TestHyperOpt().method1(rand_suggest, num_evals)
                               for _ in range(test_times)]
        method1_rand_loss = np.mean(method1_rand_losses)
        method1_rand_std = np.std(method1_rand_losses)

        tpe_suggest = partial(tpe.suggest, n_startup_jobs=10)
        method2_tpe_losses = [TestHyperOpt().method2(tpe_suggest, num_evals)
                              for _ in range(test_times)]
        method2_tpe_loss = np.mean(method2_tpe_losses)
        method2_tpe_std = np.std(method2_tpe_losses)

        rand_suggest = rand.suggest
        method2_rand_losses = [TestHyperOpt().method2(rand_suggest, num_evals)
                               for _ in range(test_times)]
        method2_rand_loss = np.mean(method2_rand_losses)
        method2_rand_std = np.std(method2_rand_losses)

        print("method1_tpe_loss: {}+-{}, method1_rand_loss: {}+-{}".format(
            method1_tpe_loss, method1_tpe_std, method1_rand_loss, method1_rand_std
        ))
        print("method2_tpe_loss: {}+-{}, method2_rand_loss: {}+-{}".format(
            method2_tpe_loss, method2_tpe_std, method2_rand_loss, method2_rand_std
        ))
        assert method1_tpe_loss < method1_rand_loss
        assert method1_tpe_loss < method2_rand_loss
        assert method2_tpe_loss < method1_rand_loss
        assert method2_tpe_loss < method2_rand_loss


if __name__ == "__main__":
    TestHyperOpt().test()



# rstate = np.random.RandomState()
# domain = base.Domain(objective, space, pass_expr_memo_ctrl=False)
# new_trials = tpe.suggest(new_ids, domain, trials, 1234)