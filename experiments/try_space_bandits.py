from space_bandits import NeuralBandits
import numpy as np

num_actions = 2
num_features = 2

def objective(context, action):
    return context[0] + context[1] + context[0] * context[1]

model = NeuralBandits(num_actions, num_features, do_scaling=False, batch_size=2, initial_pulls=2, show_training=True, training_freq_network=3, freq_summary=50)
# model.update(np.array([0.0, 1.0]), 0, 0.0)
# model.update(np.array([0.0]), 0, 0.0)

#
for i in range(120):
    context = np.random.randint(0, 10, num_features) * 1.0
    # context = np.array([1.0])
    action = np.random.randint(0, num_actions)
    reward = objective(context, action)
    print(context, action, reward)
    model.update(context, action, reward)
    # model.update(np.array([1.1,2.,3.]), 1, 1.)
    # model.update(np.array([3.,2.3, 1.2]), 0, 2.)
    # model.update(np.array([3.,2.,3.2]), 1, 3.)
#
# model.update(np.array([1.0001,2.,3.], dtype=float), 1, 1.)
# model.update(np.array([3.,2.,1.], dtype=float), 0, 2.)
# model.update(np.array([3.,2.00001,3.], dtype=float), 1, 3.)
#
# model.update(np.array([2.,2.,3.], dtype=float), 1, 1.)
# model.update(np.array([3.,2.,1.], dtype=float), 0, 2.)
# model.update(np.array([1.,2.00001,1.], dtype=float), 1, 3.)

context = np.array([1.0, 3.0])
print(model.expected_values(context))
print(model.action(context))
print(model.get_representation(context))