# this script can only be run on Python 3.6
from typing import TypeVar, List, Dict, Generic, NamedTuple

FEATURES = Dict[int, float]
ACTION = TypeVar("ACTION", str, FEATURES)


M_A = type(NamedTuple)
M_B = type(Generic[ACTION])
print(type(NamedTuple))
print(type(Generic[ACTION]))
print(M_A)
print(M_B)
class M_C(M_A, M_B): pass


class Samples(NamedTuple, Generic[ACTION], metaclass=M_C):
    mdp_ids: List[str]
    states: List[FEATURES]
    actions: List[ACTION]
    next_actions: List[ACTION]

    def shuffle(self):
        print('shuffle', self.actions)


class MultiStepSamplesInheritance(Samples):
    mdp_ids: List[str]
    states: List[FEATURES]
    actions: List[ACTION]
    next_actions: List[List[ACTION]]


class MultiStepSamples(NamedTuple, Generic[ACTION], Samples, metaclass=M_C):
    mdp_ids: List[str]
    states: List[FEATURES]
    actions: List[ACTION]
    next_actions: List[List[ACTION]]


if __name__ == "__main__":
    # pycharm show warning
    s = Samples(1, 2, 3, 4)
    # pycharm not show warning
    s = Samples(['a'], [{3: 1.0}], ['b'], ['b'])
    # pycharm show warning
    s = Samples(['a'], [{3: 1.0}], ['b'], [{3: 1.0}])
    # pycharm not show warning
    s = Samples(mdp_ids=['a'], next_actions=['b'],
                actions=['b'], states=[{3: 1.0}])
    # pycharm show warning
    s = Samples(mdp_ids=['a'], next_actions=['b'],
                actions=[1], states=[{3: 1.0}])
    s.shuffle()

    # pycharm not show warning, generic cannot be inherited
    s = MultiStepSamplesInheritance(
            mdp_ids=['a'], next_actions=['b'],
            actions=[1], states=[{3: 1.0}]
    )
    # pycharm show warning about next_actions
    s = MultiStepSamples(
        mdp_ids=['a'], next_actions=['b'],
        actions=['b'], states=[{3: 1.0}]
    )
    # pycharm not show warning
    s = MultiStepSamples(
        mdp_ids=['a'], next_actions=[['b']],
        actions=['b'], states=[{3: 1.0}]
    )

    # attribute error, cannot inherit shuffle method from Samples class
    # see: https://stackoverflow.com/questions/50367661/customizing-typing-namedtuple
    print(s.shuffle())