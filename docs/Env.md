# Env

## Construct your own simulator

### Change Learner Model

#### Inherited from existing models

##### KSS
A key concern for building your own environment model is how to define your own capacity growth model in Environment.

To do so, a simple way is to override the `learn` function in Learner 
and rewrite the `generate_learners` function in Environment.

For Example, `Learner` of KSS/Learner is:

```python
from EduSim.Envs.Learner import Learner as BaseLearner

class Learner(BaseLearner):
    def __init__(*args, **kwargs):
        ...

    def learn(self, learning_item: int):
        structure = self.structure
        a = self._state

        if self.learning_history:
            if learning_item not in influence_control(
                    structure, a, self.learning_history[-1], allow_shortcut=False, target=self._target,
            )[0]:
                return

        assert isinstance(learning_item, int), learning_item
        self.learning_history.append(learning_item)

        # capacity growth function
        discount = math.exp(sum([(5 - a[node]) for node in structure.predecessors(learning_item)] + [0]))
        ratio = 1 / discount
        inc = (5 - a[learning_item]) * ratio * 0.5

        def _promote(_ind, _inc):
            a[_ind] += _inc
            if a[_ind] > 5:
                a[_ind] = 5
            for node in structure.successors(_ind):
                _promote(node, _inc * 0.5)

        _promote(learning_item, inc)
    ...
```
And you can implement a new `Learner` class inheriting KSS/Learner and override the `learn` function:
```python
from EduSim.Envs.KSS.Learner import Learner
class NewLearner(Learner):
    def learn(self, learning_item: int):
        ...
```
After that, you need to override `generate_learners` in Environment, that is 
```python
from EduSim.Envs.KSS import KSS

class newKSS(KSS):
    def generate_learners(self, learner_num, step=20):
        learner = NewLearner(...)
        ...
```