# EduSim
[![PyPI](https://img.shields.io/pypi/v/EduSim)](https://pypi.python.org/pypi/EduSim)
[![Build Status](https://www.travis-ci.org/tswsxk/EduSim.svg?branch=master)](https://www.travis-ci.org/tswsxk/EduSim)
[![Coverage Status](https://coveralls.io/repos/github/tswsxk/EduSim/badge.svg?branch=master)](https://coveralls.io/github/tswsxk/EduSim?branch=master)
[![Coverage Status](https://codecov.io/gh/codecov/EduSim/branch/master/graph/badge.svg)](https://codecov.io/gh/codecov/EduSim)

EduSim is a platform for constructing simulation environments for education recommender systems (ERSs) 
that naturally supports sequential interaction with learners. 
Meanwhile, EduSim allows the creation of new environments that reflect particular aspects of learning elements, 
such as learning behavior of learners, knowledge structure of concepts and so on.

If you are using this package for your research, please cite our paper [1].

Refer to our [website](http://base.ustc.edu.cn/) and [github](https://github.com/bigdata-ustc) for our publications and more projects

## Installation
```bash
pip install EduSim
```

## Quick Start
```python
import gym 
from EduSim import Graph, RandomGraphAgent

env = gym.make('KSS-v0', learner_num=4000)
agent = RandomGraphAgent(Graph("KSS"))
max_episode_num = 1000
n_step = False
max_steps = 20
train = True

episode = 0

while True:
    if max_episode_num is not None and episode > max_episode_num:
        break

    try:
        agent.begin_episode(env.begin_episode())
        episode += 1
    except ValueError:  # pragma: no cover
        break

    # recommend and learn
    if n_step is True:
        # generate a learning path
        learning_path = agent.n_step(max_steps)
        env.n_step(learning_path)
    else:
        # generate a learning path step by step
        for _ in range(max_steps):
            try:
                learning_item = agent.step()
            except ValueError:  # pragma: no cover
                break
            interaction = env.step(learning_item)
            agent.observe(**interaction["performance"])

    # test the learner to see the learning effectiveness
    agent.episode_reward(env.end_episode()["reward"])
    agent.end_episode()

    if train is True:
        agent.tune()
```

## List of Environment

There are three kinds of Environments, which differs in learner capacity growth model:
* Pattern Based Simulators (PBS): the capacity growth model is designed by human experts;
* Data Driven Simulators (DDS): the capacity growth model is learned from real data;
* Hybrid Simulators (HS): the capacity growth model is learned from real data with some expert rule limitation;

We currently provide the following environments:

Name | Kind | Notation
-|-|-
[KSS-v0](docs/Env.md) | PBS | Knowledge Structure based Simulator (KSS), which is used in [1]

To construct your own environment, refer to [Env.md](docs/Env.md)

## Reference
[1] Qi Liu, Shiwei Tong, Chuanren Liu, Hongke Zhao, Enhong Chen, HaipingMa,&ShijinWang.2019.ExploitingCognitiveStructureforAdaptive Learning.InThe 25th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD’19)
```bibtex
@inproceedings{DBLP:conf/kdd/LiuTLZCMW19,
  author    = {Qi Liu and
               Shiwei Tong and
               Chuanren Liu and
               Hongke Zhao and
               Enhong Chen and
               Haiping Ma and
               Shijin Wang},
  title     = {Exploiting Cognitive Structure for Adaptive Learning},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, Anchorage, AK,
               USA, August 4-8, 2019},
  pages     = {627--635},
  year      = {2019},
  crossref  = {DBLP:conf/kdd/2019},
  url       = {https://doi.org/10.1145/3292500.3330922},
  doi       = {10.1145/3292500.3330922},
  timestamp = {Mon, 26 Aug 2019 12:44:14 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/kdd/LiuTLZCMW19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```