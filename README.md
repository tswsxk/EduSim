# EduSim
[![PyPI](https://img.shields.io/pypi/v/EduSim)](https://pypi.python.org/pypi/EduSim)
[![Build Status](https://www.travis-ci.org/tswsxk/EduSim.svg?branch=master)](https://www.travis-ci.org/tswsxk/EduSim)
[![Coverage Status](https://coveralls.io/repos/github/tswsxk/EduSim/badge.svg?branch=master)](https://coveralls.io/github/tswsxk/EduSim?branch=master)

The collection of simulators for education. If you are using this package for your research, please cite our paper [1].

Refer to our [website](http://base.ustc.edu.cn/) and [github](https://github.com/bigdata-ustc) for our publications and more projects

## Installation
```bash
pip install EduSim
```
or git clone and
```bash
pip install -e .--user
```

## List of Environment

* KSS: Knowledge Structure based Simulator, which is used in [1] 

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