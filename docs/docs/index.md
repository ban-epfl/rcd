# Welcome to RCD

`RCD` is a Python library for Recursive Causal Discovery.
This package provides efficient implementations of algorithms that recursively learn a causal graph from observational data.
`RCD` focuses on user-friendliness with a well-documented and uniform interface. Moreover, its modular design allows for the integration and expansion of other algorithms and models within the package.

## How to cite:
If you use `RCD` in a scientific publication, we would appreciate citations to the following paper:

Mokhtarian, Ehsan, Sepehr Elahi, Sina Akbari, and Negar Kiyavash. "Recursive Causal Discovery." arXiv preprint arXiv:2403.09300 (2024).

Link to the paper: [arXiv](https://arxiv.org/abs/2403.09300) 

BibTeX entry:
```bibtex
@article{mokhtarian2024recursive,
  title={Recursive Causal Discovery},
  author={Mokhtarian, Ehsan and Elahi, Sepehr and Akbari, Sina and Kiyavash, Negar},
  journal={arXiv preprint arXiv:2024},
  year={2024}
}
```

## GitHub:
The source code is available on [GitHub](https://github.com/ban-epfl/rcd).

## Website:
Documentation are available on [RCD website](https://rcdpackage.com).

## Installation
The package is available on PyPI and can be installed using pip:

```bash
pip install rcd
```

## Basic usage
The following snipped creates a random directed acyclic graph (DAG) and generates Gaussian data from it. Then, it uses one of the algorithms provided in our package, RSL-D, to learn the skeleton of the DAG from the data. Finally, it compares the learned skeleton to the true skeleton and computes the F1 score based on the edges.

```python
from rcd import RSLDiamondFree
from rcd.utilities.ci_tests import *
from rcd.utilities.data_graph_generation import *
from rcd.utilities.utils import f1_score_edges

n = 100
p = n ** (-0.85)
adj_mat = gen_er_dag_adj_mat(n, p)

# generate data from the DAG
data_df = gen_gaussian_data(adj_mat, 1000)

# run rsl-D
ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=2 / n ** 2)
rsl_d = RSLDiamondFree(ci_test)

learned_skeleton = rsl_d.learn_and_get_skeleton(data_df)

# compare the learned skeleton to the true skeleton
true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

# compute F1 score
precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
print(f'Precision: {precision}, Recall: {recall}, F1 score: {f1_score}')
```


## License

This project is provided under the BSD license.

```
BSD 2-Clause License

Copyright (c) 2024, EPFL
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```