# Recursive Causal Discovery (RCD)

> Fast, principled graph discovery for causal scientists and ML practitioners, powered by the algorithms in *Recursive Causal Discovery* (JMLR 2025).

RCD is a batteries-included Python toolkit for learning causal skeletons from observational data. It ships the exact implementations from our [JMLR paper](https://www.jmlr.org/papers/v26/24-0384.html)—RSL-D/W, L-MARVEL, MARVEL, and the ROL hill-climbing refinements—plus the utilities you need to drop them into real pipelines: conditional-independence (CI) tests, Markov-boundary estimators, synthetic-data generators, and ready-to-run demos.

<p align="center">
  <b>Install</b><br>
  <code>pip install rcd</code>
</p>

## Table of Contents

1. [Highlights](#highlights)
2. [Algorithms inside](#algorithms-inside)
3. [Quickstart](#quickstart)
4. [Choose your CI test](#choose-your-ci-test)
5. [Demos & docs](#demos--docs)
6. [Roadmap & contributing](#roadmap--contributing)
7. [How to cite](#how-to-cite)
8. [License](#license)

## Highlights

- **State-of-the-art guarantees** – Implementations follow the proofs and numbering of *Recursive Causal Discovery* (JMLR 26:61) verbatim.
- **Swappable CI tests** – Fisher-Z, Pearson residuals, a battery of power-divergence tests, or your own callable with signature `(x_idx, y_idx, cond_set, data)`.
- **Markov-boundary aware** – Efficient Gaussian estimators out of the box plus hooks for custom routines.
- **Examples that actually run** – Each algorithm ships with a runnable script in `examples/` demonstrating realistic settings and reporting precision/recall/F1.
- **Reproducible utilities** – Synthetic DAG generators, F1 helpers, clique-number estimation, and more so you can benchmark methods in minutes.

## Algorithms inside

| Module | Problem Setting | Notes                                                                |
| --- | --- |----------------------------------------------------------------------|
| `rcd.rsl.rsl_d` | Diamond-free graphs (RSL-D) | Recursive removal for diamond-free graphs. Very fast.                |
| `rcd.rsl.rsl_w` | Graphs with bounded clique number (RSL-W) | Requires clique-number upper bound; handles dense Markov boundaries. |
| `rcd.l_marvel` | Latent MARVEL | Learns skeletons when latent confounders are present.                |
| `rcd.marvel` | MARVEL | No assumption on structure of graph. Runs slow.                      |
| `rcd.rol.rol_hc` | Removal-order hill climbing (ROL-HC) | Warm-started by RSL-D and improves the ordering via local swaps.     |

Every algorithm exposes the same high-level API:

```python
learned_skeleton = algo.learn_and_get_skeleton(
    ci_test=my_ci_function,
    data=data_matrix_or_dataframe,
    **optional_kwargs,
)
```

`ci_test` is any callable with the signature `(x_idx: int, y_idx: int, cond_set: list[int], data: np.ndarray | pd.DataFrame) -> bool`, returning `True` when the variables are conditionally independent.

## Quickstart

```python
import networkx as nx
import numpy as np

from rcd import rsl_d
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges

np.random.seed(2308)
n = 60
p = n ** (-0.85)
adj_mat = gen_er_dag_adj_mat(n, p)
data = gen_gaussian_data(adj_mat, 5_000)

ci_test = lambda x, y, z, d: fisher_z(x, y, z, d, significance_level=2 / n**2)
learned = rsl_d.learn_and_get_skeleton(ci_test, data)

true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)
precision, recall, f1 = f1_score_edges(true_skeleton, learned, return_only_f1=False)
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
```

Want bound-clique graphs instead? Swap `rsl_d` for `rsl_w` and pass `clique_num=get_clique_number(nx_graph)`.

## Choose your CI test

All CI tests live under `rcd.utilities.ci_tests`:

- `fisher_z` – Gaussian Fisher-Z test (default in our paper).
- `pearsonr` – Partial correlation via linear regression residuals.
- `chi_square`, `g_sq`, `freeman_tuckey`, `neyman`, `cressie_read`, `modified_log_likelihood` – members of the power-divergence family supporting discrete data.
- `get_perfect_ci_test(adj_matrix)` – Oracle test derived from ground-truth adjacency, great for debugging.

Because every test shares the same function signature you can mix and match without touching the algorithm code.

## Demos & docs

- **Run the demos** – `python examples/rsl/rsl_d_demo.py`, `python examples/l_marvel/l_marvel_demo.py`, etc. Each prints runtime plus precision/recall/F1, and the RSL-W demo even plots the learned skeleton.
- **API reference** – See the module docstrings (`rcd/rsl/*.py`, `rcd/rol/rol_hc.py`, `rcd/l_marvel/l_marvel.py`) for NumPy-style documentation and theorem references tied to the JMLR paper.
- **Website** – https://rcdpackage.com hosts rendered docs and tutorials.
- **GitHub** – https://github.com/ban-epfl/rcd houses the full source, issues, and release history.

## Roadmap & contributing

We are actively working on:

1. **Meek rule orientation** – Extending the recursive learners with Meek rules to orient as many edges as possible.
2. **CPDAG outputs** – Returning completed partially directed acyclic graphs (CPDAGs) instead of bare skeletons.

Pull requests are welcome—especially those that add new CI tests or improve coverage. Please open an issue describing the improvement, follow our NumPy-style typing/docstring conventions, and run `pytest` plus the relevant demos before submitting.

## How to cite

If you build on RCD, please cite our JMLR article:

> Mokhtarian, E., Elahi, S., Akbari, S., & Kiyavash, N. (2025). Recursive Causal Discovery. *Journal of Machine Learning Research*, 26(61), 1–65. https://www.jmlr.org/papers/v26/24-0384.html

```bibtex
@article{JMLR:v26:24-0384,
  author  = {Ehsan Mokhtarian and Sepehr Elahi and Sina Akbari and Negar Kiyavash},
  title   = {Recursive Causal Discovery},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  volume  = {26},
  number  = {61},
  pages   = {1--65},
  url     = {https://www.jmlr.org/papers/v26/24-0384.html}
}
```

## License

RCD is distributed under the BSD 2-Clause License.

```text
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
