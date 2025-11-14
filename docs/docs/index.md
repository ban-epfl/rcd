# Recursive Causal Discovery

RCD is the reference implementation of the algorithms introduced in our JMLR paper [*Recursive Causal Discovery*](https://www.jmlr.org/papers/v26/24-0384.html) (2025). It bundles RSL-D/W, MARVEL, L-MARVEL, and the ROL hill-climbing refinements, along with CI tests, Markov-boundary utilities, and runnable demos.

## Why RCD?

- **Fast structure learners** – Highly optimized RSL variant that scales to thousands of variables and tens of thousands of samples with ease.
- **Algorithms for every regime** – Diamond-free, bounded clique number, latent-confounded, and sampling bias settings all covered.
- **High empirical accuracy** – Out-of-the-box Fisher-Z pipelines deliver strong F1 scores on synthetic benchmarks (see paper).
- **Theoretical guarantees** – Every implementation mirrors the proofs and pseudocode in the JMLR paper.
- **Plug-and-play workflows** – Works directly on tabular NumPy arrays or pandas DataFrames with only a CI callable required.

## Install

```bash
pip install rcd
```

## Quickstart

```python
from rcd import rsl_d
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges
import networkx as nx
import numpy as np

np.random.seed(2308)
n = 60
p = n ** (-0.85)
adj = gen_er_dag_adj_mat(n, p)
data = gen_gaussian_data(adj, 5_000)

ci = lambda x, y, z, d: fisher_z(x, y, z, d, significance_level=2 / n**2)
learned = rsl_d.learn_and_get_skeleton(ci, data)
true = nx.from_numpy_array(adj, create_using=nx.Graph)
print(f1_score_edges(true, learned))
```

## Learn More

- Browse the API docs for `rcd.rsl`, `rcd.marvel`, `rcd.l_marvel`, and `rcd.rol`.
- Run `python examples/rsl/rsl_d_demo.py` (and friends) to see the algorithms in action.
- Visit [rcdpackage.com](https://rcdpackage.com) for guides and tutorials, and star us on [GitHub](https://github.com/ban-epfl/rcd) to follow releases.

## Citation

If you use RCD, please cite:

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

```text
BSD 2-Clause License © 2024 EPFL
```
