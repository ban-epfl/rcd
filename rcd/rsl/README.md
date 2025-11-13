# Recursive Skeleton Learning (RSL)

This package provides the reference implementation of the Recursive Skeleton Learning algorithms described in
“Learning Bayesian Networks in the Presence of Structural Side Information” (AAAI 2022) by Mokhtarian *et al.* The
code supports both diamond-free graphs (RSL-D) and graphs with bounded clique number (RSL-W).

## Module layout

- `rsl_base.py` – shared orchestration logic (Markov-boundary estimation, recursive elimination, bookkeeping).
- `rsl_d.py` – RSL-D specialization for diamond-free graphs.
- `rsl_w.py` – RSL-W specialization for graphs whose clique number is bounded.
- `__init__.py` – convenience exports for the public API.

## Usage

The helpers accept either a NumPy array or a Pandas `DataFrame`. Every call must supply a conditional independence
test that follows the signature `(x_idx: int, y_idx: int, cond_set: list[int], data: np.ndarray) -> bool`.

```python
import numpy as np
from rcd.rsl import rsl_d, rsl_w
from rcd.utilities.ci_tests import fisher_z_test

data = np.random.randn(1_000, 6)

# Diamond-free skeleton
g_rsl_d = rsl_d.learn_and_get_skeleton(fisher_z_test, data)

# Bounded-clique skeleton (e.g., clique number <= 3)
g_rsl_w = rsl_w.learn_and_get_skeleton(fisher_z_test, data, clique_num=3)
```

When needed, you can supply a custom Markov-boundary estimator through the `find_markov_boundary_matrix_fun`
argument. See `rcd/utilities/ci_tests.py` for test implementations and `rcd/utilities/utils.py` for helper
functions (e.g., `compute_mb_gaussian`).

## Further reading

- Paper: https://www.jmlr.org/papers/v26/24-0384.html
- Section 5.3 of our paper above.
- Example scripts: `examples/rsl_d_demo.py`, `examples/rsl_w_demo.py`.

For additional guidance on CI tests or extensibility, consult the docstrings inside each module—they follow NumPy
style and document all parameters and return types.
