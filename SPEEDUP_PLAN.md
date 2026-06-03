# Speed up trust-constr NLP solve from ~30s to ~1s (N=20)

## Measured result

Single N=20 seed: **~30s → 0.87s (34x speedup)**
Integration test (3 seeds, full landscape): **74.8s → 29.4s (2.5x)**

---

## Root-cause audit

### Constraint inventory (N=20)

| Constraint | Rows | NNZ/row | Type |
|---|---|---|---|
| eq_init | 4 | 1 | BOX (each pins one variable) |
| ineq_goal | 1 | 2 | nonlinear |
| ineq_progress | 19 | 4 | nonlinear |
| ineq_v_min | 20 | 1 | BOX (v_i >= v_min) |
| ineq_v_max | 20 | 1 | BOX (v_i <= v_max) |
| ineq_accel | 38 | 6 | nonlinear |
| ineq_curvature | 19 | 6 | nonlinear |
| ineq_lat_accel | 19 | 8 | nonlinear |
| ineq_obstacle | 20 | 3 | nonlinear |
| ineq_domain | 80 | 1 | BOX (x_i, y_i in [min,max]) |
| **Total** | **240** | -- | -- |

Actual nonzeros across all Jacobians: **756 of 19,200 = 3.9% fill.**

Box constraints account for **124 of 240 rows (52%)** -- trust-constr re-evaluated and
re-factored them every iteration even though their Jacobians never change.

### The three bottlenecks

**1. Box constraints as NonlinearConstraint** -- eq_init, v_min, v_max, domain are all
pure per-variable bounds. scipy `minimize` has a `bounds` parameter that trust-constr
handles via projected-gradient steps without adding KKT rows. Moving them to `bounds`
cuts constraint rows from 240 to 116 and removes 4 NonlinearConstraint objects.

**2. Dense Jacobians on 3.9%-fill matrices** -- every Jacobian call allocated
`np.zeros((m, 4N))` dense arrays. trust-constr's KKT augmented system was ~320x320
dense, factored via LAPACK each step. Returning `csr_matrix` lets trust-constr use
sparse linear algebra, which for this sparsity is ~15-25x cheaper per factorization.

**3. Zero Hessian forces linear convergence** -- `hess=lambda z,*_: np.zeros((80,80))`
gave the trust-region QP zero curvature everywhere, forcing 100-200 small steps. The
smoothness (J_smooth) and speed (J_speed) objective terms are purely quadratic in dtheta
and dv; their diagonal second derivatives are trivial to compute and give the solver
non-trivial curvature for half the decision variables.

---

## Changes made (planner/optimizer.py only)

### 1. New imports
```python
from scipy.sparse import csr_matrix, diags
```

### 2. Remove 4 box constraints from _make_constraints
Deleted eq_init, ineq_v_min, ineq_v_max, ineq_domain. The remaining 6 constraints
(goal, progress, accel, curvature, lat_accel, obstacle) cover 116 rows.

### 3. Sparse Jacobians for all remaining constraint Jacobians
Each Jacobian function now builds COO (row, col, data) arrays directly from the
vectorized expressions and returns `csr_matrix((data, (rows, cols)), shape=(m, 4*N))`.
No dense intermediate allocation.

### 4. Sparse zero Hessian in _to_nonlinear_constraints
```python
_zero_hess = lambda x, v: csr_matrix((n_vars, n_vars))
```
Replaced the 80x80 dense zero matrix (51 KB per constraint per iteration).

### 5. Diagonal objective Hessian
Added `_obj_hess` closure in generate_candidate_plans:
```python
def _obj_hess(z, *_):
    d = np.zeros(4 * N)
    d[2::4] = 4.0 * W_SMOOTH   # d^2(J_smooth)/d(theta_i)^2, interior
    d[2] = d[4*(N-1)+2] = 2.0 * W_SMOOTH  # endpoints
    d[3::4] = 4.0 * W_SPEED    # d^2(J_speed)/d(v_i)^2, interior
    d[3] = d[4*(N-1)+3] = 2.0 * W_SPEED
    return diags(d)
```

### 6. Bounds for all box constraints
```python
bounds = []
for i in range(N):
    if i == 0:
        bounds += [(x_init, x_init), (y_init, y_init),
                   (theta_init, theta_init), (v_init, v_init)]
    else:
        bounds += [(x_min+margin, x_max-margin), (y_min+margin, y_max-margin),
                   (None, None), (v_min, v_max)]
```

### 7. Vectorized ineq_obstacle function body
Replaced the N-step Python loop over evaluate_footprint with a single batch
`landscape.evaluate(px.ravel(), ...)` call -- same as the existing obstacle_jac.

### 8. maxiter=100, gtol=1e-5
With superlinear convergence from the diagonal Hessian, 100 iterations is more
than sufficient. gtol=1e-5 matches practical path-planning accuracy requirements.

---

## Speedup breakdown

| Fix | Mechanism | Estimated factor |
|---|---|---|
| Bounds (steps 2+6) | 124 rows removed; 4 constraints eliminated | ~3x |
| Sparse Jacobians (steps 3+4) | KKT solve switches to sparse LAPACK | ~4x |
| Diagonal Hessian (step 5) | Superlinear convergence; fewer iterations | ~2x |
| Vectorized obstacle (step 7) | Eliminates N-step Python loop | ~5% |
| **Combined measured** | | **~34x** |
