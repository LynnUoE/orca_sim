"""Analytical distance gradients for the Orca_safety collision model.

``orca_safety_v2.filter`` gets its CBF rows from central differences: two full
distance queries per joint, which is 40 of the filter's ~70 ms per step on the
cube task. The distance query already returns the two closest points, and for
a pair at distance d with closest points p1, p2 and n = (p1 - p2) / d,

    dd/dq = n . (J_p1 - J_p2)

where J_p is the linear Jacobian of the point carried by its link. That needs
one Jacobian pass instead of 34 distance queries. Rows where the links touch or
the closest points coincide (no direction) fall back to central differences.

``install()`` swaps this in for the filter's gradient function. It exists so
that PPO can train with the filter in the loop; Orca_safety/delay describes
the same optimisation on the safety side, not yet in its src/.
"""

from __future__ import annotations

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def analytical_gradients(model, q, pair_names, epsilon: float = 1e-4) -> np.ndarray:
    """Same contract as orca_safety_v2.collision.finite_difference_gradients."""
    from orca_safety_v2.collision import finite_difference_gradients

    names = list(pair_names)
    q = np.asarray(q, dtype=float)
    gradients = np.zeros((len(names), q.size))
    if not names:
        return gradients
    pin = model.pin
    urdf_q = model._update(q)
    pin.computeJointJacobians(model.model, model.data, urdf_q)
    columns = _velocity_columns(model)
    degenerate = []
    for row, name in enumerate(names):
        best = None
        for index in model.pair_indices[name]:
            result = pin.computeDistance(model.geometry, model.geometry_data, index)
            if best is None or result.min_distance < best[0]:
                best = (float(result.min_distance), index,
                        np.array(result.getNearestPoint1()), np.array(result.getNearestPoint2()))
        distance, index, p1, p2 = best
        separation = p1 - p2
        length = float(np.linalg.norm(separation))
        if distance <= 1e-6 or length <= 1e-9:
            degenerate.append(row)
            continue
        normal = separation / length
        pair = model.geometry.collisionPairs[index]
        velocity = np.zeros(model.model.nv)
        for sign, geom_index, point in ((1.0, pair.first, p1), (-1.0, pair.second, p2)):
            joint = model.geometry.geometryObjects[geom_index].parentJoint
            jacobian = pin.getJointJacobian(model.model, model.data, joint, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            arm = point - model.data.oMi[joint].translation
            velocity += sign * normal @ (jacobian[:3] - _skew(arm) @ jacobian[3:])
        gradients[row] = velocity[columns]
    if degenerate:
        gradients[degenerate] = finite_difference_gradients(model, q, [names[r] for r in degenerate], epsilon)
    if not np.all(np.isfinite(gradients)):
        raise ValueError("Non-finite gradient")
    return gradients


def _velocity_columns(model) -> np.ndarray:
    cached = getattr(model, "_orca_rl_velocity_columns", None)
    if cached is None:
        cached = np.array([model.model.joints[model.model.getJointId(n)].idx_v for n in model.joint_names], dtype=int)
        model._orca_rl_velocity_columns = cached
    return cached


def install() -> None:
    """Make CBFSafetyFilter use the analytical gradients (process-wide)."""
    import orca_safety_v2.filter as safety_filter

    safety_filter.finite_difference_gradients = analytical_gradients
