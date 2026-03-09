"""Fuzzy logic operators and tree evaluation engine for HiMu."""

import numpy as np
from typing import Dict, Any, List, Tuple


def fuzzy_and_product(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    """Product t-norm for fuzzy AND: a * b."""
    return signal_a * signal_b


def fuzzy_and_min(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    """Godel t-norm (min) for fuzzy AND: min(a, b)."""
    return np.minimum(signal_a, signal_b)


def fuzzy_and_geometric_mean(signals: list) -> np.ndarray:
    """Geometric mean AND: (prod s_i)^(1/n). N-ary (not pairwise -- geometric mean is not associative)."""
    n = len(signals)
    if n == 1:
        return signals[0]
    stacked = np.stack(signals)
    return np.prod(stacked, axis=0) ** (1.0 / n)


def fuzzy_or(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    """Probabilistic sum for fuzzy OR: A + B - A*B."""
    return signal_a + signal_b - signal_a * signal_b


def temporal_seq(signals: list) -> np.ndarray:
    """
    Symmetric temporal SEQ operator: events happen in chronological order.

    SEQ(A, B, C, ...) means A happens first, then B, then C, etc.
    Children are ordered chronologically (earliest event first, latest event last).

    The result is high at time t if child l is active at t, all predecessors
    fired before t (backward gates), and all successors will fire after t
    (forward gates). The final score is the max over all children, so every
    event in the sequence contributes its own peak to the satisfaction curve.
    """
    L = len(signals)
    if L < 2:
        raise ValueError("SEQ requires at least 2 signals")

    num_frames = len(signals[0])

    # Backward gates: H[l](t) = "was child l satisfied at some time < t?"
    H = []
    for sig in signals:
        cm = np.maximum.accumulate(sig)
        past_max = np.roll(cm, 1)
        past_max[0] = 0.0
        H.append(past_max)

    # Forward gates: F[l](t) = "will child l be satisfied at some time > t?"
    F = []
    for sig in signals:
        cm = np.maximum.accumulate(sig[::-1])[::-1]
        future_max = np.roll(cm, -1)
        future_max[-1] = 0.0
        F.append(future_max)

    # Each child: u_l(t) * prod(j<l) H[j](t) * prod(j>l) F[j](t)
    result = np.zeros(num_frames)
    for ell in range(L):
        score = signals[ell].copy()
        for j in range(ell):
            score = score * H[j]
        for j in range(ell + 1, L):
            score = score * F[j]
        result = np.maximum(result, score)

    return result


def fuzzy_right_after(
    cause: np.ndarray,
    effect: np.ndarray,
    kappa: float = 2.0,
    fps: float = 1.0
) -> np.ndarray:
    """
    Symmetric temporal RIGHT_AFTER operator: effect immediately follows cause.

    Produces high scores at both cause and effect timestamps:
    - S_effect(t) = effect(t) * sum_{s<t} cause(s) * exp(-kappa*(t-s)/fps)
    - S_cause(t)  = cause(t)  * sum_{s>t} effect(s) * exp(-kappa*(s-t)/fps)
    - RIGHT_AFTER(t) = max(S_cause(t), S_effect(t))
    """
    num_frames = len(cause)

    # S_effect: effect active now, cause happened recently before
    weighted_history = np.zeros(num_frames)
    for t in range(1, num_frames):
        past_times = np.arange(t)
        time_diffs = (t - past_times) / fps
        weights = np.exp(-kappa * time_diffs)
        weighted_history[t] = np.sum(cause[past_times] * weights)
    s_effect = effect * weighted_history

    # S_cause: cause active now, effect will happen shortly after
    weighted_future = np.zeros(num_frames)
    for t in range(num_frames - 1):
        future_times = np.arange(t + 1, num_frames)
        time_diffs = (future_times - t) / fps
        weights = np.exp(-kappa * time_diffs)
        weighted_future[t] = np.sum(effect[future_times] * weights)
    s_cause = cause * weighted_future

    return np.maximum(s_effect, s_cause)


class LogicEngine:
    """Evaluates logical trees to produce truth curves."""

    VALID_AND_MODES = ("product", "min", "geometric_mean")

    def __init__(self, kappa: float = 2.0, fps: float = 1.0, and_mode: str = "product"):
        """
        Initialize the logic engine.

        Args:
            kappa: Decay rate for RIGHT_AFTER operator
            fps: Frames per second for temporal operators
            and_mode: AND operator variant -- "product", "min", or "geometric_mean"
        """
        if and_mode not in self.VALID_AND_MODES:
            raise ValueError(f"Invalid and_mode '{and_mode}'. Must be one of {self.VALID_AND_MODES}")
        self.kappa = kappa
        self.fps = fps
        self.and_mode = and_mode

        and_fn_map = {"product": fuzzy_and_product, "min": fuzzy_and_min}
        self.operator_map = {
            "AND": and_fn_map.get(and_mode),  # None for geometric_mean (handled specially)
            "OR": fuzzy_or,
        }

    def evaluate_tree(
        self,
        tree: Dict[str, Any],
        frame_scores: Dict[int, np.ndarray],
        node_id: int = 0
    ) -> tuple[np.ndarray, int]:
        """
        Recursively evaluate a logical tree to produce a truth curve.

        Args:
            tree: Logical tree node (dict with 'op', 'children', etc.)
            frame_scores: Mapping from node_id to score arrays
            node_id: Current node ID for indexing into frame_scores

        Returns:
            Tuple of (truth_curve, next_node_id)
        """
        op = tree["op"]

        if op == "LEAF":
            if node_id not in frame_scores:
                raise ValueError(f"Missing scores for LEAF node {node_id}")
            return frame_scores[node_id], node_id + 1

        children = tree.get("children", [])
        if not children:
            raise ValueError(f"Non-LEAF node must have children: {tree}")

        child_signals = []
        current_id = node_id

        for child in children:
            child_signal, current_id = self.evaluate_tree(child, frame_scores, current_id)
            child_signals.append(child_signal)

        # Apply the operator
        if op == "SEQ":
            if len(child_signals) < 2:
                raise ValueError("SEQ requires at least 2 children")
            result = temporal_seq(child_signals)
        elif op == "RIGHT_AFTER":
            if len(child_signals) != 2:
                raise ValueError("RIGHT_AFTER requires exactly 2 children")
            result = fuzzy_right_after(
                child_signals[0],
                child_signals[1],
                kappa=self.kappa,
                fps=self.fps
            )
        elif op == "AND" and self.and_mode == "geometric_mean":
            result = fuzzy_and_geometric_mean(child_signals)
        elif op in self.operator_map and self.operator_map[op] is not None:
            operator_fn = self.operator_map[op]
            if len(child_signals) == 1:
                result = child_signals[0]
            elif len(child_signals) == 2:
                result = operator_fn(child_signals[0], child_signals[1])
            elif len(child_signals) > 2:
                result = child_signals[0]
                for signal in child_signals[1:]:
                    result = operator_fn(result, signal)
            else:
                raise ValueError(f"Operator {op} has no children")
        else:
            raise ValueError(f"Unknown operator: {op}")

        return result, current_id

    def evaluate_tree_restandardized(
        self,
        tree: Dict[str, Any],
        frame_scores: Dict[int, np.ndarray],
        node_id: int = 0
    ) -> tuple[np.ndarray, int]:
        """Evaluate tree with post-operator re-standardization of OR branches.

        If the root node is OR (MCQ tree), evaluates each child branch
        independently, applies min-max [0,1] normalization to each branch,
        then combines with fuzzy OR. This ensures candidate options compete
        equally regardless of sub-tree depth.

        For non-OR root nodes, falls through to standard evaluation.
        """
        if tree["op"] != "OR":
            return self.evaluate_tree(tree, frame_scores, node_id)

        children = tree.get("children", [])
        if not children:
            raise ValueError("OR node must have children")

        branch_curves = []
        current_id = node_id
        for child in children:
            curve, current_id = self.evaluate_tree(child, frame_scores, current_id)
            cmin, cmax = curve.min(), curve.max()
            if cmax - cmin > 1e-8:
                curve = (curve - cmin) / (cmax - cmin)
            else:
                curve = np.zeros_like(curve)
            branch_curves.append(curve)

        result = branch_curves[0]
        for b in branch_curves[1:]:
            result = fuzzy_or(result, b)

        return result, current_id

    @staticmethod
    def _rescale(signal: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Min-max rescale signal to [lo, hi]."""
        smin, smax = signal.min(), signal.max()
        if smax - smin > 1e-8:
            return lo + (hi - lo) * (signal - smin) / (smax - smin)
        return np.full_like(signal, (lo + hi) / 2.0)

    def evaluate_tree_rescaled(
        self,
        tree: Dict[str, Any],
        frame_scores: Dict[int, np.ndarray],
        node_id: int = 0,
        lo: float = 0.5,
        hi: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Recursively evaluate tree with child rescaling before each operator.

        Same logic as evaluate_tree() but before applying any operator, each
        child signal is min-max rescaled to [lo, hi]. This reduces the
        depth-suppression effect of product AND.
        """
        op = tree["op"]

        if op == "LEAF":
            if node_id not in frame_scores:
                raise ValueError(f"Missing scores for LEAF node {node_id}")
            return frame_scores[node_id], node_id + 1

        children = tree.get("children", [])
        if not children:
            raise ValueError(f"Non-LEAF node must have children: {tree}")

        child_signals = []
        current_id = node_id
        for child in children:
            child_signal, current_id = self.evaluate_tree_rescaled(
                child, frame_scores, current_id, lo, hi
            )
            child_signals.append(child_signal)

        # Rescale each child to [lo, hi] before applying the operator
        child_signals = [self._rescale(s, lo, hi) for s in child_signals]

        # Apply operator (mirrors evaluate_tree)
        if op == "SEQ":
            if len(child_signals) < 2:
                raise ValueError("SEQ requires at least 2 children")
            result = temporal_seq(child_signals)
        elif op == "RIGHT_AFTER":
            if len(child_signals) != 2:
                raise ValueError("RIGHT_AFTER requires exactly 2 children")
            result = fuzzy_right_after(
                child_signals[0], child_signals[1],
                kappa=self.kappa, fps=self.fps
            )
        elif op == "AND" and self.and_mode == "geometric_mean":
            result = fuzzy_and_geometric_mean(child_signals)
        elif op in self.operator_map and self.operator_map[op] is not None:
            operator_fn = self.operator_map[op]
            if len(child_signals) == 1:
                result = child_signals[0]
            elif len(child_signals) == 2:
                result = operator_fn(child_signals[0], child_signals[1])
            elif len(child_signals) > 2:
                result = child_signals[0]
                for signal in child_signals[1:]:
                    result = operator_fn(result, signal)
            else:
                raise ValueError(f"Operator {op} has no children")
        else:
            raise ValueError(f"Unknown operator: {op}")

        return result, current_id

    def collect_leaf_nodes(self, tree: Dict[str, Any], node_id: int = 0) -> list[tuple[int, Dict[str, Any]]]:
        """
        Collect all LEAF nodes with their IDs for expert evaluation.

        Returns:
            List of (node_id, leaf_node_dict) tuples
        """
        if tree["op"] == "LEAF":
            return [(node_id, tree)]

        leaves = []
        current_id = node_id

        for child in tree.get("children", []):
            child_leaves = self.collect_leaf_nodes(child, current_id)
            leaves.extend(child_leaves)
            current_id += len(child_leaves)

        return leaves

    def group_leaves_by_expert(
        self,
        leaf_nodes: List[Tuple[int, Dict[str, Any]]]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Group leaf nodes by expert type for batch processing.

        Returns:
            Dict mapping expert_type -> List of (node_id, query) tuples
        """
        grouped: Dict[str, List[Tuple[int, str]]] = {}
        for node_id, leaf in leaf_nodes:
            expert_type = leaf["expert"]
            query = leaf["query"]
            if expert_type not in grouped:
                grouped[expert_type] = []
            grouped[expert_type].append((node_id, query))
        return grouped
