from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


RELATION_CLASSES = np.asarray(("SAME_CELL", "RIGHT", "BELOW", "NO_RELATION"))
SECOND_STAGE_CLASSES = ("RIGHT", "BELOW", "NO_RELATION")


class TwoStageExtraTreesClassifier:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = 1,
        same_cell_threshold: float = 0.45,
    ) -> None:
        if not 0.0 < same_cell_threshold < 1.0:
            raise ValueError("same_cell_threshold must be between 0 and 1")
        common = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "class_weight": "balanced",
            "max_features": "sqrt",
        }
        self.same_cell_model = ExtraTreesClassifier(**common)
        self.other_relation_model = ExtraTreesClassifier(**common)
        self.same_cell_threshold = same_cell_threshold
        self.classes_ = RELATION_CLASSES.copy()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TwoStageExtraTreesClassifier":
        y = np.asarray(y)
        binary = np.where(y == "SAME_CELL", "SAME_CELL", "OTHER")
        if set(binary) != {"SAME_CELL", "OTHER"}:
            raise ValueError("two-stage training requires SAME_CELL and non-SAME_CELL examples")
        self.same_cell_model.fit(x, binary)

        mask = y != "SAME_CELL"
        other_labels = set(y[mask])
        missing = set(SECOND_STAGE_CLASSES) - other_labels
        if missing:
            raise ValueError(f"second-stage training is missing classes: {sorted(missing)}")
        self.other_relation_model.fit(x[mask], y[mask])
        return self

    @staticmethod
    def _probability_for(model: ExtraTreesClassifier, probabilities: np.ndarray, label: str) -> np.ndarray:
        index = list(model.classes_).index(label)
        return probabilities[:, index]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        same_probabilities = self.same_cell_model.predict_proba(x)
        p_same = self._probability_for(
            self.same_cell_model, same_probabilities, "SAME_CELL"
        )
        other_probabilities = self.other_relation_model.predict_proba(x)

        combined = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        combined[:, 0] = p_same
        remaining = 1.0 - p_same
        for label in SECOND_STAGE_CLASSES:
            output_index = list(self.classes_).index(label)
            model_index = list(self.other_relation_model.classes_).index(label)
            combined[:, output_index] = remaining * other_probabilities[:, model_index]
        return combined

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        p_same = probabilities[:, 0]
        other_indices = np.argmax(probabilities[:, 1:], axis=1) + 1
        indices = np.where(p_same >= self.same_cell_threshold, 0, other_indices)
        return self.classes_[indices]
