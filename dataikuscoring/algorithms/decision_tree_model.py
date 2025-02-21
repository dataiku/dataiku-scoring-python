import numpy as np

from .common import Classifier, Regressor


def get_terminal_node_lt(node, data):
    # XGBoost
    current = node
    while not current.is_leaf:
        if current.is_missing(data):
            current = current.left_child if current.missing_goes_left else current.right_child
        elif data[current.feature_idx] < current.threshold:
            current = current.left_child
        else:
            current = current.right_child
    return current


def get_terminal_node_lte(node, data):
    # Scikit-learn & LightGBM
    current = node
    while not current.is_leaf:
        if current.is_missing(data):
            current = current.left_child if current.missing_goes_left else current.right_child
        elif data[current.feature_idx] <= current.threshold:
            current = current.left_child
        else:
            current = current.right_child
    return current


class Node:
    """
    feature_idx: the feature_idx on which the comparison is made
    label: the 'value' of the node
    """

    def __init__(self, feature_idx=None, threshold=np.nan, left_child=None, right_child=None, label=None, is_leaf=None, missing_goes_left=None, missing_value=np.nan):
        self.label = label
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.missing_goes_left = missing_goes_left
        self.missing_value = missing_value

    def is_missing(self, data):
        if np.isnan(self.missing_value):
            return np.isnan(data[self.feature_idx])
        else:
            return data[self.feature_idx] == self.missing_value

class DecisionTreeModel(Classifier, Regressor):
    """
    This class handle the cases where the tree model was trained by XGBoost, LightGBM or scikit-learn.

    There are three differences in each case: feature type, threshold type, and comparison operand. Type
    cast for the threshold is done when building the tree as an optimization.

    XGBoost:
      - feature type: float32
      - threshold type: np.float32
      - operand: <

    LightGBM:
      - feature type: float64
      - threshold type: np.float64
      - operand: <=

    scikit-learn:
      - feature type: float64(float32)
      - threshold type: np.float64
      - operand: <=
    """

    def __init__(self, model_parameters):
        self.init_tree(model_parameters)
        if self.variant == "XGBOOST":
            self.feature_converter = np.float32
            self.get_terminal_node = get_terminal_node_lt
            self.label_dtype = np.float32
        elif self.variant == "LIGHTGBM":
            self.feature_converter = np.float64
            self.get_terminal_node = get_terminal_node_lte
            self.label_dtype = np.float64
        elif self.variant == "SKLEARN":
            self.feature_converter = lambda x: np.float64(np.float32(x))
            self.get_terminal_node = get_terminal_node_lte
            self.label_dtype = np.float64

    def init_tree(self, model_parameters):
        """ Loading serialized trees

        We initialize one dictionary for the leaves and one for the nodes. The nodes are initialized
        without children. Then we iterate on each node and try to find a left and right child looking up at
        the child's index in the nodes map and then if not found in the leaves map (hence it stays None if
        not found in any).
        """
        if model_parameters.get("xgboost", False):
            self.variant = "XGBOOST"
        elif model_parameters.get("lightgbm", False):
            self.variant = "LIGHTGBM"
        else:
            self.variant = "SKLEARN"

        missing_value = model_parameters.get("missing_value", np.nan)

        convert_threshold = np.float32 if self.variant == "XGBOOST" else np.float64
        leaves = {
            leaf_id: Node(label=label, is_leaf=True, missing_value=missing_value) for leaf_id, label in zip(
                model_parameters["leaf_id"], model_parameters["label"])
        }

        missing = model_parameters.get("missing")
        if missing is None or len(missing) == 0:
            list_missing_goes_left = [None] * len(model_parameters["node_id"])
        else:
            list_missing_goes_left = [v == "l" for v in missing]
        nodes_with_children = {
            node_id: Node(feature_idx=feature, threshold=convert_threshold(threshold), is_leaf=False, missing_goes_left=missing_goes_left, missing_value=missing_value)
            for node_id, feature, threshold, missing_goes_left in zip(
                model_parameters["node_id"], model_parameters["feature"], model_parameters["threshold"], list_missing_goes_left)
        }

        # Connect the nodes to  their children
        for node_id, node in nodes_with_children.items():
            node.left_child = nodes_with_children.get(node_id * 2 + 1, leaves.get(node_id * 2 + 1))
            node.right_child = nodes_with_children.get(node_id * 2 + 2, leaves.get(node_id * 2 + 2))

            # Validation
            if node.left_child is None or node.right_child is None or np.isnan(node.threshold):
                raise ValueError("Tree node is not valid")
            if (node.left_child.is_leaf and node.left_child.label is None) or (
                    node.right_child.is_leaf and node.right_child.label is None):
                raise ValueError("Leaf node does not have a label")

        all_nodes = nodes_with_children
        all_nodes.update(leaves)
        self.root = all_nodes[0]

    def predict(self, X):
        return [self._predict(data) for data in self.feature_converter(X)]

    def _predict(self, data):
        return self.label_dtype(self.get_terminal_node(self.root, data).label)

    def predict_proba(self, X):
        return [self._predict_proba(data) for data in X]

    def _predict_proba(self, data):
        return self._predict(data)

    def __repr__(self):
        return "DecisionTree(variant={})".format(self.variant)
