import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, Tuple, Set, List
from math import factorial
import time
from copy import copy
from tqdm import tqdm
import scipy

from woodelf.parse_models import load_decision_tree_ensemble_model

class DecisionTreeNode:
    """
    Represent a decision tree node. Recursively the root node builds the tree structure (the root node knows it children and so on).
    Include several useful tree functions like BFS and n.split(c).
    """

    def __init__(
            self, feature_name: str, value: float, right: Optional["DecisionTreeNode"], left: Optional["DecisionTreeNode"], nan_go_left=True, index: int=None, cover=None,
            feature_contribution_replacement_values=None,
        ):
        """
        See decision tree definition in the paper (Def. 6)
        The tree split function is a bit different as XGBoost also support NaN values.
        The split function is "go left if df[feature_name]<value or (nan_go_left and df[feature_name] == NaN)".
        Right and left parameters can be a DesicionTreeNode if this node is an inner node or None if the node is a leaf. (The leaf weight will be saved as the 'value')
        Cover is an optional parameter, it includes how many rows in the train set reached this node.
        """
        self.index=index
        self.feature_name = feature_name
        self.value = float(value)
        self.right = right
        self.left = left
        self.nan_go_left = nan_go_left
        self.cover = cover
        self.consumer_pattern_to_characteristic_wdnf = None
        self.pc_pb_to_cube = None
        self.feature_contribution_replacement_values = feature_contribution_replacement_values
        self.parent = -1
        self.depth=None

    def shall_go_left(self, row):
        """
        This is the n.split(c) defined in Def.6
        """
        if self.nan_go_left:
            return (row[self.feature_name] < self.value) | row[self.feature_name].isna()
        else:
            return row[self.feature_name] < self.value

    def shall_go_right(self, row):
        return ~self.shall_go_left(row)

    def GPU_shall_go_left(self, row):
        """
        This is the n.split(c) defined in Def.6
        """
        if self.nan_go_left:
            return (row[self.feature_name] < self.value) | cp.isnan(row[self.feature_name])
        else:
            return row[self.feature_name] < self.value

    def GPU_shall_go_right(self, row):
        return ~self.GPU_shall_go_left(row)

    def is_leaf(self):
        return self.right is None and self.left is None

    def is_almost_leaf(self):
        return not self.is_leaf() and (self.right.is_leaf() or self.left.is_leaf())

    def predict(self, data):
        if self.is_leaf():
            return pd.Series(self.value, index=data.index)
        return self.shall_go_left(data) * self.left.predict(data) + self.shall_go_right(data) * self.right.predict(data)

    def bfs(self, including_myself: bool = True, including_leaves: bool = True):
        """
        Return all the node children (and the node itself) in BFS order. The indexes should be in an increasing order.
        """
        if self.is_leaf():
            return [self] if including_myself and including_leaves else []

        children = [self] if including_myself else []
        nodes_to_visit = []
        if self.right is not None:
            nodes_to_visit.append(self.right)
        if self.left is not None:
            nodes_to_visit.append(self.left)

        while len(nodes_to_visit) > 0:
            current_node = nodes_to_visit.pop(0)
            if current_node.right is not None:
                nodes_to_visit.append(current_node.right)
            if current_node.left is not None:
                nodes_to_visit.append(current_node.left)

            if current_node.is_leaf():
                if including_leaves:
                    children.append(current_node)
            else:
                children.append(current_node)

        return children

    def get_all_leaves(self):
        children = self.bfs(including_leaves=True)
        return [node for node in children if node.is_leaf()]

    def get_all_almost_leaves(self):
        children = self.bfs(including_leaves=True)
        return [node for node in children if node.is_almost_leaf()]

    def get_all_features(self):
        inner_nodes = self.bfs(including_leaves=False)
        return set(n.feature_name for n in inner_nodes)

    def get_all_leaves_with_path_to_root(self):
        nodes_to_visit = [(self, [])]
        leaves = []
        while len(nodes_to_visit) > 0:
            current_node, current_path_to_root = nodes_to_visit.pop(0)
            for next_node in [current_node.right, current_node.left]:
                next_node_obj = (next_node, current_path_to_root + [current_node.feature_name])
                if next_node.is_leaf():
                    leaves.append(next_node_obj)
                else:
                    nodes_to_visit.append(next_node_obj)
        return leaves

    def __repr__(self):
        if self.is_leaf():
            return f"{self.index} (cover: {self.cover}): leaf with value {self.value}"
        return f"{self.index} (cover: {self.cover}): {self.feature_name} < {self.value}"

def load_xgboost_tree(tree, features):
    """
    Given an XGBoost Regressor tree, parse it and build a DecisionTreeNode object with it structure.
    Use the Tree object returned by the shap package's XGBTreeModelLoader class (given as the 'tree' parameter).
    The function also gets the training features.
    """
    nodes = {}
    for index in range(len(tree.thresholds)):
        threshold = tree.thresholds[index]
        leaf_value = tree.values[index][0]
        if threshold == 0 and leaf_value != 0:
            value = leaf_value
        else:
            value = threshold
        nan_go_left = (tree.children_left[index] == tree.children_default[index])
        cover = tree.node_sample_weight[index]
        feature_index = tree.features[index]
        nodes[index] = DecisionTreeNode(
            feature_name=features[feature_index], value=value, right=None, left=None,
            nan_go_left=nan_go_left, index=index, cover=cover
        )

    for index in range(len(tree.thresholds)):
        child_left = tree.children_left[index]
        child_right = tree.children_right[index]

        if child_left != -1:
            nodes[index].left = nodes[child_left]
            nodes[child_left].parent = nodes[index]
        if child_right != -1:
            nodes[index].right = nodes[child_right]
            nodes[child_right].parent = nodes[index]

    nodes[0].depth = tree.max_depth
    return nodes[0]


def cast_tree_format(woodelf_node):
    if woodelf_node.is_leaf():
        leaf = DecisionTreeNode(
            feature_name=None, value=woodelf_node.value, right=None, left=None, cover=woodelf_node.cover, index=woodelf_node.index,
        )
        leaf.depth = woodelf_node.depth
        return leaf
    left = cast_tree_format(woodelf_node.left) if woodelf_node.left is not None else None
    right = cast_tree_format(woodelf_node.right) if woodelf_node.right is not None else None
    node = DecisionTreeNode(
            feature_name=woodelf_node.feature_name, value=woodelf_node.value, right=right, left=left,
            nan_go_left=woodelf_node.nan_go_left, cover=woodelf_node.cover, index=woodelf_node.index,
    )
    node.depth = woodelf_node.depth
    left.parent = node
    right.parent = node
    return node

def load_model(model, features):
    """
    Load an XGBoost regressor tree (utilizing the shap python package parsing object)
    """
    model_obj = load_decision_tree_ensemble_model(model, features)
    return [cast_tree_format(tree) for tree in model_obj.trees]

def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

class CubeCharacteristicFunctionMetric(object):
    """
    An abstruct class that calculate a metric on a cube/clause characteristic function.
    You can implement this class (override the calc_metric function) and then use this class and the WOODELF algorithm
    to calculate your metric effiecently on large background datasets.

    Here, the metrics that inherit this class are: Shapley values, Shapley interaction values, Banzhaf values and Banzhaf interaction values
    """
    INTERACTION_VALUE = False
    INTERACTION_VALUE_ONE_SIDE = False

    def calc_metric(self, s_plus, s_minus):
        raise NotImplemented()

class ShapleyValues(CubeCharacteristicFunctionMetric):
    """
    Implement the linear-time formula for Shapley value computation on WDNF/WCNF, see Formula 3 in the paper.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets

        s = s_plus | s_minus
        shapley_values = {}

        # The new simple shapley values formula
        if len(s_plus) > 0:
            contribution = (1 / (len(s_plus) * nCk(len(s), len(s_plus))))
            for must_exist_feature in s_plus:
                shapley_values[must_exist_feature] = contribution

        if len(s_minus) > 0:
            contribution = -1 / (len(s_minus) * nCk(len(s), len(s_minus)))
            for must_be_missing_feature in s_minus:
                shapley_values[must_be_missing_feature] = contribution

        return shapley_values

class ShapleyInteractionValues(CubeCharacteristicFunctionMetric):
    """
    Implement the formulas for Shapley interaction values computation on WDNF/WCNF, see Table 1 in the paper.
    """
    INTERACTION_VALUE = True
    INTERACTION_VALUE_ONE_SIDE = True

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets

        shapley_values = {}
        s = s_plus | s_minus
        if len(s_plus) > 0:
            # i,j in S+
            if len(s_plus) > 1:
                # 0.5 because the shapley interaction values in the shap package are actually divided by 2....
                contribution = 0.5 / ((len(s_plus) - 1) * nCk(len(s) - 1, len(s_plus) - 1))
                for must_exists_feature in s_plus:
                    for other_feature in s_plus:
                        if must_exists_feature < other_feature:
                            shapley_values[(must_exists_feature, other_feature)] = contribution

            # i in S+   j in S-
            if len(s_minus) > 0:
                contribution = -0.5 / (len(s_minus) * nCk(len(s) - 1, len(s_minus)))
                for must_exists_feature in s_plus:
                    for other_feature in s_minus:
                        if must_exists_feature < other_feature:
                            shapley_values[(must_exists_feature, other_feature)] = contribution

        if len(s_minus) > 0:
            # i,j in S-
            if len(s_minus) > 1:
                contribution = 0.5 / ((len(s_minus) - 1) * nCk(len(s) - 1, len(s_minus) - 1))
                for must_be_missing_feature in s_minus:
                    for other_feature in s_minus:
                        if must_be_missing_feature < other_feature:
                            shapley_values[(must_be_missing_feature, other_feature)] = contribution
            # i in S-   j in S+
            if len(s_plus) > 0:
                contribution = -0.5 / (len(s_plus) * nCk(len(s) - 1, len(s_plus)))
                for must_be_missing_feature in s_minus:
                    for other_feature in s_plus:
                        if must_be_missing_feature < other_feature:
                            shapley_values[(must_be_missing_feature, other_feature)] = contribution
        return shapley_values


class BanzahfValues(CubeCharacteristicFunctionMetric):
    """
    Implement the linear-time formula for Banzhaf value computation on WDNF/WCNF, see Formula 6 in the paper.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets

        s = s_plus | s_minus
        banzhaf_values = {}

        s_plus_contribution = 1 / (2 ** (len(s) - 1))
        s_minus_contribution = -s_plus_contribution
        # The new simple shapley values formula
        if len(s_plus) > 0:
            for must_exist_feature in s_plus:
                banzhaf_values[must_exist_feature] = s_plus_contribution

        if len(s_minus) > 0:
            for must_be_missing_feature in s_minus:
                banzhaf_values[must_be_missing_feature] = s_minus_contribution

        return banzhaf_values


class BanzhafInteractionValues(CubeCharacteristicFunctionMetric):
    """
    Implement the formulas for Banzhaf interaction values computation on WDNF/WCNF, see Formula 7 in the paper.
    """
    INTERACTION_VALUE = True
    INTERACTION_VALUE_ONE_SIDE = True

    def calc_metric(self, s_plus, s_minus):
        if len(s_plus & s_minus) > 0:
            return {} # se and sne must be disjoint sets
        banzhaf_values = {}

        contribution = (1 / (2 ** (len(s_plus) + len(s_minus) - 2)))

        s = s_plus | s_minus
        if len(s_plus) > 0:
            # i,j in S+
            if len(s_plus) > 1:
                for must_exists_feature in s_plus:
                    for other_feature in s_plus:
                        if must_exists_feature < other_feature:
                            banzhaf_values[(must_exists_feature, other_feature)] = contribution

            # i in S+   j in S-
            if len(s_minus) > 0:
                for must_exists_feature in s_plus:
                    for other_feature in s_minus:
                        if must_exists_feature < other_feature:
                            banzhaf_values[(must_exists_feature, other_feature)] = -contribution

        if len(s_minus) > 0:
            # i,j in S-
            if len(s_minus) > 1:
                for must_be_missing_feature in s_minus:
                    for other_feature in s_minus:
                        if must_be_missing_feature < other_feature:
                            banzhaf_values[(must_be_missing_feature, other_feature)] = contribution
            # i in S-   j in S+
            if len(s_plus) > 0:
                for must_be_missing_feature in s_minus:
                    for other_feature in s_plus:
                        if must_be_missing_feature < other_feature:
                            banzhaf_values[(must_be_missing_feature, other_feature)] = -contribution
        return banzhaf_values

class PathToValuesMatrix():
    """
    An object that in charge of creating the M matrix for every leaf and feature.
    It takes the features along the root-to-leaf path and build the matrix (lines 7-16 in WOODELF pseudo code)
    The class also utilize the fact that the matrix only depends on the repitting sequence of the features along the path.
    For example the feature repetition sequence of ["weight", "pluse", "age", "sex", "pluse", "sex"] is [1, 2, 3, 4, 2, 4].
    All feature lists with this feature repetition sequence have the same set of matrixes.

    This cache mechanism is improvement 2 in Sec. 9.1
    """
    def __init__(self, metric: CubeCharacteristicFunctionMetric):
        self.metric = metric

        self.cached_used = 0
        self.cache_miss = 0
        self.cache = {}


    def get_values_matrixes(self, features_in_path: List[str]):
        """
        Apply the CubeCharacteristicFunctionMetric object (the v function), to create the matrixes M.
        Use the cache when possible, and update the cache with the created matrixes
        """
        frs = self.get_feature_repetition_sequence(features_in_path)

        frs_tuple = tuple(frs)
        if frs_tuple in self.cache:
            self.cached_used += 1
            matrixes = self.cache[frs_tuple]
        else:
            self.cache_miss += 1
            pc_pb_to_cube = self.map_patterns_to_cube(frs)
            matrixes = self.build_patterns_to_values_matrix(pc_pb_to_cube, self.metric, len(features_in_path))
            self.cache[frs_tuple] = matrixes

        if not self.metric.INTERACTION_VALUE:
            matrixes_for_the_given_features = {features_in_path[index]: matrixes[index] for index in matrixes}
        else:
            matrixes_for_the_given_features = {}
            for feature_index_1, feature_index_2 in matrixes:
                the_right_index = (features_in_path[feature_index_1], features_in_path[feature_index_2])

                # The feature_appearance can change the order of a feature pair (f1,f2) in one sided interaction metric, here we fix this.
                if self.metric.INTERACTION_VALUE_ONE_SIDE and features_in_path[feature_index_1] > features_in_path[feature_index_2]:
                    the_right_index = (features_in_path[feature_index_2], features_in_path[feature_index_1])

                matrixes_for_the_given_features[the_right_index] = matrixes[(feature_index_1, feature_index_2)]

        return matrixes_for_the_given_features

    @staticmethod
    def get_feature_repetition_sequence(features_in_path: List[str]):
        """
        Generate the feature repetition sequence.
        The math is simple, the feature at index i is replaced by i
        unless it appeared before in the sequance, in that case it will be represented by the index it already received.

        Examples:
        ["sex", "pluse", "age", "weight", "heart_rate", "sugar_in_blood"] => [1, 2, 3, 4, 5, 6]
        ["weight", "pluse", "age", "sex", "pluse", "sex"] => [1, 2, 3, 4, 2, 4]
        """
        feature_to_index = {}
        frs = []
        for i, feature in enumerate(features_in_path):
            if feature in feature_to_index:
                frs.append(feature_to_index[feature])
            else:
                feature_to_index[feature] = i
                frs.append(i)

        return frs


    @staticmethod
    def build_patterns_to_values_matrix(dl, metric: CubeCharacteristicFunctionMetric, path_length):
        """
        Apply the CubeCharacteristicFunctionMetric object (the v function), to create the matrixes M.
        include lines 12-16 in WOODELF pseudo code.
        dl is the returned mapping from the map_patterns_to_cube function
        """
        matrix_details = {}
        for pc in dl:
            for pb in dl[pc]:
                s_plus, s_minus = dl[pc][pb]
                values = metric.calc_metric(s_plus, s_minus)
                for feature in values:
                    # Implement the line "M[l][feature][p_c][p_b] = value" in an efficient way that utilize the sparsity of M.
                    if feature not in matrix_details:
                        matrix_details[feature] = {"pcs": [], "pbs": [], "values": []}
                    matrix_details[feature]["pcs"].append(pc)
                    matrix_details[feature]["pbs"].append(pb)
                    matrix_details[feature]["values"].append(values[feature])

        matrixs = {}
        for feature in matrix_details:
            # Save M as a sparse matrix (Improvement 1 in Sec. 9.1)
            matrix_values = (matrix_details[feature]["values"], (matrix_details[feature]["pcs"], matrix_details[feature]["pbs"]))
            matrixs[feature] = scipy.sparse.coo_matrix(matrix_values, shape=(2**path_length, 2**path_length), dtype=np.float32).tocsc()
        return matrixs

    @staticmethod
    def map_patterns_to_cube(features_in_path: List[str]):
        """
        The function MapPatternsToCube from Sect. 5 of the article.
        :params tree: The decision tree
        :params current_wdnf_table: The format is: wdnf_table[consumer_decision_pattern][background_decision_pattern] = (cube_positive_literals, cube_negative_literals)
        """
        updated_wdnf_table = {0: {0: (set(), set())}}
        current_wdnf_table = None
        for feature in features_in_path:
            current_wdnf_table = updated_wdnf_table
            updated_wdnf_table = {}
            for consumer_pattern in current_wdnf_table:
                updated_wdnf_table[consumer_pattern * 2 + 0] = {}
                updated_wdnf_table[consumer_pattern * 2 + 1] = {}
                for background_pattern in current_wdnf_table[consumer_pattern]:
                    # Get the current cube (the possitive and negated literals) of the consumer and background patterns
                    s_plus, s_minus = current_wdnf_table[consumer_pattern][background_pattern]
                    # Implement the 4 rules
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 0] = (s_plus | {feature}, s_minus) # Rule 1
                    updated_wdnf_table[consumer_pattern * 2 + 0][background_pattern * 2 + 1] = (s_plus, s_minus | {feature}) # Rule 2
                    updated_wdnf_table[consumer_pattern * 2 + 1][background_pattern * 2 + 1] = (s_plus, s_minus) # Rule 3

        return updated_wdnf_table




def get_int_dtype_from_depth(depth):
    """
    The decision pattern, when encoded as a number, have a bit for each node of the root-to-leaf-path.
    Choose the dtype according to the tree depth (a.k.a the max pattern length).

    This is improvement 5 of Sec. 9.1
    """
    if depth <= 8:
        return np.uint8
    if depth <= 16:
        return np.uint16
    if depth <= 32:
       return np.uint32
    return np.uint64


def GPU_get_int_dtype_from_depth(depth):
    """
    Like get_int_dtype_from_depth but return CuPy types.
    """
    if depth <= 8:
        return cp.uint8
    if depth <= 16:
        return cp.uint16
    if depth <= 32:
       return cp.uint32
    return cp.uint64


def calc_decision_patterns(tree, data, depth, GPU=False):
    """
    An effiecent implementation of the CalcDecisionPatterns from Sec. 4 of the paper.
    """
    # Use a tight uint type for efficiency. This is improvement 5 of Sec. 9.1
    int_dtype = GPU_get_int_dtype_from_depth(depth) if GPU else get_int_dtype_from_depth(depth)

    leaves_patterns_dict = {} # This is the P_leaves mentioned in the paper
    inner_nodes_patterns_dict = {} # This is P_all
    if GPU:
        data_length = len(data[list(data.keys())[0]])
        inner_nodes_patterns_dict[tree.index] = cp.zeros(data_length, dtype=int_dtype)
    else:
        inner_nodes_patterns_dict[tree.index] = pd.Series(0, index=data.index).to_numpy().astype(int_dtype)

    for current_node in tree.bfs():
        if current_node.is_leaf():
            leaves_patterns_dict[current_node.index] = inner_nodes_patterns_dict[current_node.index]
            continue

        if GPU:
            left_bool_condition = current_node.GPU_shall_go_left(data)
            left_condition = left_bool_condition # .to_numpy().astype(int_dtype)
            right_condition = ~left_bool_condition # (~left_bool_condition).to_numpy().astype(int_dtype)
        else:
            left_bool_condition = current_node.shall_go_left(data)
            left_condition = left_bool_condition.to_numpy().astype(int_dtype)
            right_condition = (~left_bool_condition).to_numpy().astype(int_dtype)
        my_pattern = inner_nodes_patterns_dict[current_node.index]
        shifted_my_pattern = (my_pattern << 1)
        inner_nodes_patterns_dict[current_node.left.index] = shifted_my_pattern + left_condition
        inner_nodes_patterns_dict[current_node.right.index] = shifted_my_pattern + right_condition
    return leaves_patterns_dict


def preprocess_tree_background(tree: DecisionTreeNode, background_data: pd.DataFrame, depth: int, path_to_matrixes_calculator: PathToValuesMatrix, GPU=False):
    """
    Run all the preprocessing needed given a tree and a background_data.
    Include lines 2-21 of the pseudo-code.
    """
    background_patterns_matrix = calc_decision_patterns(tree, background_data, depth, GPU)

    # Build f, implements lines 3-4 of the pseudo-code
    Frq_b = {}
    visited_leaves_parents = {}
    data_length = len(background_data) if not GPU else len(background_data[list(background_data.keys())[0]])
    for leaf, features_in_path in tree.get_all_leaves_with_path_to_root():
        if leaf.parent.index not in visited_leaves_parents:
            # np.bincount is a faster way to implement value_counts that uses the fact all decision patterns are integers between 0 and 2**depth
            if GPU:
                Frq_b[leaf.index] = cp.bincount(background_patterns_matrix[leaf.index], minlength=2**len(features_in_path))
                Frq_b[leaf.index] = Frq_b[leaf.index] / data_length
                Frq_b[leaf.index] = cp.asnumpy(Frq_b[leaf.index])
            else:
                Frq_b[leaf.index] = np.bincount(background_patterns_matrix[leaf.index], minlength=2**len(features_in_path))
                Frq_b[leaf.index] = Frq_b[leaf.index] / data_length
            visited_leaves_parents[leaf.parent.index] = Frq_b[leaf.index]
        else:
            # neighboor leaves have similar patterns (only the last bit is different)
            # For efficiency we reuse the frequencies computed for the neighboor.

            # Given leaves l_i, l_{i+1} s.t. there is an inner node n where n.left = l_i and n.right=l_{i+1}.
            # The decision pattern of any consumer c in leaf l_i is the same as in leaf l_{i+1} except for the last bit which is different.
            # For example if the pattern of c and l_i is 010011011101 then the pattern of c and l_{i+1} is 010011011100 (the 1 in the end is replaced with 0)
            # Let the frequencies of l_i be [f1,f2,f3,f4,....,f_{n-1}, f_n], we can these conclude that the frequencies of l_{i+1} are [f2,f1,f4,f3,....,f_n, f_{n-1}].
            # We can find them by swapping any pair of numbers in the array.
            # The code below utilize this fact for efficiency - this saved half of the bincount opperations.
            # This trick is part of improvement 3 in Sec. 9.1 (this is the improvement to line 4)
            neighboor_frq = visited_leaves_parents[leaf.parent.index]
            frqs = []
            for i in range(0, len(neighboor_frq), 2):
                frqs.append(neighboor_frq[i+1])
                frqs.append(neighboor_frq[i])
            Frq_b[leaf.index] = np.array(frqs, dtype=np.float32)

    for leaf, features_in_path in tree.get_all_leaves_with_path_to_root():
        # Build M, implements lines 7-16 of the pseudo-code
        matrixes = path_to_matrixes_calculator.get_values_matrixes(features_in_path)

        # Build s, implements lines 17-21 of the pseudo-code
        features_to_values = {}
        fl = Frq_b[leaf.index]
        if GPU and 2**len(features_in_path) < len(fl): # this trim is needed only on GPU
            fl = fl[:2**len(features_in_path)]
        for feature in matrixes:
            # The matrix multiplication part is implemented in CPU, the matrix is too small for the GPU overhead to be worth it.
            # The sparse matrix multiplication here instade of the naive dense matrix multiplication is improvement 1 in Sec. 9.1
            features_to_values[feature] = matrixes[feature].dot(fl) * leaf.value
        leaf.feature_contribution_replacement_values = features_to_values
    return tree


def get_cupy_data(trees: List[DecisionTreeNode], df: pd.DataFrame):
    """
    Cast the dataframe to cupy dict mapping between columns of CuPy arrays.
    We only do this for feature partisipating in the trees.
    """
    data = {}
    for tree in trees:
        for feature in tree.get_all_features():
            if feature not in data:
                data[feature] = cp.asarray(df[feature].to_numpy())
    return data


def calculation_given_preprocessed_tree(tree: DecisionTreeNode, data: pd.DataFrame, shapley_values = None, depth: int = 6, GPU=False):
    """
    Use the preprocessing to efficiently calculate the desired metric (Shapley/Banzahf values or interaction values)
    Implements lines 22-27 of the pseudo-code
    """
    # line 22 of the pseudo-code
    decision_patterns = calc_decision_patterns(tree, data, depth, GPU)

    # lines 23-27 of the pseudo-code
    if shapley_values is None:
        shapley_values = {}

    for almost_leaf in tree.get_all_almost_leaves():
        if not almost_leaf.right.is_leaf() or not almost_leaf.left.is_leaf():
            # If only the right or the left node is a leaf use s as is
            leaf = almost_leaf.right if almost_leaf.right.is_leaf() else almost_leaf.left
            current_edp_indexes = decision_patterns[leaf.index]
            replacements_arrays = leaf.feature_contribution_replacement_values
        else:
            # If both the right and left nodes are leaves use improvement 3 of Sec. 9.1 (improvement of line 26)
            # See also the comment in preprocess_tree_background.
            # Given leaves l_i, l_{i+1} s.t. there is an inner node n where n.left=l_i and n.right=l_{i+1}.
            # mark the s vector of feature f and leaf l_i as s_i = [a1,a2,a3,...,an]
            # mark the s vector of feature f and leaf l_{i+1} as s_{i+1} = [b1,b2,b3,...,bn]
            # A trivial numpy indexing for feature f and the two leaves is [a1,a2,a3,...,an][ patterns ] + [b1,b2,b3,...,bn][ patterns ]
            # Utilizing the property explained in comment in preprocess_tree_background, we can run the equivalent numpy indexing:
            # [a1+b2, a2+b1, a3+b4, a4+b3,...,a_{n-1}+bn, an+b_{n-1}][ patterns ]
            # This saves half of the numpy indexing opperations
            current_edp_indexes = decision_patterns[almost_leaf.left.index]
            replacements_arrays = almost_leaf.left.feature_contribution_replacement_values
            for feature, replacement_values in almost_leaf.right.feature_contribution_replacement_values.items():
                values = []
                for i in range(0, len(replacement_values), 2):
                    values.append(replacement_values[i+1])
                    values.append(replacement_values[i])

                if feature not in replacements_arrays:
                    replacements_arrays[feature] = np.array(values, dtype=np.float32)
                else:
                    replacements_arrays[feature] = np.array(values, dtype=np.float32) + replacements_arrays[feature]

        for feature, replacement_values in replacements_arrays.items():
            if GPU:
                replacements_array = cp.asarray(replacement_values)
            else:
                replacements_array = np.ascontiguousarray(replacement_values)

            # This is where the numpy indexing occur (improvement 6 of Sec. 9.1):
            current_shap_contribution = replacements_array[current_edp_indexes]

            if feature not in shapley_values:
                shapley_values[feature] = current_shap_contribution
            else:
                shapley_values[feature] += current_shap_contribution

    return shapley_values

def shapley_value_calculation_given_preprocessed_tree_ensemble(
        preprocess_trees: List[DecisionTreeNode], consumer_data: pd.DataFrame, global_importance: bool = False, iv_one_sized: bool = False, GPU=False):
    """
    Run desired metric calculation on a decision tree ensemble.

    @param global_importance: Interation values can quickly fill up all the machine RAM, as there are quadratic number of them.
    To be able to run the algorithm on large datasets, when global_importance=True, we save only their sum of mean absolute values across the trees.
    While it makes the result not useful it let us run WOODELF on large datasets and test its running time.
    """
    shapley_values = {}
    for tree in tqdm(preprocess_trees, desc="Computing SHAP"):
        if global_importance:
            current_shapley_values = {}
            calculation_given_preprocessed_tree(tree, consumer_data, shapley_values=current_shapley_values, GPU=GPU)
            for key in current_shapley_values:
                if key not in shapley_values:
                    shapley_values[key] = 0
                shapley_values[key] += np.abs(current_shapley_values[key]).sum() / len(current_shapley_values[key])
        else:
            calculation_given_preprocessed_tree(tree, consumer_data, shapley_values=shapley_values, GPU=GPU)

    # Improvement 4 of Sec. 9.1
    if iv_one_sized:
        all_keys = list(shapley_values.keys())
        for f1, f2 in all_keys:
            assert (f2,f1) not in shapley_values
            shapley_values[(f2, f1)] = shapley_values[(f1, f2)]

    return shapley_values

def calculate_background_shap(model: xgb.Booster, consumer_data: pd.DataFrame, background_data: pd.DataFrame, metric: CubeCharacteristicFunctionMetric, global_importance: bool = False, GPU=False):
    """
    The WOODELF algorithm!!!

    Gets an XGBoost regressor, consumer data of size n, background data for size m and a desired metric to calculate.
    Compute the desired metric in O(n+m)
    """
    model_objs = load_model(model, list(consumer_data.columns))
    path_to_matrixes_calculator = PathToValuesMatrix(metric=metric)
    if GPU:
        consumer_data = get_cupy_data(model_objs, consumer_data)
        background_data = get_cupy_data(model_objs, background_data)
    preprocessed_trees = []
    for tree in tqdm(model_objs, desc="Preprocessing the trees"):
        preprocessed_trees.append(preprocess_tree_background(tree, background_data, depth=tree.depth, path_to_matrixes_calculator=path_to_matrixes_calculator, GPU=GPU))

    print(f"cache misses: {path_to_matrixes_calculator.cache_miss}, cache used: {path_to_matrixes_calculator.cached_used}")
    shapley = shapley_value_calculation_given_preprocessed_tree_ensemble(
        preprocessed_trees, consumer_data, global_importance, iv_one_sized = metric.INTERACTION_VALUE_ONE_SIDE, GPU=GPU
    )
    return shapley

def path_dependend_frequencies(tree: DecisionTreeNode, depth):
    """
    Estimate the frequencies of the training data using the tree cover property.
    Implement Formula 9 of the article for all the leaves in the provided tree.
    """
    if tree.is_leaf():
        return {tree.index: []}

    leaves_freq_dict = {}
    inner_nodes_freq_dict = {}
    inner_nodes_freq_dict[tree.index] = [1]
    for current_node in tree.bfs():
        current_node_freq = inner_nodes_freq_dict[current_node.index]
        if current_node.is_leaf():
            leaves_freq_dict[current_node.index] = np.array(
                inner_nodes_freq_dict[current_node.index], dtype=np.float32
            )
            continue

        freqs_l = []
        for freq in current_node_freq:
            freqs_l.append((current_node.right.cover/current_node.cover) * freq)
            freqs_l.append((current_node.left.cover/current_node.cover) * freq)
        inner_nodes_freq_dict[current_node.left.index] = freqs_l

        freqs_r = []
        for freq in current_node_freq:
            # Changed the order of the 2 lines here, now left is first.
            freqs_r.append((current_node.left.cover/current_node.cover) * freq)
            freqs_r.append((current_node.right.cover/current_node.cover) * freq)
        inner_nodes_freq_dict[current_node.right.index] = freqs_r
    return leaves_freq_dict

def fast_preprocess_path_dependent_shap(tree: DecisionTreeNode, path_to_matrixes_calculator: PathToValuesMatrix, depth=6):
    """
    Implement the preprocssing needed for Path-Dependent WOODELF
    """
    freq = path_dependend_frequencies(tree, depth)
    for leaf, features_in_path in tree.get_all_leaves_with_path_to_root():
        matrixes = path_to_matrixes_calculator.get_values_matrixes(features_in_path)
        features_to_values = {}
        for feature in matrixes:
            features_to_values[feature] = matrixes[feature].dot(freq[leaf.index]) * leaf.value
        leaf.feature_contribution_replacement_values = features_to_values
    return tree


def calculate_path_dependent_shap(model, consumer_data, metric: CubeCharacteristicFunctionMetric, global_importance: bool = False, GPU=False):
    """
    Path-Dependent WOODELF algorithm!!

    Given a model, a consumer data and a desired metric compute the metric under the Path-Dependent assumptions.
    """
    model_objs = load_model(model, list(consumer_data.columns))
    path_to_matrixes_calculator = PathToValuesMatrix(metric=metric)
    if GPU:
        consumer_data = get_cupy_data(model_objs, consumer_data)

    preprocessed_trees = []
    for tree in tqdm(model_objs, desc="Preprocessing the trees"):
        preprocessed_trees.append(fast_preprocess_path_dependent_shap(tree, path_to_matrixes_calculator=path_to_matrixes_calculator))

    print(f"cache misses: {path_to_matrixes_calculator.cache_miss}, cache used: {path_to_matrixes_calculator.cached_used}")
    return shapley_value_calculation_given_preprocessed_tree_ensemble(preprocessed_trees, consumer_data, global_importance, iv_one_sized=metric.INTERACTION_VALUE_ONE_SIDE, GPU=GPU)
