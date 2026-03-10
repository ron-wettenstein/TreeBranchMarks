import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, Tuple, Set, List
from math import factorial
import time
from copy import copy
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

from shap.explainers._tree import XGBTreeModelLoader # Use the shap package's XGBoost loading. this is cheating, I know...
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
        self.parent_index = -1
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

    def is_leaf(self):
        return self.right is None and self.left is None

    def predict(self, data):
        if self.is_leaf():
            return pd.Series(self.value, index=data.index)
        return self.shall_go_left(data) * self.left.predict(data) + self.shall_go_right(data) * self.right.predict(data)

    def bfs(self, including_myself: bool = True, including_leaves: bool = True):
        """
        Return all the node children (and the node itself) in BFS order. The indexes should be in an increasing order.
        """
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

    def __repr__(self):
        if self.is_leaf():
            return f"{self.index} (cover: {self.cover}): leaf with value {self.value}"
        return f"{self.index} (cover: {self.cover}): {self.feature_name} < {self.value}"

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
    left.parent_index = node.index
    right.parent_index = node.index
    return node


def load_model(model, features):
    """
    Load an XGBoost regressor tree (utilizing the shap python package parsing object)
    """
    model_obj = load_decision_tree_ensemble_model(model, features)
    return [cast_tree_format(tree) for tree in model_obj.trees]

def nCk(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

class WDNFCharacteristicFunctionMetric(object):
    """
    An abstruct class that calculate a metric on a WDNF/WCNF characteristic function.
    You can implement this class (override the calc_metric function) and then use this class and the WOODELF algorithm
    to calculate your metric effiecently on large background datasets.

    Here, the metrics that inherit this class are: Shapley values, Shapley interaction values, Banzhaf values and Banzhaf interaction values
    """
    INTERACTION_VALUE = False

    def calc_metric(self, wdnf):
        raise NotImplemented()

class ShapleyValues(WDNFCharacteristicFunctionMetric):
    """
    Implement the linear-time formula for Shapley value computation on WDNF/WCNF, see Formula 3 in the paper.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, wdnf):
        shapley_values = {}
        for weight, se, sne in wdnf:
            if len(se & sne) > 0:
                continue # se and sne must be disjoint sets

            for feature in se | sne:
                if feature not in shapley_values:
                    shapley_values[feature] = 0

            # The new simple shapley values formula
            if len(se) > 0:
                se_contribution = (weight / (len(se) * nCk(len(se) + len(sne), len(se))))
                for must_exist_feature in se:
                    shapley_values[must_exist_feature] += se_contribution
            if len(sne) > 0:
                sne_contribution = (-weight / (len(sne) * nCk(len(se) + len(sne), len(sne))))
                for must_be_missing_feature in sne:
                    shapley_values[must_be_missing_feature] += sne_contribution

        return shapley_values

class ShapleyInteractionValues(WDNFCharacteristicFunctionMetric):
    """
    Implement the formulas for Shapley interaction values computation on WDNF/WCNF, see Table 1 in the paper.
    """
    INTERACTION_VALUE = True

    def calc_metric(self, wdnf):
        shapley_values = defaultdict(int)
        for weight, se, sne in wdnf:
            if len(se & sne) > 0:
                continue # se and sne must be disjoint sets

            weight = weight / 2 # The shapley interaction values in the shap package are actually divided by 2....

            s = se | sne
            if len(se) > 0:
                # i,j in S+
                if len(se) > 1:
                    contribution = weight / ((len(se) - 1) * nCk(len(s) - 1, len(se) - 1))
                    for must_exists_feature in se:
                        for other_feature in se:
                            if must_exists_feature == other_feature:
                                continue

                            shapley_values[(must_exists_feature, other_feature)] += contribution

                # i in S+   j in S-
                if len(sne) > 0:
                    contribution = -weight / (len(sne) * nCk(len(s) - 1, len(sne)))
                    for must_exists_feature in se:
                        for other_feature in sne:
                            shapley_values[(must_exists_feature, other_feature)] += contribution

            if len(sne) > 0:
                # i,j in S-
                if len(sne) > 1:
                    contribution = weight / ((len(sne) - 1) * nCk(len(s) - 1, len(sne) - 1))
                    for must_be_missing_feature in sne:
                        for other_feature in sne:
                            if must_be_missing_feature == other_feature:
                                continue

                            shapley_values[(must_be_missing_feature, other_feature)] += contribution
                # i in S-   j in S+
                if len(se) > 0:
                    contribution = -weight / (len(se) * nCk(len(s) - 1, len(se)))
                    for must_be_missing_feature in sne:
                        for other_feature in se:
                            shapley_values[(must_be_missing_feature, other_feature)] += contribution
        return shapley_values


class BanzahfValues(WDNFCharacteristicFunctionMetric):
    """
    Implement the linear-time formula for Banzhaf value computation on WDNF/WCNF, see Formula 6 in the paper.
    """
    INTERACTION_VALUE = False

    def calc_metric(self, wdnf):
        banzhaf_values = {}
        for weight, se, sne in wdnf:
            if len(se & sne) > 0:
                continue # se and sne must be disjoint sets

            for feature in se | sne:
                if feature not in banzhaf_values:
                    banzhaf_values[feature] = 0

            # The new simple banzhaf values formula
            contribution = (weight / (2 ** (len(se) + len(sne) - 1)))
            if len(se) > 0:
                for must_exist_feature in se:
                    banzhaf_values[must_exist_feature] += contribution
            if len(sne) > 0:
                for must_be_missing_feature in sne:
                    banzhaf_values[must_be_missing_feature] -= contribution # Note: use -= (not +=) which is like multipling with -1

        return banzhaf_values


class BanzhafInteractionValues(WDNFCharacteristicFunctionMetric):
    """
    Implement the formulas for Banzhaf interaction values computation on WDNF/WCNF, see Formula 7 in the paper.
    """
    INTERACTION_VALUE = True

    def calc_metric(self, wdnf):
        banzhaf_values = defaultdict(int)
        for weight, se, sne in wdnf:
            if len(se & sne) > 0:
                continue # se and sne must be disjoint sets

            weight = weight / 2 # The shapley interaction values in the shap package are actually divided by 2....

            contribution = (weight / (2 ** (len(se) + len(sne) - 2)))

            s = se | sne
            if len(se) > 0:
                # i,j in S+
                if len(se) > 1:
                    for must_exists_feature in se:
                        for other_feature in se:
                            if must_exists_feature == other_feature:
                                continue

                            banzhaf_values[(must_exists_feature, other_feature)] += contribution

                # i in S+   j in S-
                if len(sne) > 0:
                    for must_exists_feature in se:
                        for other_feature in sne:
                            banzhaf_values[(must_exists_feature, other_feature)] -= contribution

            if len(sne) > 0:
                # i,j in S-
                if len(sne) > 1:
                    for must_be_missing_feature in sne:
                        for other_feature in sne:
                            if must_be_missing_feature == other_feature:
                                continue

                            banzhaf_values[(must_be_missing_feature, other_feature)] += contribution
                # i in S-   j in S+
                if len(se) > 0:
                    for must_be_missing_feature in sne:
                        for other_feature in se:
                            banzhaf_values[(must_be_missing_feature, other_feature)] -= contribution
        return banzhaf_values

def get_pattern_index(pattern):
    """
    Encode a decision pattern as an integer by replacing 'T' with 1, 'F' with 0, and treat the result as a binary number.
    """
    return int(pattern.replace("T", "1").replace("F", "0"), base=2)


def unite_wdnf(root: DecisionTreeNode, VC, background_data_size):
    """
    Step 3 of WOODELF

    Given the pd_pb_to_cube mapping and the background decision patterns value counts, build a united WDNF formula for each leaf and consumer pattern.
    In other words: compute Formula 12 of the paper.
    """
    for leaf in root.get_all_leaves():
        if leaf.pc_pb_to_cube is None:
            return

        VC_leaf = VC[leaf.index]

        characteristic_wdnfs = {}
        for consumer_pattern in leaf.pc_pb_to_cube:
            characteristic_wdnfs[consumer_pattern] = []
            for background_pattern, (se, sne) in leaf.pc_pb_to_cube[consumer_pattern].items():
                background_pattern_index = get_pattern_index(background_pattern)

                cover = (VC_leaf[background_pattern_index] * 1.0) / background_data_size
                weight = cover * leaf.value
                characteristic_wdnfs[consumer_pattern].append( (weight, se, sne) )

        leaf.consumer_pattern_to_characteristic_wdnf = characteristic_wdnfs
        leaf.pc_pb_to_cube = None # To save RAM


def calculate_metric_from_wdnf(root: DecisionTreeNode, metric: WDNFCharacteristicFunctionMetric):
    """
    Step 4 of WOODELF

    Compute the desired metric (Banzhaf/Shapley values or interaction values) on the united WDNF from Step 3
    (the WDNF that were saved in leaf.consumer_pattern_to_characteristic_wdnf)
    """
    for leaf in root.get_all_leaves():
        if leaf.consumer_pattern_to_characteristic_wdnf is None:
            return

        index_to_metric_value_dict = {}
        all_consumer_pattern_indexes = [get_pattern_index(pattern) for pattern in leaf.consumer_pattern_to_characteristic_wdnf.keys()]
        for consumer_pattern in leaf.consumer_pattern_to_characteristic_wdnf:
            wdnf = leaf.consumer_pattern_to_characteristic_wdnf[consumer_pattern]
            values = metric.calc_metric(wdnf)
            pattern_index = get_pattern_index(consumer_pattern)
            for feature in values:
                if feature not in index_to_metric_value_dict:
                    index_to_metric_value_dict[feature] = {index: 0 for index in all_consumer_pattern_indexes}
                index_to_metric_value_dict[feature][pattern_index] = values[feature]

        leaf.feature_contribution_replacement_values = index_to_metric_value_dict
        leaf.consumer_pattern_to_characteristic_wdnf = None # To save RAM

def map_patterns_to_cube(tree: DecisionTreeNode, current_wdnf_table: Optional[Dict[str, Dict[str, Tuple[Set, Set]]]] = None):
    """
    The function MapPatternsToCube from Sect. 6 of the article.
    :params tree: The decision tree
    :params current_wdnf_table: The format is: wdnf_table[consumer_decision_pattern][background_decision_pattern] = (cube_positive_literals, cube_negative_literals)
    """
    if current_wdnf_table is None:
        # At the tree root initilaize the wdnf dict with empty consumer and background patterns and an empty clause
        current_wdnf_table = {"": {"": (set(), set())}}
    if tree.is_leaf():
        # If we reached a leaf update its patterns table
        tree.pc_pb_to_cube = current_wdnf_table
        return

    left_updated_wdnf_table = {}
    right_updated_wdnf_table = {}
    for consumer_pattern in current_wdnf_table:
        left_updated_wdnf_table[consumer_pattern + "F"] = {}
        left_updated_wdnf_table[consumer_pattern + "T"] = {}
        right_updated_wdnf_table[consumer_pattern + "F"] = {}
        right_updated_wdnf_table[consumer_pattern + "T"] = {}
        for background_pattern in current_wdnf_table[consumer_pattern]:
            # Get the current cube (the possitive and negated literals) of the consumer and background patterns
            s_plus, s_minus = current_wdnf_table[consumer_pattern][background_pattern]
            # Implement the 4 rules
            left_updated_wdnf_table[consumer_pattern + "T"][background_pattern + "F"] = (s_plus | {tree.feature_name}, s_minus) # Rule 1
            left_updated_wdnf_table[consumer_pattern + "F"][background_pattern + "T"] = (s_plus, s_minus | {tree.feature_name}) # Rule 2
            left_updated_wdnf_table[consumer_pattern + "T"][background_pattern + "T"] = (s_plus, s_minus) # Rule 3
            # Implement the 4 rules (its like the left rules but negated T->F, F->T)
            right_updated_wdnf_table[consumer_pattern + "F"][background_pattern + "T"] = (s_plus | {tree.feature_name}, s_minus) # Rule 1
            right_updated_wdnf_table[consumer_pattern + "T"][background_pattern + "F"] = (s_plus, s_minus | {tree.feature_name}) # Rule 2
            right_updated_wdnf_table[consumer_pattern + "F"][background_pattern + "F"] = (s_plus, s_minus) # Rule 4

    map_patterns_to_cube(tree.left, left_updated_wdnf_table)
    map_patterns_to_cube(tree.right, right_updated_wdnf_table)


def get_int_dtype_from_depth(depth):
    """
    The decision pattern, when encoded as a number, have a bit for each node of the root-to-leaf-path.
    Choose the dtype according to the tree depth (a.k.a the max pattern length)
    """
    if depth <= 8:
        return np.uint8
    if depth <= 16:
        return np.uint16
    if depth <= 32:
       return np.uint32
    return np.uint64


def calc_decision_patterns(tree, data, depth):
    """
    An effiecent implementation of the CalcDecisionPatterns from Sec. 5 of the paper.
    Instead of a string representation we use a numerical one: "TTFFTF" becomes "110010" and treated as a number.
    The main change is now instead of appending 'T'/'F' to the parent node we apply a shift left on the parent number and add 1/0:
    In string representation:  "TTFFT" + "F" = "TTFFTF"
    In numeric representation: (11001b << 1) + 0b = 110011
    """
    if tree.is_leaf():
        return pd.DataFrame({"0": pd.Series(0, index=data.index)})

    # Use a tight uint type for efficiency
    int_dtype = get_int_dtype_from_depth(depth)

    leaves_patterns_dict = {} # This is the P mentioned in the paper
    inner_nodes_patterns_dict = {}
    inner_nodes_patterns_dict[tree.index] = tree.shall_go_left(data).to_numpy().astype(int_dtype)
    for current_node in tree.bfs(including_myself = False, including_leaves = True):
        parent_pattern = inner_nodes_patterns_dict[current_node.parent_index]
        if current_node.is_leaf():
            leaves_patterns_dict[current_node.index] = parent_pattern
        else:
            split_series = current_node.shall_go_left(data).to_numpy().astype(int_dtype)
            inner_nodes_patterns_dict[current_node.index] = (parent_pattern << 1) + split_series
    return leaves_patterns_dict


def preprocess_tree_background(tree: DecisionTreeNode, background_data: pd.DataFrame, depth: int, metric: WDNFCharacteristicFunctionMetric):
    """
    Run all the preprocessing needed given a tree and a background_data.
    The preprocssing include Steps 1-4 of the algorithm (Also see Figure 4 of the paper)
    """
    # Step 1
    map_patterns_to_cube(tree)

    # Step 2
    background_patterns_matrix = calc_decision_patterns(tree, background_data, depth)
    VCb = {}
    for leaf in tree.get_all_leaves():
        # np.bincount is a faster way to implement value counts that uses the fact all decision patterns are integers between 0 and 2**depth
        VCb[leaf.index] = np.bincount(background_patterns_matrix[leaf.index], minlength=2**depth)

    # Step 3
    unite_wdnf(tree, VCb, len(background_data))

    # Step 4
    calculate_metric_from_wdnf(tree, metric)
    return tree


def calcaltion_given_preprocessed_tree(tree: DecisionTreeNode, data: pd.DataFrame, shapley_values = None, depth: int = 6):
    """
    Use the preprocessing to efficiently calculate the desired metric (Shapley/Banzahf values or interaction values)
    Runs Steps 5 and 6 of WOODELF
    """

    # Step 5
    decision_patterns = calc_decision_patterns(tree, data, depth)

    # Step 6
    if shapley_values is None:
        shapley_values = {}

    for leaf in tree.get_all_leaves():
        current_edp_indexes = decision_patterns[leaf.index]
        for feature, replacement_values in leaf.feature_contribution_replacement_values.items():
            # Instead of patterns.map(shapley_values_dict) we use the fact that all decision patterns are integers between 0 and 2**depth.
            # We use numpy indexing. Save all the Shapley values in an array 'shapley_values_array' with the index equal the decision pattern
            # ("FFF" will be the first item, "FFT" will be the second item, "FTF" will be the third item and so on)
            # Then instead of patterns.map(shapley_values_dict) we use the much more efficient shapley_values_array[patterns]
            max_index = max(replacement_values.keys())
            replacements_array = np.array([replacement_values[i] for i in range(max_index+1)])
            replacements_array = np.ascontiguousarray(replacements_array.astype(np.float32))
            current_shap_contribution = replacements_array[current_edp_indexes]

            if feature not in shapley_values:
                shapley_values[feature] = current_shap_contribution
            else:
                shapley_values[feature] += current_shap_contribution

    return shapley_values

def shapley_value_calcaltion_given_preprocessed_tree_ensemble(preprocess_trees: List[DecisionTreeNode], consumer_data: pd.DataFrame, global_importance: bool = False):
    """
    Run desired metric calculation (Steps 5 and 6) on a decision tree ensemble.

    @param global_importance: Interation values can quickly fill up all the machine RAM, as there are quadratic number of them.
    To be able to run the algorithm on large datasets, when global_importance=True, we save only their sum of mean absolute values across the trees.
    While it makes the result not usefull it let us run WOODELF on large datasets and test its running time.
    """
    shapley_values = {}
    for tree in tqdm(preprocess_trees, desc="Computing SHAP (Steps 5 and 6)"):
        if global_importance:
            current_shapley_values = {}
            calcaltion_given_preprocessed_tree(tree, consumer_data, shapley_values=current_shapley_values)
            for key in current_shapley_values:
                if key not in shapley_values:
                    shapley_values[key] = 0
                shapley_values[key] += np.abs(current_shapley_values[key]).sum() / len(current_shapley_values[key])
        else:
            calcaltion_given_preprocessed_tree(tree, consumer_data, shapley_values=shapley_values)

    return shapley_values

def calculate_background_shap(model: xgb.Booster, consumer_data: pd.DataFrame, background_data: pd.DataFrame, metric: WDNFCharacteristicFunctionMetric, global_importance: bool = False):
    """
    The WOODELF algorithm!!!

    Gets an XGBoost regressor, consumer data of size n, background data for size m and a desired metric to calculate.
    Compute the desired metric in O(n+m)
    """
    model_v_objs = load_model(model, list(consumer_data.columns))
    preprocessed_trees = []
    for tree in tqdm(model_v_objs, desc="Preproccing the trees (Steps 1-4)"):
        preprocessed_trees.append(preprocess_tree_background(tree, background_data, depth=tree.depth, metric=metric))
    return shapley_value_calcaltion_given_preprocessed_tree_ensemble(preprocessed_trees, consumer_data, global_importance)


def path_dependend_frequencies(tree: DecisionTreeNode, depth):
    """
    Estimate the frequencies of the training data using the tree cover property.
    Implement Formula 13 of the supplementary material
    """
    if tree.is_leaf():
        return {tree.index: []}

    leaves_freq_dict = {}
    inner_nodes_freq_dict = {}
    inner_nodes_freq_dict[tree.index] = [tree.right.cover/tree.cover, tree.left.cover/tree.cover]
    nodes_to_visit = [tree]
    while len(nodes_to_visit) > 0:
        current_node = nodes_to_visit.pop(0)
        current_node_freq = inner_nodes_freq_dict[current_node.index]
        for next_node in [current_node.right, current_node.left]:
            if next_node.is_leaf():
                leaves_freq_dict[next_node.index] = current_node_freq
            else:
                freqs = []
                for freq in current_node_freq:
                    freqs.append((next_node.right.cover/next_node.cover) * freq)
                    freqs.append((next_node.left.cover/next_node.cover) * freq)

                inner_nodes_freq_dict[next_node.index] = freqs
                nodes_to_visit.append(next_node)

    int_dtype = get_int_dtype_from_depth(depth)
    for almost_leaf_index in leaves_freq_dict:
        leaves_freq_dict[almost_leaf_index] = np.array(leaves_freq_dict[almost_leaf_index])
    return leaves_freq_dict

def fast_preprocess_path_dependent_shap(tree: DecisionTreeNode, metric: WDNFCharacteristicFunctionMetric, depth=6):
    """
    Implement the preprocssing needed for Path-Dependent WOODELF
    """
    map_patterns_to_cube(tree) # Step 1 of WOODELF
    freq = path_dependend_frequencies(tree, depth) # Step 2 of Path-Dependent WOODELF (modified the original step)
    unite_wdnf(tree, freq, background_data_size=1) # Step 3 of Path-Dependent WOODELF (modified the original step by setting background_data_size=1)
    calculate_metric_from_wdnf(tree, metric) # Step 4 of WOODELF
    return tree


def calculate_path_dependent_shap(model, consumer_data, metric: WDNFCharacteristicFunctionMetric, global_importance: bool = False):
    """
    Path-Dependent WOODELF algorithm!!

    Given a model, a consumer data and a desired metric compute the metric under the Path-Dependent assumptions.
    """
    model_v_objs = load_model(model, list(consumer_data.columns))
    preprocessed_trees = []
    for tree in tqdm(model_v_objs, desc="Preproccing the trees (Steps 1-4)"):
        preprocessed_trees.append(fast_preprocess_path_dependent_shap(tree, metric=metric))
    return shapley_value_calcaltion_given_preprocessed_tree_ensemble(preprocessed_trees, consumer_data, global_importance)


