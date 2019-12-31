import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import display, Image


def decision_paths(clf, feature_list, is_print=False):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    if is_print:
        print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if %s<=%s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature_list[feature[i]],
                         round(threshold[i], 2),
                         children_right[i],
                         ))
            
    decision_path = dict({0: None})
    decision_depth = dict()
    for i in range(n_nodes):
        decision_depth.update({i: node_depth[i]})
        if not is_leaves[i]:
            left_node = children_left[i]
            right_node = children_right[i]
            left_condition = (decision_path.get(i)+' and ' if decision_path.get(
                i) else '')+feature_list[feature[i]]+'<='+str(round(threshold[i], 2))
            right_condition = (decision_path.get(i)+' and ' if decision_path.get(
                i) else '')+feature_list[feature[i]]+'>'+str(round(threshold[i], 2))
            decision_path.update({left_node: left_condition})
            decision_path.update({right_node: right_condition})

    return decision_path, decision_depth


def leaf_batches(clf, x, y):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    leaf_idx = np.array(range(n_nodes))[is_leaves]

    batch_overdue = [(y[clf.apply(x) == i] > 0).sum()/len(y[clf.apply(x) == i])
                     if len(y[clf.apply(x) == i]) > 0 else np.nan for i in leaf_idx]
    batch_num = [len(y[clf.apply(x) == i]) for i in leaf_idx]
    proportion = [len(y[clf.apply(x) == i])/len(x)if len(x)
                  > 0 else np.nan for i in leaf_idx]
    overdue_num = [(y[clf.apply(x) == i] > 0).sum()for i in leaf_idx]
    leaf_overdue = pd.DataFrame([leaf_idx, batch_num, proportion, overdue_num, batch_overdue], index=[
                                'leaf_index', 'leaf_num', 'proportion', 'overdue_num', 'leaf_overdue']).T
    leaf_overdue.sort_values(by='leaf_overdue', ascending=False, inplace=True)
    leaf_overdue[['leaf_index', 'leaf_num', 'overdue_num']] = leaf_overdue[[
        'leaf_index', 'leaf_num', 'overdue_num']].astype(int)
    leaf_overdue = leaf_overdue.append({'leaf_index': 'TOTAL', 'leaf_num': len(
        y), 'proportion': 1, 'overdue_num': y.sum(), 'leaf_overdue': y.sum()/len(y)}, ignore_index=True)
    leaf_overdue['lift'] = leaf_overdue['leaf_overdue'] / (y.sum()/len(y))
    return leaf_overdue


if __name__ == "__main__":
    x, y = make_classification(
        n_samples=1000, n_features=45, n_informative=12, n_redundant=7, random_state=1)
    feature_list = [f'feature_{i}' for i in range(0, 45)]
    x = pd.DataFrame.from_records(x, columns=feature_list)
    y = pd.Series.from_array(y, name='label')
    
    clf = DecisionTreeClassifier(criterion='gini', max_depth=4,
                                 min_samples_leaf=0.01, splitter='best', random_state=1)
    clf.fit(x, y)
    
    decision_path, decision_depth = decision_paths(clf, feature_list, True)
    leaf_batch = leaf_batches(clf, x, y)
    leaf_batch['decision_path'] = leaf_batch['leaf_index'].map(
        decision_path.get)
    print(leaf_batch)
    
#     kwargs = {'feature_names': feature_list, 'class_names': [
#         '0', '1'], 'filled': True, 'rounded': True, 'special_characters': True}
#     dot_data = export_graphviz(clf, out_file=None, **kwargs)
#     export_graphviz(clf, out_file='tree.dot', **kwargs)
#     # with open("tree.dot", 'w') as f:
#     #     f = export_graphviz(clf, out_file=f)
#     with open('tree.dot', 'r') as f:
#         dot_data = f.read()
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_png('tree.png')
#     img = Image(graph.create_png())
#     display(img)
