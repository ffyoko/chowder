# py397+econml==0.12.0/causalml=0.15.1
import re, joblib, numpy as np, pandas as pd


def uplift_tree_string(decisionTree, x_names):
	dcHeadings = {}
	for i, szY in enumerate(x_names + ["treatment_group_key"]):
		szCol = "Column %d" % i
		dcHeadings[szCol] = str(szY)
	def toString(decisionTree, indent=""):
		if decisionTree.results is not None:
			return str(decisionTree.results)
		else:
			szCol = "Column %s" % decisionTree.col
			if szCol in dcHeadings:
				szCol = dcHeadings[szCol]
			if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
				decision = "%s >= %s?" % (szCol, decisionTree.value)
			else:
				decision = "%s == %s?" % (szCol, decisionTree.value)
			trueBranch = (indent + "yes -> " + toString(decisionTree.trueBranch, indent + "\t\t"))
			falseBranch = (indent + "no  -> " + toString(decisionTree.falseBranch, indent + "\t\t"))
			return decision + "\n" + trueBranch + "\n" + falseBranch
	print(toString(decisionTree))


def uplift_tree_cate(decisionTree, x_names):
	dcHeadings = {}
	g = globals()
	g['dcPathsList'] = []
	for i, szY in enumerate(x_names + ["treatment_group_key"]):
		szCol = "Column %d" % i
		dcHeadings[szCol] = str(szY)
	def toString(decisionTree, parent=""):
		if decisionTree.results is not None:
			dcPathsList.append("dcPaths" + "".join(".setdefault('%s',dict())" % j for j in parent.split("|")))
			dcPathsList.append("dcPaths" + "".join(".get('%s')" % j for j in parent.split("|")) + ".update({'TE': %s})" % str(decisionTree.results))
			# for i, TE in enumerate(decisionTree.results):
			# 	dcPathsList.append("dcPaths" + "".join(".get('%s')" % j for j in parent.split("|")) + ".update({%s: %s})" % (i, TE))
		else:
			szCol = "Column %s" % decisionTree.col
			if szCol in dcHeadings:
				szCol = dcHeadings[szCol]
			if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
				trueDecision = "%s >= %s" % (szCol, decisionTree.value)
				falseDecision = "%s < %s" % (szCol, decisionTree.value)
			else:
				trueDecision = "%s == %s" % (szCol, decisionTree.value)
				falseDecision = "%s == %s" % (szCol, decisionTree.value)
			path = {
				trueDecision: toString(decisionTree.trueBranch, parent="|".join([parent, trueDecision])),
				falseDecision: toString(decisionTree.falseBranch, parent="|".join([parent, falseDecision]))
			}
			# dcPathsList.append("dcPaths" + "".join(".get('%s')" % j for j in parent.split("|")) + ".update(%s)" % path)
	toString(decisionTree)
	dcPaths = {"": {}}
	for i in g['dcPathsList']:
		eval(i)
	dcPaths = pd.json_normalize(dcPaths, sep=' and ').T
	dcPaths.index = dcPaths.index.str.replace(' and  and | and TE', '')
	return dcPaths[0].apply(pd.Series, index=[0, 1])


task = 'mitigate'
pwd = '/home/j-fandi-jk/mitigate_20240722/'
uplift_rf = joblib.load(filename=pwd + f'uplift_rf_{task}_20240903')
column_limit = joblib.load(filename=pwd + f'column_limit_{task}_20240903')
paths_frame = pd.DataFrame()
for i in range(uplift_rf.n_estimators):
	uplift_tree = uplift_rf.uplift_forest[i]
	paths = uplift_tree_cate(uplift_tree.fitted_uplift_tree, column_limit)
	paths.reset_index(drop=False, inplace=True)
	paths['n_estimator'] = i
	paths_frame = pd.concat([paths_frame, paths], axis=0) if len(paths_frame)>0 else paths
pattern_list = column_limit[::]
pattern_list = sorted(pattern_list)
for i in pattern_list:
	index_i = pattern_list.index(i)
	for j in pattern_list[index_i+1:]:
		if re.match(i, j) is not None:
			index_j = pattern_list.index(j)
			pattern_list.insert(index_j, pattern_list.pop(index_i))
pattern = re.compile('|'.join(pattern_list))
feat_dict = pd.read_csv(pwd + 'feat_dict', sep='\t')
feature2note = dict(feat_dict[['alias', 'comment']].applymap(str.lower).apply(tuple, axis=1).tolist())
feature2module = dict(feat_dict[['alias', 'from']].applymap(str.lower).apply(tuple, axis=1).tolist())
paths_parse = dict()
for i in paths_frame['index']:
	paths_parse.update({i: pattern.sub(lambda matched: feature2note[matched.group(0)], i)})
paths_frame['parse'] = paths_frame['index'].map(paths_parse)
paths_frame.round(4).to_csv(pwd + f'paths_{task}')
paths_parse.get('offline_inner_jtbeh_noviprepay_before3mpayfailedamtratio < 0.97732013441 and offline_inner_jtbeh_debt_currentnosettlemloanprinamt < 36400.0 and beforedraw12moverdue3daysamt < 3316.82 and offline_inner_jtbeh_overdue_maxoverduedays >= 19.0')




######################################################################################################################################################################################################
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
			  "the following tree structure:" % n_nodes)
		for i in range(n_nodes):
			if is_leaves[i]:
				print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
			else:
				print("%snode=%s test node: go to node %s if %s<=%s else to "
					  "node %s." % (
						  node_depth[i] * "\t",
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
			left_condition = (decision_path.get(i) +
							  ' and ' if decision_path.get(i) else
							  '') + feature_list[feature[i]] + '<=' + str(
								  round(threshold[i], 2))
			right_condition = (decision_path.get(i) +
							   ' and ' if decision_path.get(i) else
							   '') + feature_list[feature[i]] + '>' + str(
								   round(threshold[i], 2))
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
	batch_overdue = [
		(y[clf.apply(x) == i] > 0).sum() /
		len(y[clf.apply(x) == i]) if len(y[clf.apply(x) == i]) > 0 else np.nan
		for i in leaf_idx
	]
	batch_num = [len(y[clf.apply(x) == i]) for i in leaf_idx]
	proportion = [
		len(y[clf.apply(x) == i]) / len(x) if len(x) > 0 else np.nan
		for i in leaf_idx
	]
	overdue_num = [(y[clf.apply(x) == i] > 0).sum() for i in leaf_idx]
	leaf_overdue = pd.DataFrame(
		[leaf_idx, batch_num, proportion, overdue_num, batch_overdue],
		index=[
			'leaf_index', 'leaf_num', 'proportion', 'overdue_num',
			'leaf_overdue'
		]).T
	leaf_overdue.sort_values(by='leaf_overdue', ascending=False, inplace=True)
	leaf_overdue[['leaf_index', 'leaf_num', 'overdue_num'
				  ]] = leaf_overdue[['leaf_index', 'leaf_num',
									 'overdue_num']].astype(int)
	leaf_overdue = leaf_overdue.append(
		{
			'leaf_index': 'TOTAL',
			'leaf_num': len(y),
			'proportion': 1,
			'overdue_num': y.sum(),
			'leaf_overdue': y.sum() / len(y)
		},
		ignore_index=True)
	leaf_overdue['lift'] = leaf_overdue['leaf_overdue'] / (y.sum() / len(y))
	return leaf_overdue


def index_generator(length, step):
	body = np.arange(1, length // step + 1).repeat(step)
	tail = np.array(length // step).repeat(length % step)
	return np.append(body, tail)


def rule_reformer(rule, splitor_sup=False):
	import re
	feature_dict = {}
	splitor = '<|<=|==|>=|>| is missing' + \
		('|'+splitor_sup if splitor_sup is not False else '')
	for i in rule:
		for j in i.split(' and '):
			f, _ = re.split(splitor, j)
			f = f.strip()
			feature_dict.update({f: 'x["' + f + '"]'})
	pattern_list = list(feature_dict.keys())
	for i in pattern_list:
		index_i = pattern_list.index(i)
		for j in pattern_list[index_i+1:]:
			if re.match(i, j) is not None:
				index_j = pattern_list.index(j)
				pattern_list.insert(index_j, pattern_list.pop(index_i))
	pattern = re.compile(('(?=' + splitor + ')|').join(pattern_list))
	feature_list = [
		'(' + pattern.sub(lambda matched: feature_dict[matched.group(0)], i) +
		')' for i in rule
	]
	return ' or '.join(feature_list)


def decision_making_engine(x, rule):
	return int(eval(rule))


def rule_simulator(df, y, index, rule, splitor_sup=False):
	df['full'] = 1
	kwargs = {'rule': rule_reformer(rule, splitor_sup=splitor_sup)}
	df['hit'] = df.apply(decision_making_engine, **kwargs, axis=1)
	df[f'hit_{y}'] = df.apply(lambda x: 1
							  if (x['hit'] == 1 and x[f'{y}'] == 1) else 0,
							  axis=1)
	result = df.groupby(index)[['full', y, 'hit', f'hit_{y}']].sum()
	result.rename(columns={
		'full': 'full_num',
		y: f'{y}_num',
		'hit': 'hit_num',
		f'hit_{y}': f'hit_{y}_num'
	}, inplace=True)
	result[y] = result[f'{y}_num'] / result['full_num']
	result['hit'] = result['hit_num'] / result['full_num']
	result[f'hit_{y}'] = result[f'hit_{y}_num'] / result['hit_num']
	result['lift'] = result[f'hit_{y}'] / result[f'{y}']
	return result


def string_parser(s):
	if len(re.findall(r":leaf=", s)) == 0:
		out = re.findall(r"[\w.-]+", s)
		tabs = re.findall(r"[\t]+", s)
		if (out[4] == out[8]):
			missing_value_handling = (" or np.isnan(x['" + out[1] + "']) ")
		else:
			missing_value_handling = ""
		if len(tabs) > 0:
			return (re.findall(r"[\t]+", s)[0].replace('\t', '	') +
					'		if state == ' + out[0] + ':\n' +
					re.findall(r"[\t]+", s)[0].replace('\t', '	') +
					'			state = (' + out[4] + ' if ' + "x['" +
					out[1] + "']<" + out[2] + missing_value_handling +
					' else ' + out[6] + ')\n')
		else:
			return ('		if state == ' + out[0] + ':\n' +
					'			state = (' + out[4] + ' if ' + "x['" +
					out[1] + "']<" + out[2] + missing_value_handling +
					' else ' + out[6] + ')\n')
	else:
		out = re.findall(r"[\d.-]+", s)
		return (re.findall(r"[\t]+", s)[0].replace('\t', '	') +
				'		if state == ' + out[0] + ':\n	' +
				re.findall(r"[\t]+", s)[0].replace('\t', '	') +
				'		return ' + out[1] + '\n')


def booster_parser(tree, i):
	if i == 0:
		return ('	if num_booster == 0:\n		state = 0\n' + "".join([
			string_parser(tree.split('\n')[i])
			for i in range(len(tree.split('\n')) - 1)
		]))
	else:
		return ('	elif num_booster == ' + str(i) +
				':\n		state = 0\n' + "".join([
					string_parser(tree.split('\n')[i])
					for i in range(len(tree.split('\n')) - 1)
				]))


def model_to_py(base_score, model, out_file):
	trees = model.get_dump()
	result = ["import numpy as np\n\n" + "def xgb_tree(x, num_booster):\n"]
	for i in range(len(trees)):
		result.append(booster_parser(trees[i], i))
	with open(out_file, 'a') as the_file:
		the_file.write("".join(result) +
					   "\ndef xgb_predict(x):\n	predict = " +
					   str(base_score) + "\n" +
					   "# initialize prediction with base score\n" +
					   "	for i in range(" + str(len(trees)) +
					   "):\n		predict = predict + xgb_tree(x, i)" +
					   "\n	return predict")


def trees_to_dataframe(clf):
	tree_ids = []
	node_ids = []
	fids = []
	splits = []
	y_directs = []
	n_directs = []
	missings = []
	gains = []
	covers = []
	decision_paths = dict()
	scores = dict()
	trees = clf.get_dump(with_stats=True)
	for i, tree in enumerate(trees):
		decision_paths.update({str(i) + '-0': None})
		for line in tree.split('\n'):
			arr = line.split('[')
			# Leaf node
			if len(arr) == 1:
				# Last element of line.split is an empy string
				if arr == ['']:
					continue
				# parse string
				parse = arr[0].split(':')
				stats = re.split('=|,', parse[1])
				# append to lists
				tree_ids.append(i)
				node_ids.append(int(re.findall(r'\b\d+\b', parse[0])[0]))
				fids.append('Leaf')
				splits.append(float('NAN'))
				y_directs.append(float('NAN'))
				n_directs.append(float('NAN'))
				missings.append(float('NAN'))
				gains.append(float(stats[1]))
				covers.append(float(stats[3]))
				# update dicts
				parent_key = str(i) + '-' + re.findall(r'\b\d+\b', arr[0])[0]
				scores.update({parent_key: float(stats[1])})
			# Not a Leaf Node
			else:
				# parse string
				fid = arr[1].split(']')
				parse = fid[0].split('<')
				stats = re.split('=|,', fid[1])
				# append to lists
				tree_ids.append(i)
				node_ids.append(int(re.findall(r'\b\d+\b', arr[0])[0]))
				fids.append(parse[0])
				splits.append(float(parse[1]))
				str_i = str(i)
				y_directs.append(str_i + '-' + stats[1])
				n_directs.append(str_i + '-' + stats[3])
				missings.append(str_i + '-' + stats[5])
				gains.append(float(stats[7]))
				covers.append(float(stats[9]))
				# update dicts
				parent_key = str(i) + '-' + re.findall(r'\b\d+\b', arr[0])[0]
				scores.update({parent_key: float(stats[7])})
				left_direct_key = str_i + '-' + stats[1]
				right_direct_key = str_i + '-' + stats[3]
				left_direct_value = parse[0] + '<' + parse[1]
				right_direct_value = parse[0] + '>=' + parse[1]
				missing_value = parse[0] + ' is missing'
				if (str_i + '-' + stats[5]) == left_direct_key:
					left_direct_value = '(' + left_direct_value + \
						' or ' + missing_value + ')'
					right_direct_value = '(' + right_direct_value + ')'
				else:
					left_direct_value = '(' + left_direct_value + ')'
					right_direct_value = '(' + right_direct_value + \
						' or ' + missing_value + ')'
				if decision_paths.get(parent_key) is not None:
					left_direct_value = '(' + decision_paths.get(parent_key) + \
						' and ' + left_direct_value + ')'
					right_direct_value = '(' + decision_paths.get(parent_key) + \
						' and ' + right_direct_value + ')'
				decision_paths.update({left_direct_key: left_direct_value})
				decision_paths.update({right_direct_key: right_direct_value})
	ids = [
		str(t_id) + '-' + str(n_id) for t_id, n_id in zip(tree_ids, node_ids)
	]
	import pandas as pd
	df = pd.DataFrame({
		'Tree': tree_ids,
		'Node': node_ids,
		'ID': ids,
		'Feature': fids,
		'Split': splits,
		'Yes': y_directs,
		'No': n_directs,
		'Missing': missings,
		'Gain': gains,
		'Cover': covers
	})
	df = df[[
		'Tree', 'Node', 'ID', 'Feature', 'Split', 'Yes', 'No', 'Missing',
		'Gain', 'Cover'
	]]
	return df, decision_paths, scores


if __name__ == '__main__':
	from IPython.core.interactiveshell import InteractiveShell
	InteractiveShell.ast_node_interactivity = 'all'
	import warnings
	warnings.filterwarnings('ignore')
	from sklearn.datasets import make_classification
	x, y = make_classification(n_samples=1000,
							   n_features=45,
							   n_informative=12,
							   n_redundant=7,
							   random_state=1)
	feature_list = [f'feature_{i}' for i in range(0, 45)]
	y_name = 'label'
	x = pd.DataFrame(data=x, columns=feature_list)
	y = pd.Series(data=y, name=y_name)
	from sklearn.tree import DecisionTreeClassifier
	params = {
		'criterion': 'gini',
		'max_depth': 3,
		'min_samples_leaf': 0.01,
		'splitter': 'best',
		'random_state': 1
	}
	clf = DecisionTreeClassifier(**params)
	clf.fit(x, y)
	decision_path, decision_depth = decision_paths(clf=clf,
												   feature_list=feature_list,
												   is_print=False)
	leaf_batch = leaf_batches(clf=clf, x=x, y=y)
	leaf_batch['decision_path'] = leaf_batch['leaf_index'].map(
		decision_path.get)
	leaf_batch.loc[[0, 1, len(leaf_batch) - 1]]
	df = pd.concat([x, y], axis=1)
	step = 300
	df['index'] = index_generator(len(df), step)
	index_list = ['index']
	rule_list = [
		'feature_37<=3.15 and feature_15>-1.17 and feature_39<=-1.13 and feature_9>-2.36',
		'feature_37<=3.15 and feature_15<=-1.17 and feature_21>2.51'
	]
	simulaton = rule_simulator(df=df,
							   y=y_name,
							   index=index_list,
							   rule=rule_list,
							   splitor_sup=False)
	from sklearn.tree import export_graphviz
	import pydotplus
	from IPython.display import display, Image
	kwargs = {
		'feature_names': feature_list,
		'class_names': ['0', '1'],
		'filled': True,
		'rounded': True,
		'special_characters': True
	}
	with open('tree.dot', 'w') as f:
		f = export_graphviz(clf, out_file=f)
	with open('tree.dot', 'r') as f:
		dot_data = f.read()
	export_graphviz(clf, out_file='tree.dot', **kwargs)
	dot_data = export_graphviz(clf, out_file=None, **kwargs)
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_png('tree.png')
	img = Image(graph.create_png())
	display(img)
	params = {
		'booster': 'gbtree',
		'nthread': -1,
		'num_boost_round': 30,
		'eta': 0.1,
		'gamma': 3,
		'max_depth': 3,
		'min_child_weight': 2,
		'alpha': 0,
		'lambda': 0.1,
		'tree_method': 'exact',
		'objective': 'binary:logistic',
		'base_score': np.mean(y),
		'eval_metric': ['auc'],
		'seed': 1
	}
	num_boost_round = params.get('num_boost_round')
	early_stopping_rounds = 10
	import xgboost as xgb
	params_constrained = params.copy()
	params_constrained['monotone_constraints'] = {'feature_0': 1, 'feature_2': -1}
	params_constrained['interaction_constraints'] = '[[0, 2], [1, 3, 4], [5, 6]]'
	clf = xgb.train(params=params_constrained,
					num_boost_round=num_boost_round,
					dtrain=xgb.DMatrix(data=x, label=y, weight=None, feature_weights=None),
					evals=[(xgb.DMatrix(data=x, label=y), y_name)],
					early_stopping_rounds=early_stopping_rounds)
	params = {
		'booster': 'gbtree',
		'n_jobs': -1,
		'n_estimators': 30,
		'learning_rate': 0.1,
		'gamma': 3,
		'max_depth': 3,
		'min_child_weight': 2,
		'reg_alpha': 0,
		'reg_lambda': 0.1,
		'tree_method': 'exact',
		'objective': 'binary:logistic',
		'base_score': np.mean(y),
		'eval_metric': ['auc'],
		'random_state': 1
	}
	from xgboost.sklearn import XGBClassifier
	clf = XGBClassifier(**params)
	clf.fit(X=x,
			y=y,
			eval_set=[(x, y)],
			early_stopping_rounds=early_stopping_rounds)
	clf = clf.get_booster()
	clf.get_dump()
	clf.trees_to_dataframe()
	# import os
	# os.environ['PATH'] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'
	xgb.to_graphviz(booster=clf, num_trees=0)
	model_to_py(base_score=params['base_score'], model=clf, out_file='clf.py')
	import clf
	clf.xgb_predict(x.loc[0])
	df, decision_path, scores = trees_to_dataframe(clf)
	predictor = pd.DataFrame(
		data=clf.predict(data=xgb.DMatrix(data=x), pred_leaf=True),
		columns=[f'num_trees_{i}' for i in range(0, num_boost_round)])
	for i in range(0, num_boost_round):
		predictor[f'num_trees_{i}'] = predictor[f'num_trees_{i}'].map(
			lambda x: str(i) + '-' + str(x))
		predictor[f'decision_paths_{i}'] = predictor[f'num_trees_{i}'].map(
			decision_path)
		predictor[f'scores_{i}'] = predictor[f'num_trees_{i}'].map(scores)
	predictor['linear_predictor'] = predictor[[
		f'scores_{i}' for i in range(0, num_boost_round)
	]].sum(axis=1)
	predictor['predict_result_repeat'] = predictor['linear_predictor'].map(
		lambda x: 1 / (1 + np.exp(-x)))
	predictor['predict_result'] = clf.predict(data=xgb.DMatrix(data=x))
