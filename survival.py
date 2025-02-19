import numpy as np
import pandas as pd
import xgboost as xgb


def convert_to_structured(T, E):
	"""
	Converts data in time (T) and event (E) format to a structured numpy array.
	Provides common interface to other libraries such as sksurv and sklearn.
	Args:
		T (np.array): Array of times
		E (np.array): Array of events
	Returns:
		np.array: Structured array containing the boolean event indicator
			as first field, and time of event or time of censoring as second field
	"""
	# dtypes for conversion
	default_dtypes = {"names": ("c1", "c2"), "formats": ("bool", "f8")}
	# concat of events and times
	concat = list(zip(E.values, T.values))
	# return structured array
	return np.array(concat, dtype=default_dtypes)


def convert_y(y):
	"""
	Convert structured array y into an array of
	event indicators (E) and time of events (T).
	Args:
		y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
			and time of event or time of censoring as second field.
	Returns:
		T ([np.array, pd.Series]): Time of events
		E ([np.array, pd.Series]): Binary event indicator
	"""
	event_field, time_field = y.dtype.names
	return y[event_field], y[time_field]


def convert_data_to_xgb_format(X, y, objective):
	"""Convert (X, y) data format to xgb.DMatrix format, either using cox or aft models.
	Args:
		X ([pd.DataFrame, np.array]): features to be used while fitting
			XGBoost model
		y (structured array(numpy.bool_, numpy.number)): binary event indicator as first field,
			and time of event or time of censoring as second field.
		objective (string): one of 'survival:aft' or 'survival:cox'
	Returns:
		xgb.DMatrix: data to train xgb
	"""
	E, T = convert_y(y)
	# converting data to xgb format
	if objective == "survival:aft":
		d_matrix = build_xgb_aft_dmatrix(X, T, E)
	elif objective == "survival:cox":
		d_matrix = build_xgb_cox_dmatrix(X, T, E)
	else:
		raise ValueError("Objective not supported. Use survival:cox or survival:aft")
	return d_matrix


# Building XGB Design matrices - AFT and Cox Model
def build_xgb_aft_dmatrix(X, T, E):
	"""Builds a XGB DMatrix using specified Data Frame of features (X)
	 arrays of times (T) and censors/events (E).
	Args:
		X ([pd.DataFrame, np.array]): Data Frame to be converted to
			XGBDMatrix format.
		T ([np.array, pd.Series]): Array of times.
		E ([np.array, pd.Series]): Array of censors(False) / events(True).
	Returns:
		xgb.DMatrix: A XGB DMatrix is returned including features and target.
	"""
	d_matrix = xgb.DMatrix(X)
	y_lower_bound = T
	y_upper_bound = np.where(E, T, np.inf)
	d_matrix.set_float_info("label_lower_bound", y_lower_bound.copy())
	d_matrix.set_float_info("label_upper_bound", y_upper_bound.copy())
	return d_matrix


def build_xgb_cox_dmatrix(X, T, E):
	"""Builds a XGB DMatrix using specified Data Frame of features (X)
		arrays of times (T) and censors/events (E).
	Args:
		X ([pd.DataFrame, np.array]): Data Frame to be converted to XGBDMatrix format.
		T ([np.array, pd.Series]): Array of times.
		E ([np.array, pd.Series]): Array of censors(False) / events(True).
	Returns:
		(DMatrix): A XGB DMatrix is returned including features and target.
	"""
	target = np.where(E, T, -T)
	return xgb.DMatrix(X, label=target)


def hazard_to_survival(interval):
	"""Convert hazards (interval probabilities of event) into survival curve
	Args:
		interval ([pd.DataFrame, np.array]): hazards (interval probabilities of event)
		usually result of predict or  result from _get_point_probs_from_survival
	Returns:
		[pd.DataFrame, np.array]: survival curve
	"""
	return (1 - interval).cumprod(axis=1)


# epsilon to prevent division by zero
EPS = 1e-6


def get_time_bins(T, E, size=12):
	"""
	Method to automatically define time bins
	"""
	lower_bound = max(T[E == 0].min(), T[E == 1].min()) + 1
	upper_bound = min(T[E == 0].max(), T[E == 1].max()) - 1
	return np.linspace(lower_bound, upper_bound, size, dtype=int)


def sort_and_add_zeros(x, ind):
	"""
	Sorts an specified array x according to a reference index ind
	Args:
		x (np.array): Array to be sorted according to index specified in ind
		ind (np.array): Index to be used as reference to sort array x
	Returns:
		np.array: Array x sorted according to ind indexes
	"""
	x = np.take_along_axis(x, ind, axis=1)
	# Check with concatenate
	x = np.c_[np.zeros(x.shape[0]), x]
	return x


# Utils to calculate survival intervals
def calculate_exp_boundary(survival_func, V, z):
	"""
	Creates confidence intervals using the Exponential Greenwood formula.
	Available at: https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
	Args:
		survival_func ([np.array, pd.Series]): Survival function estimates
		V ([np.array, pd.Series]): Exponential Greenwood variance component
		z (Int): Normal quantile to be used - depends on the confidence level
	Returns:
		pd.DataFrame: Confidence intervals
	"""
	C = np.log(-np.log(survival_func)) + z * np.sqrt(V)
	C_exp = np.exp(-np.exp(C))
	return pd.DataFrame(C_exp).fillna(method="bfill").fillna(method="ffill").values


def sample_time_bins(surv_array, T_neighs, time_bins):
	"""
	Sample survival curve at specified points in time to get a survival df
	Args:
		surv_array (np.array): Survival array to be sampled from
		T_neighs (np.array): Array of observed times
		time_bins (List): Specified time bins to retrieve survival estimates
	Returns:
		pd.DataFrame: DataFrame with survival for each specified time
		bin
	"""
	surv_df = []
	for t in time_bins:
		survival_at_t = (surv_array + (T_neighs > t)).min(axis=1)
		surv_df.append(survival_at_t)
	surv_df = pd.DataFrame(surv_df, index=time_bins).T
	return surv_df


def sort_times_and_events(T, E):
	"""
	Collects and sorts times and events from neighbors set
	Args:
		T (np.array): matrix of times (will compute one kaplan meier for each row)
		E (np.array): matrix of events (will compute one kaplan meier for each row)
	Returns:
		(np.array, np.array): matrix of times, sorted by most recent, matrix of events. sorted by most recent
	"""
	# getting sorted indices for times along each neighbor-set
	argsort_ind = np.argsort(T, axis=1)
	# reordering times and events according to sorting and adding t=0
	T_sorted = sort_and_add_zeros(T, argsort_ind)
	E_sorted = sort_and_add_zeros(E, argsort_ind)
	return T_sorted, E_sorted


def calculate_survival_func(E_sorted):
	"""
	Calculates the survival function for a given set of neighbors
	Args:
		E_sorted (np.array): A time-sorted array (row-wise) of events/censors
	Returns:
		np.array: The survival function evaluated for each row
	"""
	# max number of elements in leafs
	# TODO: allow numpy work with variable size arrays
	n_samples = E_sorted.shape[1] - 1
	# number of elements at risk
	at_risk = np.r_[n_samples, np.arange(n_samples, 0, -1)]
	# product argument for surivial
	survival_prod_arg = 1 - (E_sorted / at_risk)
	return np.cumprod(survival_prod_arg, axis=1)


def calculate_confidence_intervals(E_sorted, survival_func, z):
	"""
	Calculates confidence intervals based on the Exponential Greenwood
	formula. Available at: https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
	Args:
		E_sorted (np.array): A time-sorted array (row-wise) of events/censors
		survival_func (np.array): Survival function array to be used as a
		baseline to calculate confidence interval
	Returns:
		(np.array, np.array): Upper confidence interval (95%), Lower condidence interval (95%)
	"""
	# guarantee that survival_func
	# is strictly positive and not exactly equal to 1
	# for numerical purposess
	survival_func = np.clip(survival_func, EPS, 1 - EPS)
	# max number of elements in leafs
	# TODO: allow numpy work with variable size arrays
	n_samples = E_sorted.shape[1] - 1
	# number of elements at risk
	at_risk = np.r_[n_samples, np.arange(n_samples, 0, -1)]
	# also calculating confidence intervals
	numerator = E_sorted.astype(float)
	denominator = at_risk * (at_risk - E_sorted)
	ci_prod_arg = numerator / np.clip(denominator, EPS, None)
	# exponential greenwood variance component
	numerator = np.cumsum(ci_prod_arg, axis=1)
	denominator = np.power(np.log(survival_func), 2)
	V = numerator / np.clip(denominator, EPS, None)
	# calculating upper and lower confidence intervals (one standard deviation)
	upper_ci = calculate_exp_boundary(survival_func, V, z)
	lower_ci = calculate_exp_boundary(survival_func, V, -z)
	return upper_ci, lower_ci


def calculate_kaplan_vectorized(T, E, time_bins, z=1.0):
	"""
	Predicts a Kaplan Meier estimator in a vectorized manner, including
	its upper and lower confidence intervals based on the Exponential
	Greenwod formula. See _calculate_survival_func for theoretical reference
	Args:
		T (np.array): matrix of times (will compute one kaplan meier for each row)
		E (np.array): matrix of events (will compute one kaplan meier for each row)
		time_bins (List): Specified time bins to retrieve survival estimates
	Returns:
		(np.array, np.array, np.array): survival values at specified time bins,
			upper confidence interval, lower confidence interval
	"""
	# sorting times and events
	T_sorted, E_sorted = sort_times_and_events(T, E)
	E_sorted = E_sorted.astype(int)
	# calculating survival functions
	survival_func = calculate_survival_func(E_sorted)
	# calculating confidence intervals
	upper_ci, lower_ci = calculate_confidence_intervals(E_sorted, survival_func, z)
	# only returning time bins asked by user
	survival_func = sample_time_bins(survival_func, T_sorted, time_bins)
	upper_ci = sample_time_bins(upper_ci, T_sorted, time_bins)
	lower_ci = sample_time_bins(lower_ci, T_sorted, time_bins)
	return survival_func, upper_ci, lower_ci


def _get_conditional_probs_from_survival(surv):
	"""
	Return conditional failure probabilities (for each time interval) from survival curve.
	P(T < t+1 | T > t): probability of failure up to time t+1 conditional on individual
	survival up to time t.
	Args:
		surv (pd.DataFrame): dataframe of survival estimates, as .predict() methods return
	Returns:
		pd.DataFrame: conditional failurer probability of event
			specifically at time bucket
	"""
	conditional_preds = 1 - (surv / surv.shift(1, axis=1).fillna(1))
	conditional_preds = conditional_preds.fillna(0)
	return conditional_preds


def _get_point_probs_from_survival(conditional_preds):
	"""
	Transform conditional failure probabilities into point probabilities
	(at each interval) from survival curve.
	P(t < T < t+1): point probability of failure between time t and t+1.
	Args:
		conditional_preds (pd.DataFrame): dataframe of conditional failure probability -
		output of _get_conditional_probs_from_survival function
	Returns:
		pd.DataFrame: probability of event at all specified time buckets
	"""
	sample = conditional_preds.reset_index(drop=True)
	# list of event probabilities summing up to 1
	event = []
	# event in time interval 0
	v_0 = 1 * sample[0]
	event.append(v_0)
	# Looping over other time intervals
	for i in range(1, len(sample)):
		v_i = (1 - sum(event)) * sample[i]
		event.append(v_i)
	return pd.Series(event)


def calculate_interval_failures(surv):
	"""
	Return point probabilities (at each interval) from survival curve.
	P(t < T < t+1): point probability of failure between time t and t+1.
	Args:
		surv (pd.DataFrame): dataframe of (1 - cumulative survival estimates),
		complementary of .predict() methods return
	Returns:
		pd.DataFrame: probability of event at all specified time buckets
	"""
	interval_preds = _get_conditional_probs_from_survival(surv).apply(
		_get_point_probs_from_survival, axis=1
	)
	interval_preds.columns = surv.columns
	return interval_preds


def concordance_index(y_true, survival, risk_strategy="mean", which_window=None):
	"""
	Compute the C-index for a structured array of ground truth times and events
	and a predicted survival curve using different strategies for estimating risk from it.
	!!! Note
		* Computation of the C-index is $\\mathcal{O}(n^2)$.
	Args:
		y_true (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
			and time of event or time of censoring as second field.
		survival ([pd.DataFrame, np.array]): A dataframe of survival probabilities
			for all times (columns), from a time_bins array, for all samples of X (rows).
			If risk_strategy is 'precomputed', is an array with representing risks for each sample.
		risk_strategy (string):
			Strategy to compute risks from the survival curve. For a given sample:
			* `mean` averages probabilities across all times
			* `window`: lets user choose on of the time windows available (by which_window argument)
				and uses probabilities of this specific window
			* `midpoint`: selects the most central window of index int(survival.columns.shape[0]/2)
				and uses probabilities of this specific window
			* `precomputed`: assumes user has already calculated risk.
				The survival argument is assumed to contain an array of risks instead
		which_window (object): Which window to use when risk_strategy is 'window'. Should be one
			of the columns of the dataframe. Will raise ValueError if column is not present
	Returns:
		Float: Concordance index for y_true and survival
	"""
	# choosing risk calculation strategy
	if risk_strategy == "mean":
		risks = 1 - survival.mean(axis=1)
	elif risk_strategy == "window":
		if which_window is None:
			raise ValueError("Need to set which window to use via the which_window parameter")
		risks = 1 - survival[which_window]
	elif risk_strategy == "midpoint":
		midpoint = int(survival.columns.shape[0] / 2)
		midpoint_col = survival.columns[midpoint]
		risks = 1 - survival[midpoint_col]
	elif risk_strategy == "precomputed":
		risks = survival
	else:
		raise ValueError(f"Chosen risk computing strategy of {risk_strategy} is not available.")
	# organizing event, time and risk data
	events, times = convert_y(y_true)
	events = events.astype(bool)
	cind_df = pd.DataFrame({"t": times, "e": events, "r": risks})
	count_pairs = 0
	concordant_pairs = 0
	tied_pairs = 0
	# running loop for each uncensored sample,
	# as by https://arxiv.org/pdf/1811.11347.pdf
	for _, row in cind_df.query("e == True").iterrows():
		# getting all censored and uncensored samples
		# after current row
		samples_after_i = cind_df.query(f"""{row['t']} < t""")
		# counting total, concordant and tied pairs
		count_pairs += samples_after_i.shape[0]
		concordant_pairs += (samples_after_i["r"] < row["r"]).sum()
		tied_pairs += (samples_after_i["r"] == row["r"]).sum()
	return (concordant_pairs + tied_pairs / 2) / count_pairs


def _match_times_to_windows(times, windows):
	"""
	Match a list of event or censoring times to the corresponding
	time window on the survival dataframe.
	"""
	from bisect import bisect_right
	matches = np.array([bisect_right(windows, e) for e in times])
	matches = np.clip(matches, 0, len(windows) - 1)
	return windows[matches]


def approx_brier_score(y_true, survival, aggregate="mean"):
	"""
	Estimate brier score for all survival time windows. Aggregate scores for an approximate
	integrated brier score estimate.
	Args:
		y_true (structured array(numpy.bool_, numpy.number)): B inary event indicator as first field,
			and time of event or time of censoring as second field.
		survival ([pd.DataFrame, np.array]): A dataframe of survival probabilities
			for all times (columns), from a time_bins array, for all samples of X (rows).
			If risk_strategy is 'precomputed', is an array with representing risks for each sample.
		aggregate ([string, None]): How to aggregate brier scores from different time windows:
			* `mean` takes simple average
			* `None` returns full list of brier scores for each time window
	Returns:
		[Float, np.array]:
			single value if aggregate is 'mean'
			np.array if aggregate is None
	"""
	events, times = convert_y(y_true)
	events = events.astype(bool)
	# calculating censoring distribution
	censoring_dist, _, _ = calculate_kaplan_vectorized(
		times.reshape(1, -1), ~events.reshape(1, -1), survival.columns
	)
	# initializing scoring df
	scoring_df = pd.DataFrame({"e": events, "t": times}, index=survival.index)
	# adding censoring distribution survival at event
	event_time_windows = _match_times_to_windows(times, survival.columns)
	scoring_df["cens_at_event"] = censoring_dist[event_time_windows].iloc[0].values
	# list of window results
	window_results = []
	# loop for all suvival time windows
	for window in survival.columns:
		# adding window info to scoring df
		scoring_df = scoring_df.assign(surv_at_window=survival[window]).assign(
			cens_at_window=censoring_dist[window].values[0]
		)
		# calculating censored brier score first term
		# as by formula on B4.3 of https://arxiv.org/pdf/1811.11347.pdf
		first_term = (
			(scoring_df["t"] <= window).astype(int)
			* (scoring_df["e"])
			* (scoring_df["surv_at_window"]) ** 2
			/ (scoring_df["cens_at_event"])
		)
		# calculating censored brier score second term
		# as by formula on B4.3 of https://arxiv.org/pdf/1811.11347.pdf
		second_term = (
			(scoring_df["t"] > window).astype(int)
			* (1 - scoring_df["surv_at_window"]) ** 2
			/ (scoring_df["cens_at_window"])
		)
		# adding and taking average
		result = (first_term + second_term).sum() / scoring_df.shape[0]
		window_results.append(result)
	if aggregate == "mean":
		return np.array(window_results).mean()
	elif aggregate is None:
		return np.array(window_results)
	else:
		raise ValueError(f"Chosen aggregating strategy of {aggregate} is not available.")


def dist_calibration_score(y_true, survival, n_bins=10, returns="pval"):
	"""
	Estimate D-Calibration for the survival predictions.
	Args:
		y_true (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
			and time of event or time of censoring as second field.
		survival ([pd.DataFrame, np.array]): A dataframe of survival probabilities
			for all times (columns), from a time_bins array, for all samples of X (rows).
			If risk_strategy is 'precomputed', is an array with representing risks for each sample.
		n_bins (Int): Number of bins to equally divide the [0, 1] interval
		returns (string):
			What information to return from the function:
			* `statistic` returns the chi squared test statistic
			* `pval` returns the chi squared test p value
			* `max_deviation` returns the maximum percentage deviation from the expected value,
			calculated as `abs(expected_percentage - real_percentage)`,
			where `expected_percentage = 1.0/n_bins`
			* `histogram` returns the full calibration histogram per bin
			* `all` returns all of the above in a dictionary
	Returns:
		[Float, np.array, Dict]:
		* Single value if returns is in `['statistic','pval','max_deviation']``
		* np.array if returns is 'histogram'
		* dict if returns is 'all'
	"""
	from scipy.stats import chisquare
	# calculating bins
	bins = np.round(np.linspace(0, 1, n_bins + 1), 2)
	events, times = convert_y(y_true)
	events = events.astype(bool)
	# mapping event and censoring times to survival windows
	event_time_windows = _match_times_to_windows(times, survival.columns)
	survival_at_ti = np.array(
		[survival.iloc[i][event_time_windows[i]] for i in range(len(survival))]
	)
	survival_at_ti = np.clip(survival_at_ti, EPS, None)
	# creating data frame to calculate uncensored and censored counts
	scoring_df = pd.DataFrame(
		{
			"survival_at_ti": survival_at_ti,
			"t": times,
			"e": events,
			"bin": pd.cut(survival_at_ti, bins, include_lowest=True),
			"cens_spill_term": 1 / (n_bins * survival_at_ti),
		}
	)
	# computing uncensored counts:
	# sum the number of events per bin
	count_uncens = scoring_df.query("e == True").groupby("bin").size()
	# computing censored counts at bin of censoring
	# formula (A) as by page 49 of
	# https://arxiv.org/pdf/1811.11347.pdf
	count_cens = (
		scoring_df.query("e == False")
		.groupby("bin")
		.apply(lambda x: (1 - np.clip(x.name.left, 0, 1) / x["survival_at_ti"]).sum())
	)
	# computing censored counts at bins after censoring
	# effect of 'blurring'
	# formula (B) as by page 49 of
	# https://arxiv.org/pdf/1811.11347.pdf
	count_cens_spill = (
		scoring_df.query("e == False")
		.groupby("bin")["cens_spill_term"]
		.sum()
		.iloc[::-1]
		.shift()
		.fillna(0)
		.cumsum()
		.iloc[::-1]
	)
	final_bin_counts = count_uncens + count_cens + count_cens_spill
	if returns == "statistic":
		result = chisquare(final_bin_counts)
		return result.statistic
	elif returns == "pval":
		result = chisquare(final_bin_counts)
		return result.pvalue
	elif returns == "max_deviation":
		proportions = final_bin_counts / final_bin_counts.sum()
		return np.abs(proportions - 1 / n_bins).max()
	elif returns == "histogram":
		return final_bin_counts
	elif returns == "all":
		result = chisquare(final_bin_counts)
		proportions = final_bin_counts / final_bin_counts.sum()
		max_deviation = np.abs(proportions - 1 / n_bins).max()
		return {
			"statistic": result.statistic,
			"pval": result.pvalue,
			"max_deviation": max_deviation,
			"histogram": final_bin_counts,
		}
	else:
		raise ValueError(f"Chosen return of {returns} is not available.")




from pycox.datasets import metabric
df = metabric.read_df()
X = df.drop(['duration', 'event'], axis=1)
y = convert_to_structured(df['duration'], df['event'])
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, random_state=42)
PARAMS_XGB_COX = {
	'objective': 'survival:cox',
	'tree_method': 'hist', 
	'learning_rate': 5e-2, 
	'max_depth': 8, 
	'booster':'dart',
	'subsample':0.5,
	'min_child_weight': 50, 
	'colsample_bynode':0.5
}
PARAMS_XGB_AFT = {
	'objective': 'survival:aft',
	'eval_metric': 'aft-nloglik',
	'aft_loss_distribution': 'normal',
	'aft_loss_distribution_scale': 1.0,
	'tree_method': 'hist', 
	'learning_rate': 5e-2, 
	'max_depth': 8, 
	'booster':'dart',
	'subsample':0.5,
	'min_child_weight': 50,
	'colsample_bynode':0.5
}
PARAMS_TREE = {
	'objective': 'survival:cox',
	'eval_metric': 'cox-nloglik',
	'tree_method': 'hist', 
	'max_depth': 100, 
	'booster':'dart', 
	'subsample': 1.0,
	'min_child_weight': 50, 
	'colsample_bynode': 1.0
}
PARAMS_LR = {
	'C': 1e-3,
	'max_iter': 500
}
N_NEIGHBORS = 50
TIME_BINS = np.arange(15, 315, 15)


dtrain = convert_data_to_xgb_format(X_train, y_train, 'survival:cox')
dval = convert_data_to_xgb_format(X_valid, y_valid, 'survival:cox')
bst = xgb.train(
	PARAMS_XGB_COX,
	dtrain,
	num_boost_round=1000,
	early_stopping_rounds=10,
	evals=[(dval, 'val')],
	verbose_eval=0
)
preds = bst.predict(dval)
cind = concordance_index(y_valid, preds, risk_strategy='precomputed')
print(f"C-index: {cind:.3f}")
print(f"Model predictions: {preds[:5]}")


dtrain = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
dval = convert_data_to_xgb_format(X_valid, y_valid, 'survival:aft')
bst = xgb.train(
	PARAMS_XGB_AFT,
	dtrain,
	num_boost_round=1000,
	early_stopping_rounds=10,
	evals=[(dval, 'val')],
	verbose_eval=0
)
preds = bst.predict(dval)
cind = concordance_index(y_valid, -preds, risk_strategy='precomputed')
print(f"C-index: {cind:.3f}")
print(f"Average survival time: {preds.mean():.0f} days")
