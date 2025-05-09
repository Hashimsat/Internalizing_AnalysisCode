import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage import label


def permutation_testing(data, num_perm, alpha_cluster, alpha, test_type, condition=None):
    # data = subj * time * cond matrix

    # set a random seed for reproducibility
    np.random.seed(42)

    # isolate clusters in observed data and calculate associated test statistics
    if test_type == 0:
        if condition is None:
            # perform one-sample t-test without conditions
            t_values, p_values = stats.ttest_1samp(data, 0)
        else:
            # perform one-sample t-test for one condition
            t_values, p_values = stats.ttest_1samp(data[:, :, condition], 0)
    else:
        # perform paired-sample t-test
        t_values, p_values = stats.ttest_rel(data[:, :, 0], data[:, :, 1])

    # determine significant t-values
    sig_t_values = np.abs(t_values) > stats.t.ppf(1.0 - alpha_cluster / 2., data.shape[0] - 1)
    sig_t_values_uncor = sig_t_values.astype(float)
    sig_t_values_uncor[sig_t_values_uncor == 0] = np.nan

    # find significant clusters if any, label them and calculate t values
    if np.any(sig_t_values):
        cluster_labels, num_clust = label(sig_t_values)
        cluster_t_values = np.abs([np.sum(t_values[cluster_labels == c]) for c in range(1, num_clust + 1)])
    else:
        cluster_labels = np.ones_like(sig_t_values)
        num_clust = 1
        cluster_t_values = np.array([np.max(np.abs(t_values))])

    # run successive permutations
    perm_t_values_max = np.zeros(num_perm)
    for p in range(num_perm):
        # shuffle condition labels
        if test_type == 0:
            # shuffle data and multiply with random -1 or 1 for one-sample t-tests
            signed_arr = np.random.choice([-1, 1], size=data.shape)
            perm_data = signed_arr * data
        else:
            # shuffle condition labels for paired-sample t-tests
            perm_data = np.empty_like(data)
            cond_order = np.random.choice([0, 1], size=data.shape[0])
            for sub in range(data.shape[0]):
                for cond in range(data.shape[2]):
                    perm_data[sub, :, cond] = data[sub, :, cond_order[sub] if cond == 0 else 1 - cond_order[sub]]

        # isolate clusters in permuted data and store peak test statistic
        if test_type == 0:
            # perform one-sample t-test on permuted data
            perm_t_values, _ = stats.ttest_1samp(perm_data, 0)
        else:
            # perform paired-sample t-test on permuted data
            perm_t_values, _ = stats.ttest_rel(perm_data[:, :, 0], perm_data[:, :, 1])
        sig_perm_t_values = np.abs(perm_t_values) > stats.t.ppf(1.0 - alpha_cluster / 2., data.shape[0] - 1)

        if not np.any(sig_perm_t_values):
            perm_t_values_max[p] = np.max(np.abs(perm_t_values))
        else:
            perm_cluster_labels, num_perm_clusters = label(sig_perm_t_values)
            perm_cluster_t_values = np.abs([np.sum(perm_t_values[perm_cluster_labels == c])
                                            for c in range(1, num_perm_clusters + 1)])
            perm_t_values_max[p] = np.max(perm_cluster_t_values)

    # calculate corrected p-value for each observed cluster and form vector of significant clusters
    sig_t_values_cor = sig_t_values_uncor.copy()
    p_values_cor = np.ones(num_clust)
    for c in range(num_clust):
        p_values_cor[c] = 1 - (len(np.where(perm_t_values_max <= cluster_t_values[c])[0]) / num_perm)
        if p_values_cor[c] > alpha:
            sig_t_values_cor[cluster_labels == c + 1] = np.nan

    return sig_t_values_cor, p_values_cor


def perform_permutation_analysis(data, cond_var, test_type, condition=None):
    time_array = np.arange(-2, 10, 0.1)
    # sig_t_values_cor, p_values_cor = permutation_testing(data, data.shape[1], 0.05, 0.05, test_type, condition)
    sig_t_values_cor, p_values_cor = permutation_testing(data, 1000, 0.05, 0.05, test_type, condition)
    sig_t_values_cor = pd.DataFrame(sig_t_values_cor).assign(time=time_array)
    print(cond_var)
    print(p_values_cor)
    significant_ranges = extract_significant_ranges(sig_t_values_cor)
    return significant_ranges, p_values_cor


def extract_significant_ranges(sig_t_values_cor):
    significant_ranges = []
    start_time = None

    for i in range(len(sig_t_values_cor)):
        current_time = sig_t_values_cor['time'].iloc[i]
        current_value = sig_t_values_cor.iloc[i, 0]

        if current_time <= -0.3:
            continue

        # find start of a new significant range
        if current_value == 1.0 and start_time is None:
            start_time = current_time
        # find end of a significant range
        elif np.isnan(current_value) and start_time is not None:
            end_time = sig_t_values_cor['time'].iloc[i - 1]
            significant_ranges.append((float(round(start_time, 1)), float(round(end_time, 1))))
            start_time = None

    if start_time is not None:
        end_time = sig_t_values_cor['time'].iloc[-1]
        significant_ranges.append((float(round(start_time, 1)), float(round(end_time, 1))))

    return significant_ranges


def analyse_scr_data(df, cond_var, conditions, is_regression=False):
    if is_regression:
        pivot_table = df.pivot(index=['sub_id', 'bin'], columns='coef_name', values='beta_value').reset_index()
        pivot_table.set_index(['sub_id', 'bin'], inplace=True)
        data = pivot_table[cond_var].unstack(level='bin').values.reshape(-1, 120, 1)
    else:
        df_grouped = df.groupby(['sub_id', 'times', cond_var])['data'].mean().reset_index()
        pivot_table = pd.pivot_table(df_grouped, values='data', index=['sub_id', 'times'], columns=[cond_var])[conditions]
        df_list = [group for _, group in pivot_table.groupby(level='sub_id')]
        data = np.stack([df.values for df in df_list])

    significant_ranges = {}
    p_vals = {}
    time_array = np.arange(-2, 10, 0.1)
    print(f'number of analyzed participants: {data.shape[0]}')
    print(f'number of permutations: {data.shape[1]}')

    if is_regression:
        significant_ranges[cond_var], p_vals[cond_var] = perform_permutation_analysis(data, cond_var, test_type=0)
    else:
        significant_ranges[cond_var], p_vals[cond_var] = perform_permutation_analysis(data, cond_var, test_type=1)
        for i, condition in enumerate(conditions):
            significant_ranges[condition], p_vals[condition] = perform_permutation_analysis(data, cond_var, test_type=0, condition=i)

    print(significant_ranges)
    return significant_ranges, p_vals
