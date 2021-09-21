import numpy as np
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis


def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def noise_augmentation(x, y, mean=0, std=10, augmentation_factor=10, min_threshold=None, max_threshold=None,
                       time_series=True):
    # self duplicate
    print(len(x))
    x_repeat = np.repeat(x, repeats=augmentation_factor, axis=0)
    y_repeat = np.repeat(y, repeats=augmentation_factor, axis=0)

    # augumentation
    for sample_index in range(0, len(x_repeat)):
        sample_shape = x_repeat[sample_index].shape
        x_repeat[sample_index] = x_repeat[sample_index] + np.random.normal(mean, std, x_repeat[sample_index].shape)
        a = x_repeat[sample_index]
        # thresholding x sample
        if min_threshold:
            x_repeat[sample_index][x_repeat[sample_index] <= min_threshold] = min_threshold

        if max_threshold:
            x_repeat[sample_index][x_repeat[sample_index] >= max_threshold] = max_threshold

    x = np.concatenate((x, x_repeat))
    y = np.concatenate((y, y_repeat))

    return x, y


# def dtw_image_archive(target_ts_img, known_ts_img, duration=120):
#     # 120 * 8 * 16   120 * 8 * 16
#     # for target_ts_data
#     # (height, width, 3) # 3 is 3 channels for Red, Green, Blue
#     total_distance = 0
#     target_ts_img = target_ts_img.reshape((duration, -1))
#     known_ts_img = known_ts_img.reshape((duration, -1))
#
#     for channel in range(0, target_ts_img.shape[-1]):
#         distance, paths = dtw.warping_paths(target_ts_img[:, channel], known_ts_img[:, channel])
#         total_distance += distance
#
#     return total_distance
#
#
# def dtw_rd(target_ts_img, known_ts_img, duration=120, filename="warp.png"):
#     # 120 * 8 * 16   120 * 8 * 16
#     # for target_ts_data
#     # (height, width, 3) # 3 is 3 channels for Red, Green, Blue
#
#     # target_ts_img = np.abs(target_ts_img)
#     # known_ts_img = np.abs(known_ts_img)
#
#     target_ts_img[target_ts_img < 0] = 0
#     known_ts_img[known_ts_img < 0] = 0
#
#     tar_neg_speed_avg = target_ts_img[:, :, 0:8].mean(axis=(1, 2))
#     tar_pos_speed_avg = target_ts_img[:, :, 8:16].mean(axis=(1, 2))
#     known_neg_speed_avg = known_ts_img[:, :, 0:8].mean(axis=(1, 2))
#     known_pos_speed_avg = known_ts_img[:, :, 8:16].mean(axis=(1, 2))
#
#     distance_neg, paths_neg = dtw.warping_paths(s1=tar_neg_speed_avg, s2=known_neg_speed_avg)
#     distance_pos, paths_pos = dtw.warping_paths(s1=tar_pos_speed_avg, s2=known_pos_speed_avg)
#
#     distance = distance_neg + distance_pos
#     # for channel in range(0, target_ts_img.shape[-1]):
#     #     distance, paths = dtw.warping_paths(target_ts_img[:, channel], known_ts_img[:, channel])
#     #     total_distance += distance
#
#     path = dtw.warping_path(tar_neg_speed_avg, known_neg_speed_avg)
#     dtwvis.plot_warping(tar_neg_speed_avg.flatten(), known_neg_speed_avg.flatten(), path, filename=filename + '_neg')
#
#     path = dtw.warping_path(tar_pos_speed_avg, known_pos_speed_avg)
#     dtwvis.plot_warping(tar_pos_speed_avg.flatten(), known_pos_speed_avg.flatten(), path, filename=filename + '_pos')
#
#     return distance


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # if empty result

    if len(s) == 0 or len(t) == 0:
        if len(s) == 0 and len(t) == 0:
            ratio = 1
            distance = 0
        else:
            # longer = len(s) if len(s) > 0 else longer = len(t)
            if len(s)!=0:
                longer=len(s)
            else:
                longer=len(t)
            ratio=0
            distance=longer

        if ratio_calc is True:
            return ratio
        else:
            return "The strings are {} edits away".format(distance)
        # if ratio_calc==True:
        #     if len(s)==0 and len(t)==0:
        #         return float(0)
        #     else:
        #         return len(s) if len(s)>len(t) else len(t)

        # else:
        #     distance=len(s)
        #
        #     return "The strings are {} edits away".format()

    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def replace_special(target_str: str, replacement_dict):
    for special, replacement in replacement_dict.items():
        # print('replacing ' + special)
        target_str = target_str.replace(special, replacement)
    return target_str

#
# def train_test_eq_split(X, y, n_per_class, random_state=3):
#     if random_state:
#         np.random.seed(random_state)
#     sampled = X.groupby(y, sort=False).apply(
#         lambda frame: frame.sample(n_per_class))
#     mask = sampled.index.get_level_values(1)
#
#     X_train = X.drop(mask)
#     X_test = X.loc[mask]
#     y_train = y.drop(mask)
#     y_test = y.loc[mask]
#
#     return X_train, X_test, y_train, y_test

