import numpy as np
import pandas as pd

X = np.random.randint(5, size=(15, 2))
y = np.random.randint(2, size=15)


###########################################################################################


class timefold(object):
    """
    Cross-validation tools for timeseries data.

    TO DO:  * Binary stratification
            * Multiclass stratification
            * Continuous stratification
            * Holdout sample
            * Split into one def per CV method
            * Shrinkage folds
            * Write docstrings
            * Write documentation
            * Write Github doc including designs
            * Create examples ipynb
            * Package and publish to PyPI https://pypi.org/help/#publishing
            * Write unit-tests
    """

    def __init__(self, folds=10, method='nested', step_size=1):
        self.folds = folds
        self.method = method
        self.step_size = step_size

    def split(self, X):
        """
        Generate indices for splitting data into train-test pairs.
        """
        folds = self.folds
        method = self.method
        step_size = self.step_size

        X_obs = X.shape[0]
        indices = np.arange(X_obs)

        if folds >= X_obs:
            raise ValueError(
                ("The number of folds {0} must be smaller than the number of observations {1}".format(folds, X_obs)))

        folds += 1
        fold_indices = np.array_split(indices, folds, axis=0)
        fold_sizes = [len(fold) for fold in fold_indices][:-1]
        train_starts = [fold[0] for fold in fold_indices][:-1]
        train_ends = [fold[0] for fold in fold_indices][1:]

        if method == 'nested':
            for end, size in zip(train_ends, fold_sizes):
                yield(indices[:end], indices[end:end + size])

        elif method == 'window':
            for start, end, size in zip(train_starts, train_ends, fold_sizes):
                yield(indices[start:end], indices[end:end + size])

        elif method == 'step':
            steps = indices[1:]
            for step in steps:
                yield(indices[:step], indices[step:step + step_size])

        elif method == 'stratified':
            pass

        elif method == 'shrink':
            for step in steps:
                yield(indices[:step], indices[step:step + step_size])

        else:
            raise ValueError("Unknown method supplied '{0}'. Method must be one of: 'nested', 'window', 'step', "
                             "'stratified'".format(method))


# Create a timefold object with desired settings
tf = timefold(folds=10, method='nested')

# Generate and print the generated train-test pairs
for train_index, test_index in tf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    y_train, y_test = y[train_index], y[test_index]


###########################################################################################
# STRATIFIED
###########################################################################################

y = np.random.randint(2, size=25)
y_obs = y.shape[0]

y = np.asarray(y)  # coerce to array
y_unique, y_inverse = np.unique(y, return_inverse=True)
class_count = np.bincount(y_inverse)


folds = 3 # self.folds
class_timefolds = [
    timefold(folds, method='nested').split(np.zeros(max(count, folds))) for count in class_count]

test_folds = np.zeros(y_obs, dtype=np.int)



for test_fold_indices, per_cls_splits in enumerate(list(zip(*class_timefolds))):


for test_fold_indices, per_cls_splits in enumerate(zip(*class_timefolds)):
    for cls, (_, test_split) in zip(y_unique, per_cls_splits):
        cls_test_folds = test_folds[y == cls]
        test_split = test_split[test_split < len(cls_test_folds)]
        cls_test_folds[test_split] = test_fold_indices
        test_folds[y == cls] = cls_test_folds

test_folds

list(zip(y, test_folds))



len(test_folds)
X = np.random.randint(5, size=(len(test_folds), 2))
np.unique(test_folds)

np.bincount(test_folds)

x = pd.DataFrame(list(zip(y, test_folds)))
x.groupby(x[1]).sum()


np.unique(test_folds)


def yield_strat(test_folds, folds):
    len_test_folds = len(test_folds)
    fold_0 = len_test_folds // folds
    test_folds[0:fold_0] = -1
    unique_test_folds = np.unique(test_folds)
    for f in unique_test_folds:
        train_index = np.where(test_folds == f)
        test_index = np.where(test_folds == f + 2)
        yield(train_index, test_index)


for train_index, test_index in yield_strat(test_folds, folds):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]



###########################################################################################

X = np.random.randint(5, size=(25, 2))
y = np.random.randint(2, size=25)

df = pd.DataFrame(X)
df['target'] = y
df.head()


def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        print('\n%s:\n---------\ntotal:%d\ntrain_df:%d\ntest_df:%d' % (lbl, len(lbl_df), len(lbl_train_df), len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)

    return train_df, test_df



train, test = split_to_train_test(df, 'target', 0.5)


y = np.random.randint(2, size=25)


def get_train_test_inds(y, train_proportion=0.5):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and
    testing sets are preserved (stratified sampling).
    '''

    y = np.array(y)
    train_index = np.zeros(len(y), dtype=bool)
    test_index = np.zeros(len(y), dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        # np.random.shuffle(value_inds)
        n = int(train_proportion * len(value_inds))

        train_index[value_inds[:n]] = True
        test_index[value_inds[n:]] = True

    return train_index, test_index
    # yield train_index, test_index


for train_index, test_index in get_train_test_inds(y):
    print("TRAIN IND:", train_index, "TEST IND:", test_index)
    print("TRAIN:", y[train_index], "TEST:", y[test_index])

train_index, test_index = get_train_test_inds(y)
sum(y[train_index]), sum(y[test_index])
