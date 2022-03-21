from CONSTANTS import *


def cut_by_82_with_shuffle(instances):
    '''
    Experimental Setting with LogAnomaly
    Parameters
    ----------
    instances All instances in log data.

    Returns
    -------
    Training validating and testing log data.
    '''
    np.random.seed(seed)
    np.random.shuffle(instances)
    train_split = math.ceil(0.8 * len(instances))
    train = random.sample(instances, train_split)
    train = instances[:train_split]
    test = instances[train_split:]
    return train, None, test


def cut_by_613(instances):
    '''
    Experimental Setting according to PLELog.
    Parameters
    ----------
    instances: All instances in log data

    Returns
    -------
    training, validating and testing log data.
    '''
    train_split = int(0.7 * len(instances))
    train, test = instances[:train_split], instances[train_split:]
    random.seed(seed)
    random.shuffle(train)
    train_split = int(0.6 * len(instances))
    dev = train[train_split:]
    train = train[:train_split]
    return train, dev, test


