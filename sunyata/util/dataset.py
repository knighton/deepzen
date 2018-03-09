def train_test_split(x, y, val):
    """
    Perform a training/test split.

    in:
        np.ndarray  x     X samples
        np.ndarray  y     Y samples
        float       val   Fraction of data held out

    out:
        tuple       data  The dataset as (x_train, y_train), (x_test, y_test)
    """
    assert 0 < val < 1
    data = list(zip(x, y))
    np.random.shuffle(data)
    split = int(len(data) * val)
    x_train = x[split:]
    y_train = y[split:]
    x_test = x[:split]
    y_test = y[:split]
    return (x_train, y_train), (x_test, y_test)
