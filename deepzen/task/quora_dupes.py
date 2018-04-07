import numpy as np
import os
import string

from ..transform import \
    Dict, Filter, Length, LowerEachSample, Numpy, Pipeline, Split
from ..util.net import download
from ..util.py import require_kwargs
from ..util.config import get_dataset_root_dir
from ..util.dataset import train_test_split


DATASET_NAME = 'quora_dupes'
URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'


def _load(filename, verbose, y_dtype):
    question_1s = []
    question_2s = []
    is_dupes = []
    lines = open(filename).read().strip().split('\n')[1:]
    for i, line in enumerate(lines):
        try:
            int(line.split()[0])
        except:
            lines[i - 1] += lines[i]
            lines[i] = None
    lines = list(filter(bool, lines))
    for line in lines:
        ss = line.split('\t')
        q1 = ss[3]
        question_1s.append(q1)
        q2 = ss[4]
        question_2s.append(q2)
        is_dupe = int(ss[5])
        assert is_dupe in {0, 1}
        is_dupes.append(is_dupe)
    y = np.array(is_dupes, y_dtype)
    return question_1s, question_2s, y


@require_kwargs
def load_quora_dupes_text(dataset_name=DATASET_NAME, test_frac=0.2, url=URL,
                          verbose=2, y_dtype='float32'):
    dataset_dir = get_dataset_root_dir(dataset_name)
    local = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    return _load(local, verbose, y_dtype)


@require_kwargs
def load_quora_dupes(dataset_name=DATASET_NAME, test_frac=0.2, url=URL,
                     verbose=2, x_transform=None, y_dtype='float32'):
    if x_transform is None:
        x_transform = Pipeline(
            LowerEachSample(), Filter(string.ascii_lowercase + ' '), Split(),
            Length(16), Dict(), Numpy('int64'))

    x1, x2, y = load_quora_dupes_text(
        dataset_name=dataset_name, url=url, verbose=verbose, y_dtype=y_dtype)

    x = list(map(lambda x: x[0] + '|' + x[1], zip(x1, x2)))

    (x_train, y_train), (x_test, y_test) = train_test_split(x, y, test_frac)

    x_train = x_transform.fit_transform(x_train)
    y_train = np.array(y_train, y_dtype)
    x_test = x_transform.transform(x_test)
    y_test = np.array(y_test, y_dtype)

    dataset = (x_train, y_train), (x_test, y_test)

    return dataset, x_transform
