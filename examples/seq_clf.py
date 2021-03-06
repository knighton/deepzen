from argparse import ArgumentParser

from deepzen.layer import *  # noqa
from deepzen.model import Model
from deepzen.task.imdb import load_imdb
from deepzen.task.quora_dupes import load_quora_dupes


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--task', type=str, default='imdb',
                    help='The dataset to train on.')
    ap.add_argument('--model', type=str, default='cnn',
                    help='The model architecture to train.')
    ap.add_argument('--start', type=int, default=0, help='Start epoch.')
    ap.add_argument('--stop', type=int, default=100, help='Stop epoch.')
    ap.add_argument('--batch', type=int, default=128, help='Batch size.')
    ap.add_argument('--optim', type=str, default='adam', help='Optimizer.')
    ap.add_argument('--spy', type=str, default='server,progress_bar,rows',
                    help='List of training monitors.')
    return ap.parse_args()


class Tasks(object):
    """
    A collection of image classification tasks.

    Select with --task.
    """

    @classmethod
    def imdb(cls):
        dataset, x_transform = load_imdb()
        class_names = 'neg', 'pos'
        return dataset, class_names

    @classmethod
    def quora(cls):
        dataset, x_transform = load_quora_dupes()
        class_names = 'ok', 'dupe'
        return dataset, class_names

    @classmethod
    def get(cls, dataset_name):
        get_dataset = getattr(cls, dataset_name)
        return get_dataset()


class Models(object):
    """
    A collection of image classification models.

    Select with --model.
    """

    @classmethod
    def cnn(cls, seq_len, vocab_size, x_dtype, y_dtype):
        return SequenceSpec([
            DataSpec((seq_len,), x_dtype),
            EmbedSpec(vocab_size, 16, y_dtype),

            ConvSpec(8, stride=2),
            ReLUSpec(),

            ConvSpec(8, stride=2),
            ReLUSpec(),

            ConvSpec(8, stride=2),
            ReLUSpec(),

            ConvSpec(8, stride=2),
            ReLUSpec(),

            ConvSpec(8, stride=2),
            ReLUSpec(),

            ConvSpec(8, stride=2),
            ReLUSpec(),

            FlattenSpec(),

            DenseSpec(1),
            SigmoidSpec(),
        ])

    @classmethod
    def get(cls, model_name, dataset):
        (x_train, y_train), _ = dataset
        seq_len = x_train.shape[1]
        vocab_size = int(x_train.max()) + 1
        x_dtype = x_train.dtype.name
        y_dtype = y_train.dtype.name
        get_model_spec = getattr(cls, model_name)
        spec = get_model_spec(seq_len, vocab_size, x_dtype, y_dtype)
        return Model(spec)


def run(args):
    dataset, class_names = Tasks.get(args.task)
    model = Models.get(args.model, dataset)
    model.fit_clf(dataset, epochs=args.epochs, batch=args.batch,
                  optim=args.optim, spy=args.spy)


if __name__ == '__main__':
    run(parse_args())
