from argparse import ArgumentParser
import numpy as np

from deepzen.data.dataset import Dataset
from deepzen.data.split import Split
from deepzen.data import unpack_dataset
from deepzen.model.base.trainer import Trainer
from deepzen.node import *  # noqa
from deepzen.task.mnist import load_mnist


def parse_flags():
    ap = ArgumentParser()

    ap.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs to fit.')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    ap.add_argument('--latent_dim', type=int, default=128,
                    help='Dimensionality of the embeddings fed to generator.')
    ap.add_argument('--soft_zero', type=float, default=0)
    ap.add_argument('--soft_one', type=float, default=0.95)

    ap.add_argument('--d_metrics', type=str, default='xe,acc;xe,acc')
    ap.add_argument('--d_optim', type=str, default='adam')
    ap.add_argument('--d_spy', type=str, default='rows,server:1337')
    ap.add_argument('--d_timer_cache', type=int, default=10000)

    ap.add_argument('--gd_metrics', type=str, default='xe,acc;xe,acc')
    ap.add_argument('--gd_optim', type=str, default='adam')
    ap.add_argument('--gd_spy', type=str, default='progress_bar,server:1338')
    ap.add_argument('--gd_timer_cache', type=int, default=10000)

    return ap.parse_args()


def build_generator(latent_dim, num_classes):
    latent = Input((latent_dim,), 'float32')

    klass = Sequence(
        Input((1,), 'int64'),
        Embed(num_classes, latent_dim),
        Flatten,
    )

    fake_image = Sequence(
        Product()(latent, klass),

        Dense(384 * 3 * 3),
        ReLU,

        # To (384, 3, 3).
        Reshape((384, 3, 3)),

        # To (192, 7, 7).
        ConvTranspose(192, 5, padding=0),
        ReLU,
        BatchNorm,

        # To (96, 14, 14).
        ConvTranspose(96, 5, stride=2, padding=2, out_padding=1),
        ReLU,
        BatchNorm,

        # To (1, 28, 28).
        ConvTranspose(1, 5, stride=2, padding=2, out_padding=1),
        Tanh,
    )

    return Network([latent, klass], fake_image)


def build_discriminator(num_classes):
    image = Sequence(
        Input((1, 28, 28), 'float32'),

        Conv(32, 3, stride=2, padding=1),
        LeakyReLU(0.2),
        Dropout(0.3),

        Conv(64, 3, stride=1, padding=1),
        LeakyReLU(0.2),
        Dropout(0.3),

        Conv(128, 3, stride=2, padding=1),
        LeakyReLU(0.2),
        Dropout(0.3),

        Conv(256, 3, stride=1, padding=1),
        LeakyReLU(0.2),
        Dropout(0.3),

        Flatten,
    )
    is_real = (Dense(1) > Sigmoid)(image)
    klass = (Dense(num_classes) > Softmax)(image)
    return Network(image, [is_real, klass])


class DiscriminatorSplit(Split):
    def __init__(self, mnist, generator, latent_dim, num_classes, soft_zero,
                 soft_one):
        self.mnist = mnist
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.soft_zero = soft_zero
        self.soft_one = soft_one

    def num_samples(self):
        return self.mnist.num_samples() * 2

    def sample(self, index):
        assert False

    def shapes(self, batch_size=None):
        if batch_size is None:
            images = 1, 28, 28
            is_reals = 1,
            classes = 10,
        else:
            images = batch_size, 1, 28, 28
            is_reals = batch_size, 1
            classes = batch_size, 10
        xx = images,
        yy = is_reals, classes
        return xx, yy

    def dtypes(self):
        return ('float32'), ('float32', 'float32')

    def num_batches(self, batch_size):
        assert batch_size % 2 == 0
        return self.mnist.num_batches(batch_size // 2)

    def make_batch(self, batch_size, xx, yy):
        assert batch_size % 2 == 0
        half_batch_size = batch_size // 2

        # Collect input images (real + fake).
        real_images, = xx
        noise = np.random.uniform(-1, 1, (half_batch_size, self.latent_dim))
        noise = noise.astype('float32')
        fake_class_indices = \
            np.random.randint(0, self.num_classes, (half_batch_size, 1))
        fake_images, = self.generator.predict([noise, fake_class_indices])
        images = np.concatenate([real_images, fake_images], axis=0)

        # Collect output is_real's (real + fake).
        #
        # Uses one-sided soft real/fake classes (see original Keras).
        is_reals = np.array(
            [self.soft_one] * half_batch_size +
            [self.soft_zero] * half_batch_size, 'float32')

        # Collect output classes (real + fake).
        real_classes, = yy
        fake_classes = np.zeros((half_batch_size, self.num_classes), 'float32')
        fake_classes[np.arange(half_batch_size), fake_class_indices[:, 0]] = 1
        classes = np.concatenate([real_classes, fake_classes], axis=0)

        # Organize and return.
        xx = images,
        yy = is_reals, classes
        return xx, yy

    def each_batch(self, batch_size):
        assert batch_size % 2 == 0
        for xx, yy in self.mnist.each_batch(batch_size // 2):
            yield self.make_batch(batch_size, xx, yy)


def make_discriminator_dataset(mnist, generator, latent_dim, num_classes,
                               soft_zero, soft_one):
    train = DiscriminatorSplit(
        mnist.train, generator, latent_dim, num_classes, soft_zero, soft_one)
    test = DiscriminatorSplit(
        mnist.test, generator, latent_dim, num_classes, soft_zero, soft_one)
    return Dataset(train, test)


class GeneratorSplit(Split):
    def __init__(self, samples_per_epoch, latent_dim, num_classes, soft_one):
        self.samples_per_epoch = samples_per_epoch
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.soft_one = soft_one

    def num_samples(self):
        return self.samples_per_epoch

    def sample(self, index):
        assert False

    def shapes(self, batch_size=None):
        if batch_size is None:
            noise = self.latent_dim,
            class_indices = 1,
            tricked_is_reals = 1,
            classes_one_hot = self.num_classes,
        else:
            assert batch_size % 2 == 0
            noise = batch_size, self.latent_dim
            class_indices = batch_size, 1
            tricked_is_reals = batch_size, 1
            classes_one_hot = batch_size, self.num_classes
        xx = noise, class_indices
        yy = tricked_is_reals, classes_one_hot
        return xx, yy

    def dtypes(self):
        return ('float32', 'int64'), ('float32', 'float32')

    def make_batch(self, batch_size):
        assert batch_size % 2 == 0

        # Create input noise.
        noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
        noise = noise.astype('float32')

        # Create input classes as indices.
        class_indices = np.random.randint(0, self.num_classes, (batch_size, 1))

        # Create output is_real assuming we've fooled the discriminator.
        tricked_is_reals = np.full((batch_size, 1), self.soft_one, 'float32')

        # Create output classes as one-hot floats.
        classes_one_hot = np.zeros((batch_size, self.num_classes), 'float32')
        classes_one_hot[np.arange(batch_size), class_indices[:, 0]] = 1

        # Organize and return.
        xx = noise, class_indices
        yy = tricked_is_reals, classes_one_hot
        return xx, yy

    def each_batch(self, batch_size):
        num_batches = self.samples_per_epoch // batch_size
        for i in range(num_batches):
            yield self.make_batch(batch_size)


def make_generator_dataset(train_samples_per_epoch, test_samples_per_epoch,
                           latent_dim, num_classes, soft_one):
    train = GeneratorSplit(
        train_samples_per_epoch, latent_dim, num_classes, soft_one)
    test = GeneratorSplit(
        test_samples_per_epoch, latent_dim, num_classes, soft_one)
    return Dataset(train, test)


def run(flags):
    # Load the MNIST dataset.
    mnist, class_names = load_mnist()
    mnist = unpack_dataset(mnist)
    num_classes = len(class_names)

    # Define the models.
    generator = build_generator(flags.latent_dim, num_classes)
    discriminator = build_discriminator(num_classes)
    generator_discriminator = generator > discriminator

    # Wrap MNIST with custom datasets tailored to the two models.
    d_dataset = make_discriminator_dataset(
        mnist, generator, flags.latent_dim, num_classes, flags.soft_zero,
        flags.soft_one)
    gd_dataset = make_generator_dataset(
        d_dataset.train.num_samples(), d_dataset.test.num_samples(),
        flags.latent_dim, num_classes, flags.soft_one)

    # Now create the model trainers.
    begin_epoch = 0
    d_trainer = Trainer.init_from_args(
        d_dataset, flags.d_metrics, None, flags.d_optim, flags.batch_size,
        begin_epoch, flags.num_epochs, flags.d_spy, flags.d_timer_cache)
    gd_trainer = Trainer.init_from_args(
        gd_dataset, flags.gd_metrics, None, flags.gd_optim, flags.batch_size,
        begin_epoch, flags.num_epochs, flags.gd_spy, flags.gd_timer_cache)

    # Build the models and give their parameters to their optimizers.
    generator_discriminator.build()
    d_trainer.optimizer.set_params(discriminator.params())
    gd_trainer.optimizer.set_params(generator.params())

    # Fit the models, alternating batches between them.
    discriminator.fit_before(d_trainer)
    generator_discriminator.fit_before(gd_trainer)
    batches_per_epoch = d_dataset.num_batches(flags.batch_size)
    d_each_batch = d_dataset.each_batch_forever(flags.batch_size)
    gd_each_batch = gd_dataset.each_batch_forever(flags.batch_size)
    for epoch in range(flags.num_epochs):
        for batch in range(batches_per_epoch):
            # Fit the discriminator.
            (xx, yy), is_training = next(d_each_batch)
            discriminator.fit_on_batch(d_trainer, is_training, xx, yy)

            # Fit the generator (via the combined model).
            (xx, yy), is_training = next(gd_each_batch)
            generator_discriminator.fit_on_batch(
                gd_trainer, is_training, xx, yy)
    discriminator.fit_after(d_trainer)
    generator_discriminator.fit_after(gd_trainer)


if __name__ == '__main__':
    run(parse_flags())
