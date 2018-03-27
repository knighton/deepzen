from collections import defaultdict
from glob import glob
import json
import numpy as np
import os
from PIL import Image
from time import time
from tqdm import tqdm

from ..util.config import get_dataset_root_dir
from ..util.py import require_kwargs


ALL_SPLITS = 'train', 'dev', 'test'
DATASET_NAME = 'nlvr'
PROC_SUBDIR = 'proc'
REPO_URL = 'https://github.com/clic-lab/nlvr'


def _clone_dataset_repo(repo_url, dataset_dir):
    os.mkdir(dataset_dir)
    cmd = 'mkdir -p %s; cd %s; git clone %s' % \
        (dataset_dir, dataset_dir, repo_url)
    assert not os.system(cmd)


def _load_image(f):
    im = Image.open(f)
    im = im.convert('RGB')
    assert im.size == (400, 100)
    im = im.resize((200, 50))
    im = np.array(im, dtype=np.uint8)
    im = np.rollaxis(im, 2)
    return im


def _get_meta_filename(proc_dir, split):
    return os.path.join(proc_dir, '%s_meta.txt' % split)


def _get_images_filename(proc_dir, split):
    return os.path.join(proc_dir, '%s_images.bin' % split)


def _get_samples_filename(proc_dir, split):
    return os.path.join(proc_dir, '%s_samples.jsonl' % split)


def _save_split(proc_dir, split, filename2image, samples, verbose=2):
    assert verbose in {0, 1, 2}

    # Write sample images as single numpy array.
    filenames = list(zip(*samples))[0]
    images = np.stack(map(lambda f: filename2image[f], filenames))
    if verbose:
        print('* Saving images of shape %s...' % (images.shape,))
        t0 = time()
    filename = _get_images_filename(proc_dir, split)
    images.tofile(filename)
    if verbose:
        t = time() - t0
        print('* ...took %.3f sec.' % t)

    # Write the samples as JSON per line.
    filename = _get_samples_filename(proc_dir, split)
    if verbose:
        t0 = time()
    with open(filename, 'wb') as out:
        for image, text, label in samples:
            x = {
                'image': image,
                'text': text,
                'label': label,
            }
            line = json.dumps(x) + '\n'
            out.write(line.encode('utf-8'))
    if verbose:
        t = time() - t0
        print('* Saving %d samples took %.3f sec.' % (len(samples), t))

    # Write metadata.
    filename = _get_meta_filename(proc_dir, split)
    with open(filename, 'wb') as out:
        x = {
            'count': len(samples),
            'image_shape': images[0].shape,
        }
        line = json.dumps(x) + '\n'
        out.write(line.encode('utf-8'))


def _load_split(proc_dir, split, verbose=2):
    assert verbose in {0, 1, 2}

    if verbose:
        print('Loading %s split:' % split)

    filename = _get_meta_filename(proc_dir, split)
    x = json.load(open(filename))
    count = x['count']
    image_shape = tuple(x['image_shape'])
    images_shape = (count,) + image_shape

    if verbose:
        t0 = time()
    filename = _get_images_filename(proc_dir, split)
    images = np.fromfile(filename, 'uint8')
    images = images.reshape(images_shape)
    if verbose:
        t = time() - t0
        print('* Loading images of shape %s took %.3f sec.' % (images_shape, t))

    if verbose:
        t0 = time()
    filename = _get_samples_filename(proc_dir, split)
    texts = []
    labels = []
    for line in open(filename):
        x = json.loads(line)

        text = x['text']
        assert text
        assert isinstance(text, str)
        texts.append(text)

        label = x['label']
        assert label in {0, 1}
        labels.append(label)
    labels = np.array(labels, 'uint8')
    if verbose:
        t = time() - t0
        print('* Loading %d samples took %.3f sec.' % (len(texts), t))

    return (images, texts), labels


def _preprocess_split(repo_url, dataset_dir, proc_dir, split, verbose=2):
    assert verbose in {0, 1, 2}

    if verbose:
        print('Preprocessing %s split:' % split)

    # Find the cloned repo.
    repo_dir = os.path.join(dataset_dir, os.path.basename(repo_url))

    # Load each sample: (sentence, label, image collection name).
    collection_samples = []
    filename = os.path.join(repo_dir, split, '%s.json' % split)
    for line in open(filename):
        j = json.loads(line)
        sentence = j['sentence']
        label = int(bool(j['label']))
        collection = j['identifier']
        collection_samples.append((collection, sentence, label))
    if verbose:
        print('* Read %d samples (image collection name, sentence, label).' %
              len(collection_samples))

    # Find image files by filename.
    pattern = os.path.join(repo_dir, split, 'images', '*', '*')
    filenames = glob(pattern)
    if verbose:
        print('* Found %d image files.' % len(filenames))

    # Build mapping of image collection name -> list of image filenames.
    collection2filenames = defaultdict(list)
    for filename in filenames:
        a = filename.rfind(split) + len(split) + 1
        z = filename.rfind('-')
        collection = filename[a:z]
        collection2filenames[collection].append(filename)
    if verbose:
        print('* Grouped image files into %d collections.' %
              len(collection2filenames))

    # Load each image: filename -> image pixel data.
    filename2image = {}
    if verbose:
        print('* Loading images into memory...')
        t0 = time()
    if verbose == 2:
        filenames = tqdm(filenames, leave=False)
    for filename in filenames:
        image = _load_image(filename)
        filename2image[filename] = image
    if verbose:
        t = time() - t0
        print('* ...took %.3f sec.' % t)

    # Get each sample: (image filename, sentence, label).
    #
    # We impose an ordering so the resulting dataset will be the same across
    # split preprocessing runs, although sample order will be randomized
    # normally each epoch during training.
    samples = []
    for collection, sentence, label in sorted(collection_samples):
        for filename in sorted(collection2filenames[collection]):
            samples.append((filename, sentence, label))
    if verbose:
        print('* Now have %d samples (sentence, label, image file).' %
              len(samples))

    # Write the organized split data.
    _save_split(proc_dir, split, filename2image, samples, verbose)


@require_kwargs
def load_nlvr(all_splits=ALL_SPLITS, dataset_name=DATASET_NAME,
              proc_subdir=PROC_SUBDIR, repo_url=REPO_URL, test_split='dev',
              train_split='train', verbose=2):
    assert test_split in all_splits
    assert train_split in all_splits
    dataset_dir = get_dataset_root_dir(dataset_name)
    proc_dir = os.path.join(dataset_dir, proc_subdir)
    if not os.path.exists(proc_dir):
        repo_dir = os.path.join(dataset_dir, os.path.basename(repo_url))
        if not os.path.exists(repo_dir):
            _clone_dataset_repo(repo_url, dataset_dir)
        os.mkdir(proc_dir)
        for split in all_splits:
            _preprocess_split(repo_url, dataset_dir, proc_dir, split, verbose)
    train = _load_split(proc_dir, train_split, verbose)
    test = _load_split(proc_dir, test_split, verbose)
    return train, test
