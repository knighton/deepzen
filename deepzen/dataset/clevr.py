from io import BytesIO
import json
import numpy as np
import os
from PIL import Image
import re
from time import time
from tqdm import tqdm
from zipfile import ZipFile

from ..util.config import get_dataset_root_dir
from ..util.net import download
from ..util.py import require_kwargs


DATASET_NAME = 'clevr'

MAIN_ALL_SPLITS = 'train', 'val', 'test'
MAIN_IMAGE_SHAPE = 64, 96
MAIN_ORIG_IMAGE_SHAPE = 320, 480
MAIN_PROC_SUBDIR = 'main_proc'
MAIN_URL = 'https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip'

COGENT_ALL_SPLITS = 'trainA', 'valA', 'valB', 'testA', 'testB'
COGENT_IMAGE_SHAPE = 64, 96
COGENT_ORIG_IMAGE_SHAPE = 320, 480
COGENT_PROC_SUBDIR = 'cogent_processed'
COGENT_URL = 'https://s3-us-west-1.amazonaws.com/clevr/CLEVR_CoGenT_v1.0.zip'

IMAGE_RE = re.compile('.+/images/.+.png')
QUESTION_RE = re.compile('.+/questions/.+.json')


def _get_samples_filename(proc_dir, split):
    return os.path.join(proc_dir, '%s_samples.jsonl' % split)


def _unzip_samples_split(zip_file, zip_paths, proc_dir, split, verbose):
    paths = []
    for path in zip_paths:
        if not QUESTION_RE.match(path):
            continue
        if split not in path:
            continue
        paths.append(path)
    assert len(paths) == 1
    zip_path, = paths

    text = zip_file.open(zip_path).read().decode('utf-8')
    x = json.loads(text)
    questions = x['questions']

    if 2 <= verbose:
        questions = tqdm(questions, leave=False)

    filename = _get_samples_filename(proc_dir, split)
    with open(filename, 'wb') as out:
        for x in questions:
            image = x['image_index']
            question = x['question']
            answer = x.get('answer')
            x = {
                'image': image,
                'text': question,
                'label': answer,
            }
            line = json.dumps(x, sort_keys=True) + '\n'
            out.write(line.encode('utf-8'))


def _unzip_samples(zip_file, zip_paths, proc_dir, all_splits, verbose):
    os.mkdir(proc_dir)
    if verbose:
        print('Samples not collected, processing all splits:')
    for split in all_splits:
        if verbose:
            t0 = time()
        _unzip_samples_split(zip_file, zip_paths, proc_dir, split, verbose)
        if verbose:
            t = time() - t0
            print('* %s took %.3f sec.' % (split, t))


def _get_images_filename(proc_dir, split):
    return os.path.join(proc_dir, '%s_images.bin' % split)


def _unzip_and_scale_image(zip_file, zip_path, orig_image_shape, image_shape):
    data = zip_file.open(zip_path).read()
    im = Image.open(BytesIO(data))
    im = im.convert('RGB')
    oh, ow = orig_image_shape
    assert im.size == (ow, oh)
    h, w = image_shape
    im = im.resize((w, h))
    x = np.array(im, 'uint8')
    x = np.rollaxis(x, 2)
    return x


def _unzip_and_scale_images_split(zip_file, zip_paths, proc_dir, split,
                                  orig_image_shape, image_shape, verbose):
    paths = []
    for path in zip_paths:
        if not IMAGE_RE.match(path):
            continue
        if split not in path:
            continue
        paths.append(path)
    paths.sort()
    if verbose:
        print('  * Unzipping and downsampling %d images %s -> %s...' %
              (len(paths), orig_image_shape, image_shape))
        t0 = time()
    if 2 <= verbose:
        paths = tqdm(paths, leave=False)
    filename = _get_images_filename(proc_dir, split)
    with open(filename, 'wb') as out:
        for path in paths:
            image = _unzip_and_scale_image(
                zip_file, path, orig_image_shape, image_shape)
            out.write(image.tobytes())
    if verbose:
        t = time() - t0
        print('  * ...took %.3f sec.' % t)


def _unzip_and_scale_images(zip_file, zip_paths, proc_dir, all_splits,
                            orig_image_shape, image_shape, verbose):
    os.mkdir(proc_dir)
    if verbose:
        print('Images not cached at that resolution, processing all splits:')
    for split in all_splits:
        if verbose:
            print('* Split %s:' % split)
        _unzip_and_scale_images_split(zip_file, zip_paths, proc_dir, split,
                                      orig_image_shape, image_shape, verbose)


def _load_split_images(proc_dir, image_shape, split, verbose):
    if verbose:
        t0 = time()
    filename = _get_images_filename(proc_dir, split)
    images = np.fromfile(filename, 'uint8')
    images = images.reshape((-1, 3) + image_shape)
    if verbose:
        t = time() - t0
        print('  * Loading %s images took %.3f sec.' % (images.shape, t))
    return images


def _load_split_samples(proc_dir, split, verbose):
    if verbose:
        t0 = time()
    filename = _get_samples_filename(proc_dir, split)
    image_indices = []
    texts = []
    labels = []
    for line in open(filename):
        x = json.loads(line)
        image_indices.append(x['image'])
        texts.append(x['text'])
        labels.append(x['label'])
    image_indices = np.array(image_indices, 'uint32')
    if verbose:
        t = time() - t0
        print('  * Loading %d samples took %.3f sec.' % (len(image_indices), t))
    return image_indices, texts, labels


def _load_split(parent_proc_dir, child_proc_dir, image_shape, split, verbose):
    if verbose:
        print('* Split %s:' % split)
    images = _load_split_images(child_proc_dir, image_shape, split, verbose)
    samples = _load_split_samples(parent_proc_dir, split, verbose)
    return images, samples


def _unpack_image_shape(image_shape, orig_image_shape):
    if isinstance(image_shape, str):
        h, w = map(int, image_shape.split('x'))
    else:
        h, w = image_shape
        assert isinstance(h, int)
        assert isinstance(w, int)
    oh, ow = orig_image_shape
    assert 0 < h <= oh
    assert 0 < w <= ow
    return h, w


def _get_image_shape_str(image_shape):
    return 'x'.join(map(str, image_shape))


def _load_zip(zip_filename, verbose):
    if verbose:
        print('* Loading archive at %s' % zip_filename)
        t0 = time()
    zip_file = ZipFile(zip_filename)
    zip_paths = sorted(zip_file.namelist())
    if verbose:
        t = time() - t0
        print('* ...took %.3f sec.' % t)
    return zip_file, zip_paths


def _load(all_splits, dataset_name, image_shape, orig_image_shape, proc_subdir,
          test_split, train_split, url, verbose):
    image_shape = _unpack_image_shape(image_shape, orig_image_shape)
    dataset_dir = get_dataset_root_dir(dataset_name)
    zip_filename = os.path.join(dataset_dir, os.path.basename(url))
    parent_proc_dir = os.path.join(dataset_dir, proc_subdir)
    child_subdir = _get_image_shape_str(image_shape)
    child_proc_dir = os.path.join(parent_proc_dir, child_subdir)
    if not os.path.exists(zip_filename):
        download(url, zip_filename, verbose)
    if not os.path.exists(parent_proc_dir) or \
            not os.path.exists(child_proc_dir):
        zip_file, zip_paths = _load_zip(zip_filename, verbose)
        if not os.path.exists(parent_proc_dir):
            _unzip_samples(zip_file, zip_paths, parent_proc_dir, all_splits,
                           verbose)
        if not os.path.exists(child_proc_dir):
            _unzip_and_scale_images(zip_file, zip_paths, child_proc_dir,
                                    all_splits, orig_image_shape, image_shape,
                                    verbose)
    if verbose:
        print('Loading:')
    train = _load_split(parent_proc_dir, child_proc_dir, image_shape,
                        train_split, verbose)
    test = _load_split(parent_proc_dir, child_proc_dir, image_shape, test_split,
                       verbose)
    return train, test


@require_kwargs
def load_clevr_main(
        all_splits=MAIN_ALL_SPLITS, dataset_name=DATASET_NAME,
        image_shape=MAIN_IMAGE_SHAPE, orig_image_shape=MAIN_ORIG_IMAGE_SHAPE,
        proc_subdir=MAIN_PROC_SUBDIR, test_split='val', train_split='train',
        url=MAIN_URL, verbose=2):
    return _load(all_splits, dataset_name, image_shape, orig_image_shape,
                 proc_subdir, test_split, train_split, url, verbose)


@require_kwargs
def load_clevr_cogent_same(
        all_splits=COGENT_ALL_SPLITS, dataset_name=DATASET_NAME,
        image_shape=COGENT_IMAGE_SHAPE,
        orig_image_shape=COGENT_ORIG_IMAGE_SHAPE,
        proc_subdir=COGENT_PROC_SUBDIR, test_split='valA', train_split='trainA',
        url=COGENT_URL, verbose=2):
    return _load(all_splits, dataset_name, image_shape, orig_image_shape,
                 proc_subdir, test_split, train_split, url, verbose)


@require_kwargs
def load_clevr_cogent_different(
        all_splits=COGENT_ALL_SPLITS, dataset_name=DATASET_NAME,
        image_shape=COGENT_IMAGE_SHAPE,
        orig_image_shape=COGENT_ORIG_IMAGE_SHAPE,
        proc_subdir=COGENT_PROC_SUBDIR, test_split='valB', train_split='trainA',
        url=COGENT_URL, verbose=2):
    return _load(all_splits, dataset_name, image_shape, orig_image_shape,
                 proc_subdir, test_split, train_split, url, verbose)
