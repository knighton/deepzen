from deepzen.dataset.clevr import load_clevr_main
from deepzen.dataset.nlvr import load_nlvr


train, test = load_nlvr()
(images, texts), labels = train
print(images.shape, images.dtype)
print(len(texts), texts[0].__class__.__name__)
print(labels.shape, labels.dtype)

print()

train, test = load_clevr_main()
images, (image_indices, texts, labels) = train
print(images.shape, images.dtype)
print(image_indices.shape, image_indices.dtype)
print(len(texts), texts[0].__class__.__name__)
print(len(labels), labels[0].__class__.__name__)
