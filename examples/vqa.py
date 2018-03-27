from deepzen.dataset.nlvr import load_nlvr


train, test = load_nlvr()
(images, texts), labels = train
print(images.shape, images.dtype)
print(len(texts), texts[0].__class__.__name__)
print(labels.shape, labels.dtype)
