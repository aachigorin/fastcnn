from __future__ import print_function

import imagenet_image_processing as imagenet
from imagenet_data import ImagenetData

from fastcnn.classifier.reader import BaseReader

if __name__ == '__main__':
    dataset = ImagenetData('validation')
    images, labels = imagenet.inputs(dataset)
    print(images, labels)
