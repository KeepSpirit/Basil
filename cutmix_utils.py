# https://github.com/ayulockin/DataAugmentationTF/blob/master/CIFAR_10_with_CutMix_Augmentation.ipynb
import numpy as np
import tensorflow as tf


def get_bbox(l, IMG_SHAPE=32):
    cut_rat = tf.math.sqrt(1. - l)

    cut_w = IMG_SHAPE * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMG_SHAPE * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cx = tf.random.uniform((1,), minval=0, maxval=IMG_SHAPE, dtype=tf.int32)  # rx
    cy = tf.random.uniform((1,), minval=0, maxval=IMG_SHAPE, dtype=tf.int32)  # ry

    bbx1 = tf.clip_by_value(cx[0] - cut_w // 2, 0, IMG_SHAPE)
    bby1 = tf.clip_by_value(cy[0] - cut_h // 2, 0, IMG_SHAPE)
    bbx2 = tf.clip_by_value(cx[0] + cut_w // 2, 0, IMG_SHAPE)
    bby2 = tf.clip_by_value(cy[0] + cut_h // 2, 0, IMG_SHAPE)

    target_h = bby2 - bby1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - bbx1
    if target_w == 0:
        target_w += 1

    return bbx1.numpy(), bby1.numpy(), target_h.numpy(), target_w.numpy()


def cutmix(x, y, alpha, IMG_SHAPE=32):
    indexes = np.random.permutation(range(len(x)))
    mixed_x = []
    y_a = []
    y_b = []
    lam = []
    for i in range(len(x)):
        ## Get sample from beta distribution
        # dist = tfp.distributions.Beta(alpha, beta)
        ## Lambda
        # l = dist.sample(1)[0][0]

        l = np.random.beta(alpha, alpha)

        ## Get bbox ofsets and heights and widths
        bbx1, bby1, target_h, target_w = get_bbox(l, IMG_SHAPE=IMG_SHAPE)

        ## Get patch from image2
        crop2 = tf.image.crop_to_bounding_box(x[indexes[i]], bby1, bbx1, target_h, target_w)
        ## Pad the patch with same offset
        image2 = tf.image.pad_to_bounding_box(crop2, bby1, bbx1, IMG_SHAPE, IMG_SHAPE)
        ## Get patch from x
        crop1 = tf.image.crop_to_bounding_box(x[i], bby1, bbx1, target_h, target_w)
        ## Pad the patch with same offset
        img1 = tf.image.pad_to_bounding_box(crop1, bby1, bbx1, IMG_SHAPE, IMG_SHAPE)

        ## Subtract the patch from image1 so that patch from image2 can be put on instead
        image1 = x[i] - img1
        ## Add modified image1 and image2 to get cutmix image
        image = image1 + image2
        mixed_x.append(image)

        ## Adjust lambda according to pixel ration
        l = 1 - (target_w * target_h) / (IMG_SHAPE * IMG_SHAPE)
        lam.append(l)
        # l = tf.cast(l, tf.float32)
        # print(l)

        ## Combine labels
        # label = l * y[i] + (1 - l) * y[indexes[i]]
        y_a.append(y[i])
        y_b.append(y[indexes[i]])

    mixed_x = np.array(mixed_x).reshape(x.shape)
    y_a = np.reshape(y_a, (-1, 1))
    y_b = np.reshape(y_b, (-1, 1))
    lam = np.reshape(lam, (-1, 1))
    return mixed_x, y_a, y_b, lam, indexes


