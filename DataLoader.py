import cv2
import numpy as np
import os
import random
import tensorflow.keras.backend as K
from IPython.display import Image as iImage


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128), root_dir="sketch_animated_by_scene_split"):  # "shapes"):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.color_dir = 'original'
        self.sketch_dir = 'sketch'
        self.root_dir = root_dir

        dirs = []
        n_files = []

        dir_path = self.root_dir
        if not os.path.isfile(dir_path):
            dirs.append(dir)
            n_files.append(len([filename for filename in os.listdir(dir_path)]))

        ### Para teste
        ### Pegando apenas metade dos diretórios
        # dirs = dirs[::8]
        # n_files = n_files[::8]

        n_files = np.asarray(n_files)

        self.scenes = dirs
        self.count = n_files.sum()
        self.scene_weights = n_files / self.count

    def rotate_images(self, imgs):
        # Rotate
        M = cv2.getRotationMatrix2D((imgs[0].shape[1] / 2, imgs[0].shape[0] / 2), random.randint(-5, 5), 1)

        rotated_images = [cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255)) for img in
                          imgs]

        return rotated_images

    def aug_flip(self, imgs):
        flip = random.randint(0, 3)
        if flip == 0:
            return imgs
        return [cv2.flip(img, flip - 2) for img in imgs]

    def load_image_triplet(self, normalize=True, augment=False):

        base_dir = self.root_dir

        sampled_files = [filename for filename in sorted(os.listdir(base_dir))]

        selected_index = random.randint(0, len(sampled_files) - 3)

        imgs = []

        for i in range(selected_index, selected_index + 3):
            imgs.append(self.load_image(os.path.join(base_dir, sampled_files[i]), normalize=normalize, grayscale=True))

        if augment:
            imgs = self.aug_flip(imgs)

        return imgs

    def load_all_image_triplet(self, normalize=True, augment=False):

        base_dir = self.root_dir

        sampled_files = [filename for filename in sorted(os.listdir(base_dir)) if
                         filename.endswith(".png") or filename.endswith(".jpg")]

        img_batch = []

        for selected_index in range(0, len(sampled_files) - 3):
            triplet = []
            for i in range(selected_index, selected_index + 3):
                triplet.append(
                    self.load_image(os.path.join(base_dir, sampled_files[i]), normalize=normalize, grayscale=True))

            if augment:
                triplet = self.aug_flip(triplet)

            img_batch.append(triplet)

        img_A = np.vstack([np.expand_dims(triplet[0], axis=0) for triplet in img_batch])
        img_B = np.vstack([np.expand_dims(triplet[1], axis=0) for triplet in img_batch])
        img_C = np.vstack([np.expand_dims(triplet[2], axis=0) for triplet in img_batch])

        return img_A, img_B, img_C

    def load_halfway_image_triplet(self, normalize=True, augment=False):

        base_dir = self.root_dir

        sampled_files = [filename for filename in sorted(os.listdir(base_dir))]

        halfway = (len(sampled_files) - 2) // 2

        images = []

        for i in range(halfway, halfway + 3):
            images.append(
                self.load_image(os.path.join(base_dir, sampled_files[i]), normalize=normalize, grayscale=True))

        if augment:
            images = self.aug_flip(images)

        im_set = [images]

        img_A = np.vstack([np.expand_dims(triplet[0], axis=0) for triplet in im_set])
        img_B = np.vstack([np.expand_dims(triplet[1], axis=0) for triplet in im_set])
        img_C = np.vstack([np.expand_dims(triplet[2], axis=0) for triplet in im_set])

        return img_A, img_B, img_C

    def load_image(self, filepath, normalize=True, grayscale=False):
        img = cv2.imread(filepath)

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_AREA)

        if normalize:
            img = img.astype(np.float32) / 127.5 - 1

        return img

    def load_data(self, batch_size, ids=None, augment=False):

        if ids == None:
            sampled_scenes = random.choices(
                population=self.scenes,
                weights=self.scene_weights,
                k=batch_size)
        else:
            scenes = []
            weights = []

            for scene, weight in zip(self.scenes, self.scene_weights):
                if scene in ids:
                    scenes.append(scene)
                    weights.append(weight)

            sampled_scenes = random.choices(
                population=scenes,
                weights=weights,
                k=batch_size)

        img_batch = []

        for scene_folder in sampled_scenes:
            img_batch.append(self.load_image_triplet(scene_folder, augment=augment))

        img_A = np.vstack([np.expand_dims(triplet[0], axis=0) for triplet in img_batch])
        img_B = np.vstack([np.expand_dims(triplet[1], axis=0) for triplet in img_batch])
        img_C = np.vstack([np.expand_dims(triplet[2], axis=0) for triplet in img_batch])

        return img_A, img_B, img_C

    def load_batch(self, batch_size, batches=20, augment=False):
        self.n_batches = batches
        for i in range(self.n_batches):
            yield self.load_data(batch_size, augment=augment)

    def load_dataset(self, augment=False):
        random.shuffle(self.scenes)

        for scene in self.scenes:
            img_batch = []
            img_batch.append(self.load_image_triplet(scene, augment=augment))

            img_A = np.vstack([np.expand_dims(triplet[0], axis=0) for triplet in img_batch])
            img_B = np.vstack([np.expand_dims(triplet[1], axis=0) for triplet in img_batch])
            img_C = np.vstack([np.expand_dims(triplet[2], axis=0) for triplet in img_batch])

            yield img_A, img_B, img_C


def distance_transform(img):
    _, dist = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 3)

    pi = 15
    dist = 1 - np.exp(-dist / pi)

    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist = dist.astype(np.uint8)

    dist = dist.astype(np.float32) / 127.5 - 1

    dist = np.expand_dims(dist, axis=2)

    return dist


def distance_transform_float(img):
    int_matrix = (127.5 * (img + 1)).astype(np.uint8)
    int_matrix = np.expand_dims(int_matrix, axis=2)
    return distance_transform(int_matrix)