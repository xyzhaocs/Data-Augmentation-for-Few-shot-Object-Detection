from functools import cache
import cv2
import numpy as np
import random
import os
import math
import yaml
import sys
from collections import defaultdict


class Config:
    def __init__(self, config: dict):
        self.__dict__.update(config)


class DataSet:
    def __init__(self, image_path: str, label_path: str) -> None:
        if os.path.exists(image_path) == False:
            raise ValueError(f'Image path "{image_path}" is not exist')
        if os.path.exists(label_path) == False:
            raise ValueError(f'Label path "{label_path}" is not exist')
        labels = os.listdir(label_path)
        invalid_jpg = []
        no_corresponding_label = []
        valid_image = []
        for file in os.listdir(image_path):
            if file[-4:] != ".jpg":
                invalid_jpg.append(file)
            elif file.replace(".jpg", ".txt") not in labels:
                no_corresponding_label.append(file)
            else:
                valid_image.append(file)
        if invalid_jpg:
            print(
                f"Number of invalid jpg: {len(invalid_jpg)} ",
                ",".join(invalid_jpg),
                file=sys.stderr,
            )
        if no_corresponding_label:
            print(
                f"Number of image without corresponding label {len(no_corresponding_label)} : ",
                ",".join(no_corresponding_label),
                file=sys.stderr,
            )

        self.image_path = image_path
        self.label_path = label_path
        self.images = valid_image

    @cache
    def getLabel(self, index: int):
        return open(
            os.path.join(self.label_path, self.images[index].replace(".jpg", ".txt")),
            "r",
        ).readlines()

    @cache
    def getImage(self, index: int):
        return cv2.imread(os.path.join(self.image_path, self.images[index]))

    def __getitem__(self, index: int):
        return self.getImage(index), self.getLabel(index)

    def __len__(self):
        return len(os.listdir(self.image_path))


class Instance:
    def __init__(
        self,
        dataset: DataSet,
        index: int,
        tag: int,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        self.tag, self.x, self.y, self.w, self.h = tag, x, y, w, h
        self.index = index
        self.dataset = dataset

    @cache
    def get(self):
        image = dataset.getImage(self.index)
        img_height, img_width, _ = image.shape
        h, w = self.h * img_height, self.w * img_width
        instance = image[
            int(self.y * img_height - h / 2) : int(self.y * img_height + h / 2),
            int(self.x * img_width - w / 2) : int(self.x * img_width + w / 2),
        ]
        if h >= CONFIG.height or w >= CONFIG.width:
            while h >= CONFIG.height or w >= CONFIG.width:
                h /= 2
                w /= 2
            return cv2.resize(instance, (int(w), int(h)))
        else:
            return instance


class InstanceDict:
    def __init__(self, dataset: DataSet) -> None:
        self.instance_dict = defaultdict(list)
        for index in range(len(dataset)):
            for label in dataset.getLabel(index):
                tag, x, y, w, h = label.split()
                self.instance_dict[int(tag)].append(
                    Instance(
                        dataset, index, int(tag), float(x), float(y), float(w), float(h)
                    )
                )

    def __len__(self):
        return len(self.instance_dict.keys())

    def __getitem__(self, index: int):
        return self.instance_dict[index]

    def distribution(self):
        return sorted(
            [(tag, len(instances)) for (tag, instances) in self.instance_dict.items()]
        )

    def get_instances(self):
        instance_list = []
        for index, num in self.distribution():
            if num < CONFIG.target_instances_per_class:
                instance_list.extend(
                    random.choices(
                        instance_dict[index], k=CONFIG.target_instances_per_class - num
                    )
                )

        random.shuffle(instance_list)

        return instance_list


class Canvas:
    def __init__(self) -> None:
        canvas_filename = random.choice(os.listdir(CONFIG.background_path))
        self.name = os.path.splitext(canvas_filename)[0]
        self.instances = []
        self.tags = []
        canvas = cv2.imread(
            os.path.join(
                CONFIG.background_path,
                canvas_filename,
            )
        )
        canvas_height, canvas_width, _ = canvas.shape
        if canvas_height != CONFIG.height or canvas_width != CONFIG.width:
            self.canvas = cv2.resize(canvas, (CONFIG.width, CONFIG.height))
        else:
            self.canvas = canvas
        self.limitation = random.randint(CONFIG.min_instances, CONFIG.max_instances)

    def place(self, instance: Instance):
        if self.limitation == 0:
            return False

        canvas_height, canvas_width, _ = self.canvas.shape
        img_instance = instance.get()
        h, w, _ = img_instance.shape
        x = random.randint(0, canvas_width - w)
        for _ in range(100):
            y = random.randint(0, canvas_height - h)
            for ix, iy, iw, ih in self.instances:
                if x < ix + iw and x + w > ix and y < iy + ih and y + h > iy:
                    break
            else:
                break
        else:
            return False

        self.name += f"_{instance.tag}_{x}_{y}_{w}_{h}"
        self.instances.append((x, y, w, h))
        self.canvas[y : y + h, x : x + w] = img_instance
        self.limitation -= 1
        self.tags.append(instance.tag)
        return True

    def save(self):
        cv2.imwrite(
            os.path.join(CONFIG.output_path, "images", f"{self.name}.jpg"), self.canvas
        )
        with open(
            os.path.join(CONFIG.output_path, "labels", f"{self.name}.txt"), "w"
        ) as f:
            for tag, xywh in zip(self.tags, self.instances):
                x, y, w, h = map(float, xywh)
                f.write(
                    f"{tag} {(x+w/2)/CONFIG.width} {(y+h/2)/CONFIG.height} {w/CONFIG.width} {h/CONFIG.height}\n"
                )


class CanvasManagement:
    def __init__(self) -> None:
        self.canvas = Canvas()

    def add_instance(self, instance):
        while not self.canvas.place(instance):
            self.canvas.save()
            self.canvas = Canvas()


if __name__ == "__main__":

    CONFIG = Config(yaml.safe_load(open("config.yaml", "r")))
    dataset = DataSet(CONFIG.image_path, CONFIG.label_path)

    if os.path.exists(CONFIG.background_path) == False:
        raise ValueError(f'Background_path "{CONFIG.background_path}" is not exist')

    if os.path.exists(os.path.join(CONFIG.output_path, "images")) == False:
        os.makedirs(os.path.join(CONFIG.output_path, "images"))
    if os.path.exists(os.path.join(CONFIG.output_path, "labels")) == False:
        os.makedirs(os.path.join(CONFIG.output_path, "labels"))

    instance_dict = InstanceDict(dataset)
    distribution = instance_dict.distribution()
    print("Distribution before augmentation: ", distribution)

    manager = CanvasManagement()
    for instance in instance_dict.get_instances():
        manager.add_instance(instance)

    new_dataset = DataSet(
        os.path.join(CONFIG.output_path, "images"),
        os.path.join(CONFIG.output_path, "labels"),
    )
    instance_dict = InstanceDict(new_dataset)
    distribution = instance_dict.distribution()
    print("Distribution before augmentation: ", distribution)
