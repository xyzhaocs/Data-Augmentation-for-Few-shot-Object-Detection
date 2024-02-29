import cv2
import random
import os
import yaml
import sys
from collections import defaultdict
from functools import cache
import shutil

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
        return len(self.images)


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

    def get(self):
        image = dataset.getImage(self.index)
        img_height, img_width, _ = image.shape
        h, w = self.h * img_height, self.w * img_width
        instance = image[
            int(self.y * img_height - h / 2) : int(self.y * img_height + h / 2),
            int(self.x * img_width - w / 2) : int(self.x * img_width + w / 2),
        ]
        scale = (
            CONFIG.min_scale + (CONFIG.max_scale - CONFIG.min_scale) * random.random()
        )
        h, w = h * scale, w * scale
        while h >= CONFIG.height or w >= CONFIG.width:
            h /= 2
            w /= 2
        return cv2.resize(instance, (int(w), int(h)))


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

        if len(self.name) < 128:
            self.name += f"_{instance.tag}_{x}_{y}_{w}_{h}"
        self.instances.append((x, y, w, h))
        self.canvas = cv2.seamlessClone(
            img_instance, self.canvas, None, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE
        )
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

class Spliter():
    def __init__(self, dataset: DataSet, val_ratio: float) -> None:
        self.total_size=len(dataset)
        self.val_ratio=val_ratio
        self.images_path=dataset.image_path
        self.labels_path=dataset.label_path
        self.train_images_path=os.path.join(self.images_path, "train")
        self.val_images_path=os.path.join(self.images_path, "val")
        self.train_labels_path=os.path.join(self.labels_path, "train")
        self.val_labels_path=os.path.join(self.labels_path, "val")
        
        if os.path.exists(self.train_images_path) == False:
            os.makedirs(self.train_images_path)
        if os.path.exists(self.val_images_path) == False:
            os.makedirs(self.val_images_path)
        if os.path.exists(self.train_labels_path) == False:
            os.makedirs(self.train_labels_path)
        if os.path.exists(self.val_labels_path) == False:
            os.makedirs(self.val_labels_path)
        
        # 获取所有图像文件和标签文件
        self.images_files = [f for f in os.listdir(self.images_path) if f.endswith(".jpg")]
        self.labels_files = [f for f in os.listdir(self.labels_path) if f.endswith(".txt")]

    def split(self):
        # 计算训练集和验证集大小
        self.val_size =  int(self.total_size * self.val_ratio)
        self.train_size = self.total_size - self.val_size
        
        # 随机选择验证集的图像文件和标签文件
        val_image_files = random.sample(self.images_files, self.val_size)
        val_label_files = [f.replace(".jpg", ".txt") for f in val_image_files]
        # 剩下文件即为训练集
        train_image_files = [x for x in self.images_files if x not in val_image_files]
        train_label_files = [x for x in self.labels_files if x not in val_label_files]

        if CONFIG.use_tqdm:
            # 将训练集文件移动到训练集目录
            for train_image_file, train_label_file in tqdm(zip(train_image_files, train_label_files), desc="Split Train Set... ", unit="(jpg+txt)"):
                shutil.move(os.path.join(self.images_path, train_image_file), os.path.join(self.train_images_path, train_image_file))
                shutil.move(os.path.join(self.labels_path, train_label_file), os.path.join(self.train_labels_path, train_label_file))
            # 将验证集文件移动到验证集目录
            for val_image_file, val_label_file in tqdm(zip(val_image_files, val_label_files), desc="Split Validation Set... ", unit="(jpg+txt)"):
                shutil.move(os.path.join(self.images_path, val_image_file), os.path.join(self.val_images_path, val_image_file))
                shutil.move(os.path.join(self.labels_path, val_label_file), os.path.join(self.val_labels_path, val_label_file))
        else:
            # 将训练集文件移动到训练集目录
            for train_image_file, train_label_file in zip(train_image_files, train_label_files):
                shutil.move(os.path.join(self.images_path, train_image_file), os.path.join(self.train_images_path, train_image_file))
                shutil.move(os.path.join(self.labels_path, train_label_file), os.path.join(self.train_labels_path, train_label_file))
            # 将验证集文件移动到验证集目录
            for val_image_file, val_label_file in zip(val_image_files, val_label_files):
                shutil.move(os.path.join(self.images_path, val_image_file), os.path.join(self.val_images_path, val_image_file))
                shutil.move(os.path.join(self.labels_path, val_label_file), os.path.join(self.val_labels_path, val_label_file))
        
        return self.train_size, self.val_size

    def __len__(self):
        return self.total_size



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
    if CONFIG.use_tqdm:
        from tqdm import tqdm
        for instance in tqdm(instance_dict.get_instances(), desc="Generating... ", unit="instance"):
            manager.add_instance(instance)
    else:
        for instance in instance_dict.get_instances():
            manager.add_instance(instance)

    new_dataset = DataSet(
        os.path.join(CONFIG.output_path, "images"),
        os.path.join(CONFIG.output_path, "labels"),
    )
    instance_dict = InstanceDict(new_dataset)
    new_distribution = instance_dict.distribution()

    print(
        "Distribution before augmentation: ",
        [(t1[0], t1[1] + t2[1]) for t1, t2 in zip(distribution, new_distribution)],
    )
    
    total_size = len(new_dataset)
    
    if CONFIG.val_ratio == -1 :
        print(f"The size of newDataSet: {total_size}")
        sys.exit()

    spliter = Spliter(new_dataset, CONFIG.val_ratio)
    train_size, val_size = spliter.split()
    
    print(f'The size of newDataSet: {total_size}, ',
          f'{train_size} for Train, {val_size} for Validation.')