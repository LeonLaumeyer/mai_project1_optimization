import kagglehub
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class IntelCustomDataset(Dataset):
    def __init__(self, root_dir, resize=(150, 150), transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform or transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class IntelImageClassificationDataset:
    def __init__(self, resize=(150, 150)) -> None:
        path = kagglehub.dataset_download("puneet6060/intel-image-classification")
        train_path = os.path.join(path, "seg_train/seg_train")
        test_path = os.path.join(path, "seg_test/seg_test")
        eval_path = os.path.join(path, "seg_pred")

        self.train_dataset = IntelCustomDataset(train_path, resize=resize)
        self.test_dataset = IntelCustomDataset(test_path, resize=resize)
        self.eval_dataset = IntelCustomDataset(eval_path, resize=resize)

    def label(self, i: int) -> str:
        return {
            0: "buildings",
            1: "forest",
            2: "glacier",
            3: "mountain",
            4: "sea",
            5: "street"
        }[i]