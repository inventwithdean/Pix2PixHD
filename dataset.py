import os
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, photos_dir, sketches_dir, target_width=1024):
        self.photos_dir = photos_dir
        self.sketches_dir = sketches_dir
        self.photos = os.listdir(photos_dir)
        self.sketches = os.listdir(sketches_dir)
        self.target_width = target_width

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        photo_file = self.photos[idx]
        sketch_file = self.sketches[idx]

        photo_path = os.path.join(self.photos_dir, photo_file)
        sketch_path = os.path.join(self.sketches_dir, sketch_file)

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")
        transform = T.Compose(
            [
                T.Resize((self.target_width, self.target_width)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        photo = transform(photo)
        sketch = transform(sketch)

        return photo, sketch
