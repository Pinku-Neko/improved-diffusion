import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

def main(class_to_dump=None):
    for split in ["train", "test"]:
        out_dir = f"fashion_mnist_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.FashionMNIST(
                root=tmp_dir, train=split == "train", download=True
            )
        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            
            # If a specific class is specified and the current image is not from that class, skip it
            if class_to_dump is not None and CLASSES[label] != class_to_dump:
                continue
            
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)

if __name__ == "__main__":
    # To dump images from a specific class, provide the class name (e.g., "T-shirt/top")
    # If you want to dump images from all classes, leave class_to_dump as None
    main()
