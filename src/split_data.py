import shutil
import random
from pathlib import Path

def split_dataset(raw_dir: str, splits_dir: str, val_ratio: float = 0.2):
    random.seed(42)
    src = Path(raw_dir)
    dst = Path(splits_dir)

    if dst.exists():
        shutil.rmtree(dst)

    for species_dir in src.iterdir():
        if not species_dir.is_dir():
            continue

        images = list(species_dir.glob('*.*'))
        random.shuffle(images)

        n_val = int(len(images) * val_ratio)
        
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split_name, img_list in [('train', train_images), ('val', val_images)]:
            split_path = dst / split_name / species_dir.name
            split_path.mkdir(parents=True, exist_ok=True)

            for img in img_list:
                shutil.copy2(img, split_path / img.name)

    print("Data splitting complete.")
    
    for split_name in ['train', 'val']:
        split_path = dst / split_name
        if split_path.exists():
            count = sum(len(list(p.glob('*.*'))) for p in split_path.iterdir())
            print(f"{split_name.capitalize()} images: {count}")

if __name__ == '__main__':
    split_dataset(
        raw_dir='data/raw/gb_wildlife_dataset',
        splits_dir='data/splits',
        val_ratio=0.2
    )