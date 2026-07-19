import shutil
import random
from pathlib import Path


def split_dataset(raw_dir: str, splits_dir: str, val_ratio: float = 0.15):
    random.seed(42)

    src = Path(raw_dir)
    dst = Path(splits_dir)

    if dst.exists():
        shutil.rmtree(dst)

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    for species_dir in src.iterdir():
        if not species_dir.is_dir():
            continue

        images = [
            p for p in species_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        random.shuffle(images)

        n_val = int(len(images) * val_ratio)

        val_images = images[:n_val]
        train_images = images[n_val:]

        for split_name, img_list in [
            ('train', train_images),
            ('val', val_images)
        ]:
            split_path = dst / split_name / species_dir.name
            split_path.mkdir(parents=True, exist_ok=True)

            for img in img_list:
                # Avoid duplicate filenames when flattening nested folders
                if img.parent != species_dir:
                    new_name = f"{img.parent.name}_{img.name}"
                else:
                    new_name = img.name

                shutil.copy2(
                    img,
                    split_path / new_name
                )

    print("Data splitting complete.")

    for split_name in ['train', 'val']:
        split_path = dst / split_name

        if split_path.exists():
            print(f"\n{split_name.capitalize()}:")

            total = 0

            for class_dir in sorted(split_path.iterdir()):
                if class_dir.is_dir():
                    count = len([
                        f for f in class_dir.iterdir()
                        if f.is_file()
                    ])
                    print(f"  {class_dir.name}: {count}")
                    total += count

            print(f"  Total: {total}")


if __name__ == '__main__':
    split_dataset(
        raw_dir='data/raw/gb_wildlife_dataset',
        splits_dir='data/splits',
        val_ratio=0.15
    )