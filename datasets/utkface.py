from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, pil_to_tensor, resize

AGE_CLASSES: tuple[str, ...] = ("0-19", "20-39", "40+")
GENDER_CLASSES: tuple[str, ...] = ("male", "female")  # 0, 1 in CSV
RACE_CLASSES: tuple[str, ...] = ("White", "Black", "Asian", "Indian", "Others")
ATTR_CARDINALITIES: dict[str, int] = {"age": 3, "gender": 2, "race": 5}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_ROOT = "./data/UTKFace"
TRAIN_CSV = "utkface_label_train_stratified.csv"
VAL_CSV = "utkface_label_val_stratified.csv"
TEST_CSV = "utkface_label_test.csv"
IMAGE_SUBDIR = "utkface_aligned_cropped/UTKFace"
DEFAULT_TARGET = "gender"  # 0=male, 1=female in CSV

TargetAttr = Literal["age", "gender", "race"]
DemographicMode = Literal["age", "gender", "race"]


@dataclass
class UTKFaceDataConfig:
    data_root: str = DEFAULT_DATA_ROOT
    image_subdir: str = IMAGE_SUBDIR
    train_csv: str = TRAIN_CSV
    val_csv: str = VAL_CSV
    test_csv: str = TEST_CSV
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    pin_memory: bool = True


class UTKFaceDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        csv_name: str,
        image_size: int = 224,
        image_subdir: str = IMAGE_SUBDIR,
    ):
        self.data_root = Path(data_root)
        self.image_root = self.data_root / image_subdir
        self.image_size = image_size
        csv_path = self.data_root / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"Label CSV not found: {csv_path}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")

        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns and "file" in self.df.columns:
            self.df = self.df.rename(columns={"file": "image"})
        required = {"image", "age", "gender", "race"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        self.age = torch.tensor(self.df["age"].astype("int64").tolist(), dtype=torch.long)
        self.gender = torch.tensor(self.df["gender"].astype("int64").tolist(), dtype=torch.long)
        self.race = torch.tensor(self.df["race"].astype("int64").tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> Path:
        return self.image_root / Path(rel_path).name

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self._resolve_path(rel_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            image = pil_to_tensor(img.convert("RGB")).float() / 255.0
        image = resize(image, size=[self.image_size, self.image_size], antialias=True)
        return normalize(image, IMAGENET_MEAN, IMAGENET_STD)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rel_path = str(self.df.iloc[idx]["image"])
        return {
            "image": self._load_image(rel_path),
            "age": self.age[idx],
            "gender": self.gender[idx],
            "race": self.race[idx],
            "image_path": rel_path,
            "sample_id": Path(rel_path).name,
        }


def utkface_collate_with_target(target: TargetAttr | str = DEFAULT_TARGET) -> Callable:
    """ERM / Fusion / Fairness collate: stacks attrs and sets ``label = target``."""
    if target not in {"age", "gender", "race"}:
        raise ValueError(f"target must be age, gender, or race; got {target!r}")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in batch[0]:
            if key in ("image", "age", "gender", "race"):
                out[key] = torch.stack([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        out["label"] = out[target]
        return out

    return collate


def utkface_collate_with_subgroup(
    target: TargetAttr | str = DEFAULT_TARGET,
    demographic_mode: DemographicMode | str = "race",
    num_subgroups: int | None = None,
) -> Callable:
    """
    ContextFair / staged collate: ``label`` from target + ``subgroup`` from
    demographic_mode (clamped).
    """
    base = utkface_collate_with_target(target)
    mode = str(demographic_mode).strip().lower()
    if mode not in {"age", "gender", "race"}:
        raise ValueError(f"demographic_mode must be age, gender, or race; got {demographic_mode!r}")
    n = int(num_subgroups) if num_subgroups is not None else ATTR_CARDINALITIES[mode]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out = base(batch)
        out["subgroup"] = out[mode].long().clamp(0, n - 1)
        return out

    return collate


def build_utkface_dataloaders(
    cfg: UTKFaceDataConfig | None = None,
    target: TargetAttr | str = DEFAULT_TARGET,
    demographic_mode: DemographicMode | str | None = None,
    num_subgroups: int | None = None,
    collate_fn: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test loaders.

    - ERM / Fusion / Fairness: leave ``demographic_mode=None`` (default).
    - ContextFair: set ``demographic_mode`` (e.g. ``race`` or ``age``) to add
      ``subgroup`` via the staged collate.
    """
    cfg = cfg or UTKFaceDataConfig()
    if collate_fn is None:
        if demographic_mode is not None:
            collate_fn = utkface_collate_with_subgroup(
                target=target,
                demographic_mode=demographic_mode,
                num_subgroups=num_subgroups,
            )
        else:
            collate_fn = utkface_collate_with_target(target)

    datasets = [
        UTKFaceDataset(cfg.data_root, csv, cfg.image_size, cfg.image_subdir)
        for csv in (cfg.train_csv, cfg.val_csv, cfg.test_csv)
    ]
    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory and torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(datasets[0], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets[1], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets[2], shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# Convenience alias
build_dataloaders = build_utkface_dataloaders


if __name__ == "__main__":
    root = os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
    if not os.path.isdir(root):
        print(f"DATA_ROOT not found ({root}); skip smoke test.")
    else:
        cfg = UTKFaceDataConfig(data_root=root, batch_size=4, num_workers=0)
        train, val, test = build_utkface_dataloaders(cfg, demographic_mode="race")
        print(f"train={len(train.dataset)} val={len(val.dataset)} test={len(test.dataset)}")
        b = next(iter(train))
        print({k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in b.items()})
