from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_ROOT = "./data/FairVision"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GENDER_NAMES: tuple[str, ...] = ("female", "male")  # NPZ key "male": 0/1
RACE_NAMES: tuple[str, ...] = ("asian", "black", "white")  # 0/1/2
ATTR_CARDINALITIES: dict[str, int] = {"gender": 2, "race": 3}

DemographicMode = Literal["gender", "race", ""]


class FundusNpzDataset(Dataset):
    """FairVision Glaucoma NPZ dataset (matches ERM / Fusion / Fairness / ContextFair)."""

    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = 224,
        demographic_mode: DemographicMode | str = "",
    ):
        """
        Args:
            data_root: Root with train/val/test subfolders of ``.npz`` files.
            split: ``train`` | ``val`` | ``test``.
            image_size: Resize target (default 224).
            demographic_mode: If ``gender`` or ``race``, also return ``subgroup``.
                Empty string → ERM/Fusion/Fairness dict only (no ``subgroup``).
        """
        self.split_dir = os.path.join(data_root, split)
        self.image_size = int(image_size)
        self.demographic_mode = str(demographic_mode).strip().lower()
        if self.demographic_mode not in ("", "gender", "race"):
            raise ValueError(
                f"demographic_mode must be '', 'gender', or 'race'; got {demographic_mode!r}"
            )
        # ContextFair / staged pipelines read NPZ key "male" for gender mode.
        self.demographic_key = "male" if self.demographic_mode == "gender" else (
            "race" if self.demographic_mode == "race" else ""
        )

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        self.files = sorted(f for f in os.listdir(self.split_dir) if f.endswith(".npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _get_scalar(data: Any, key_candidates: list[str], default: int = -1) -> int:
        for k in key_candidates:
            if k in data:
                return int(float(np.asarray(data[k]).reshape(-1)[0]))
        return default

    def _load_image(self, data: Any, path: str) -> torch.Tensor:
        fundus = data["slo_fundus"].astype(np.float32) / 255.0
        image = torch.from_numpy(fundus)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        elif image.ndim == 3 and image.shape[0] in (1, 3):
            pass
        else:
            raise ValueError(f"Unexpected image shape {tuple(image.shape)} in {path}")

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
        return ((image - mean) / std).float()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filename = self.files[idx]
        path = os.path.join(self.split_dir, filename)
        data = np.load(path, allow_pickle=True)

        image = self._load_image(data, path)
        label = int(np.asarray(data["glaucoma"]).reshape(-1)[0])
        gender = self._get_scalar(data, ["male", "gender", "sex", "is_male"], default=-1)
        race = self._get_scalar(data, ["race"], default=-1)

        sample_id = None
        for key in ("sample_id", "id", "image_id"):
            if key in data:
                sample_id = str(np.asarray(data[key]).reshape(-1)[0])
                break
        if sample_id is None:
            sample_id = Path(filename).stem

        out: dict[str, Any] = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.long),
            "race": torch.tensor(race, dtype=torch.long),
            "sample_id": sample_id,
        }

        if self.demographic_key:
            subgroup_key = self.demographic_key
            if subgroup_key not in data:
                for k in ("male", "gender", "sex", "is_male", "race"):
                    if k in data:
                        subgroup_key = k
                        break
                else:
                    raise KeyError(f"Demographic key '{self.demographic_key}' missing in {path}")
            subgroup = int(float(np.asarray(data[subgroup_key]).reshape(-1)[0]))
            out["subgroup"] = torch.tensor(subgroup, dtype=torch.long)

        return out


def fairvision_collate_with_subgroup(
    demographic_mode: str,
    num_subgroups: int | None = None,
) -> Any:
    """
    Collate for ContextFair when the dataset was built without ``demographic_mode``.
    Sets ``subgroup`` from ``gender`` (mode=gender) or ``race`` (mode=race).
    """
    mode = str(demographic_mode).strip().lower()
    if mode not in ("gender", "race"):
        raise ValueError(f"demographic_mode must be gender or race; got {demographic_mode!r}")
    n = int(num_subgroups) if num_subgroups is not None else (2 if mode == "gender" else 3)
    src = "gender" if mode == "gender" else "race"

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in batch[0]:
            if key in ("image", "label", "gender", "race", "subgroup"):
                out[key] = torch.stack([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        if "subgroup" not in out:
            out["subgroup"] = out[src].clamp(0, n - 1)
        else:
            out["subgroup"] = out["subgroup"].clamp(0, n - 1)
        return out

    return collate


def build_fairvision_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    demographic_mode: DemographicMode | str = "",
    pin_memory: bool = True,
    collate_fn: Any | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test loaders.

    - ERM / Fusion / Fairness: leave ``demographic_mode=""`` (default).
    - ContextFair: set ``demographic_mode="gender"`` or ``"race"`` so batches
      include ``subgroup``.
    """
    mode = str(demographic_mode).strip().lower()
    train_ds = FundusNpzDataset(data_root, "train", image_size, mode)
    val_ds = FundusNpzDataset(data_root, "val", image_size, mode)
    test_ds = FundusNpzDataset(data_root, "test", image_size, mode)

    kw: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
    }
    if collate_fn is not None:
        kw["collate_fn"] = collate_fn

    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# Alias used by some ERM scripts
build_dataloaders = build_fairvision_dataloaders


if __name__ == "__main__":
    root = os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
    if not os.path.isdir(root):
        print(f"DATA_ROOT not found ({root}); skip smoke test.")
    else:
        train, val, test = build_fairvision_dataloaders(root, batch_size=2, num_workers=0, demographic_mode="gender")
        print(f"train={len(train.dataset)} val={len(val.dataset)} test={len(test.dataset)}")
        b = next(iter(train))
        print({k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in b.items()})
