"""Datasets under <data_root>/datasets/.

    train  DIV2K/DIV2K_train_HR, Flickr2K/Flickr2K_HR, CLIC/train (any mix)
    valid  DIV2K/DIV2K_valid_HR, CLIC/valid
    test   Kodak, CLIC/test

Protocols: train = random crop + flip; native = centre crop at the training
size; crop = resize + centre crop; full = the whole image, cropped only to the
model's size multiple (batch 1).
"""

import os
import random
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

TRAIN = {"DIV2K": "DIV2K/DIV2K_train_HR", "Flickr2K": "Flickr2K/Flickr2K_HR", "CLIC": "CLIC/train"}
VALID = {"DIV2K": "DIV2K/DIV2K_valid_HR", "CLIC": "CLIC/valid"}
TEST = {"Kodak": "Kodak", "CLIC": "CLIC/test"}
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def gather_paths(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"dataset folder not found: {folder}")
    out = [os.path.join(r, f) for r, _, files in os.walk(folder) for f in files
           if f.lower().endswith(EXTENSIONS)]
    if not out:
        raise FileNotFoundError(f"no images in {folder}")
    return sorted(out)


class CenterCropToMultiple:
    def __init__(self, multiple):
        self.m = multiple

    def __call__(self, img):
        w, h = img.size
        nw, nh = w - w % self.m, h - h % self.m
        if nw == 0 or nh == 0:
            raise ValueError(f"{w}x{h} image is smaller than the size multiple {self.m}")
        return transforms.functional.center_crop(img, (nh, nw))


def build_transform(mode, size, multiple):
    tail = [transforms.ToTensor()]
    if mode == "train":
        return transforms.Compose([transforms.RandomCrop(size, pad_if_needed=True,
                                                         padding_mode="reflect"),
                                   transforms.RandomHorizontalFlip()] + tail)
    if mode == "native":
        return transforms.Compose([transforms.CenterCrop(size)] + tail)
    if mode == "crop":
        return transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)] + tail)
    if mode == "full":
        return transforms.Compose([CenterCropToMultiple(multiple)] + tail)
    raise ValueError(f"unknown protocol {mode!r}")


class ImageFolder(Dataset):
    def __init__(self, folder, transform):
        self.paths, self.transform = gather_paths(folder), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB"))


class FixedCrops(Dataset):
    """Deterministic crops for offline sweeps (tools/alloc_sweep.py): crop j of
    image i is the same on every run, so curve files of different checkpoints
    describe the same pixels. Images smaller than the crop are upscaled first."""

    def __init__(self, folders, size, per_image=1, seed=0, max_images=None):
        paths = [p for f in folders for p in gather_paths(f)]
        if max_images is not None and max_images < len(paths):
            paths = sorted(random.Random(seed).sample(paths, max_images))
        self.items = [(p, j) for p in paths for j in range(per_image)]
        self.size, self.seed = int(size), int(seed)
        self.names = [f"{os.path.basename(p)}#{j}" for p, j in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, _ = self.items[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        w, h, s = img.size[0], img.size[1], self.size
        if min(w, h) < s:
            f = s / min(w, h)
            img = img.resize((max(s, round(w * f)), max(s, round(h * f))), Image.BICUBIC)
            w, h = img.size
        rng = random.Random(self.seed * 1_000_003 + i)
        left, top = rng.randint(0, w - s), rng.randint(0, h - s)
        return transforms.functional.to_tensor(img.crop((left, top, left + s, top + s)))


def sweep_dataset(cfg, split, crop=512, per_image=1, seed=0, max_images=None):
    """The images an allocation sweep scores: the test set under the test
    protocol, or deterministic crops of the train / valid folders."""
    root = os.path.join(cfg.data_root, "datasets")
    if split == "test":
        ds = ImageFolder(os.path.join(root, TEST[cfg.testset]),
                         build_transform(cfg.test_protocol, cfg.img_size, cfg.size_multiple))
        if max_images is not None:
            ds.paths = ds.paths[:max_images]
        ds.names = [os.path.basename(p) for p in ds.paths]
        return ds
    if crop % cfg.size_multiple:
        raise ValueError(f"--crop {crop} must be a multiple of {cfg.size_multiple} for this model")
    folders = ([os.path.join(root, TRAIN[n]) for n in cfg.trainsets] if split == "train"
               else [os.path.join(root, VALID[cfg.validset])])
    return FixedCrops(folders, crop, per_image, seed, max_images)


def _loader(ds, batch, workers, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last, persistent_workers=workers > 0)


def get_loaders(cfg, rank=0, world_size=1, train=True):
    root = os.path.join(cfg.data_root, "datasets")
    workers = cfg.num_workers if cfg.num_workers is not None else min(8, os.cpu_count() or 1)
    out = SimpleNamespace(train=None, valid=None, test=None, sampler=None)
    if train:
        tf = build_transform("train", cfg.img_size, cfg.size_multiple)
        ds = ConcatDataset([ImageFolder(os.path.join(root, TRAIN[n]), tf) for n in cfg.trainsets])
        if world_size > 1:
            out.sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        out.train = _loader(ds, cfg.batch_size, workers, shuffle=True, sampler=out.sampler,
                            drop_last=True)
    if rank == 0:   # evaluation runs on rank 0 only
        wanted = (("valid", VALID, cfg.validset, cfg.valid_protocol),) if train else ()
        for name, table, key, proto in wanted + (("test", TEST, cfg.testset, cfg.test_protocol),):
            ds = ImageFolder(os.path.join(root, table[key]),
                             build_transform(proto, cfg.img_size, cfg.size_multiple))
            setattr(out, name, _loader(ds, 1 if proto == "full" else cfg.batch_size,
                                       min(workers, 4)))
    return out
