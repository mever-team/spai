# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script for generating a CSV file for the DMID LDM training-validation dataset.

The DMID LDM dataset was introduced by the paper:
    Corvi, R., Cozzolino, D., Zingarini, G., Poggi, G., Nagano, K.,
    & Verdoliva, L. (2023, June). On the detection of synthetic images
    generated by diffusion models. In ICASSP 2023-2023 IEEE International
    Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5).
    IEEE.

Github page: https://github.com/grip-unina/DMimageDetection

The script requires the following:
- The train and test LDM data provided by the above github repository.
- The COCO 2017 train and val splits.
- The CNNDetect dataset, from which the LSUN authentic samples are obtained.
  This dataset can be downloaded by following the instructions from the following
  github repository: https://github.com/PeterWang512/CNNDetection
"""
from pathlib import Path
import random
from typing import Any, Optional

import click
from tqdm import tqdm

from spai import data_utils


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


@click.command()
@click.option("--train_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--val_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--coco_dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--lsun_dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--real_coco_filename", type=str, default="real_coco.txt")
@click.option("--real_lsun_filename", type=str, default="real_lsun.txt")
@click.option("-o", "--output_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True)
@click.option("-r", "--csv_root_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-d", "--output_csv_delimiter", type=str, default=",", show_default=True)
@click.option("-n", "--samples_num", type=int, default=None, show_default=True)
@click.option("-f", "--filter", type=str, multiple=True)
def main(
    train_dir: Optional[Path],
    val_dir: Optional[Path],
    coco_dir: Path,
    lsun_dir: Path,
    real_coco_filename: str,
    real_lsun_filename: str,
    output_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv_delimiter: str,
    samples_num: Optional[int],
    filter: list[str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    entries: list[dict[str, Any]] = []

    coco_copy_dir_name: str = "real_coco"
    lsun_copy_dir_name: str = "real_lsun"

    split_dirs: list[Path] = []
    split_labels: list[str] = []
    if train_dir is not None:
        split_dirs.append(train_dir)
        split_labels.append("train")
    if val_dir is not None:
        split_dirs.append(val_dir)
        split_labels.append("val")

    for s_dir, s_label in tqdm(zip(split_dirs, split_labels),
                               desc="Finding synthetic images", unit="image"):
        # Make entries for the synthetic LDM data.
        data_gen = s_dir.rglob("*")
        for p in data_gen:
            # if filetype.is_image(p):
            path_parts: list[str] = p.parts
            if (p.is_file() and p.suffix == ".png"
                    and coco_copy_dir_name not in path_parts
                    and lsun_copy_dir_name not in path_parts):
                filter_found: bool = False if len(filter) > 0 else True
                for f in filter:
                    if f in path_parts:
                        filter_found = True
                        break
                if not filter_found:
                    continue

                entries.append({
                    "image": str(p.relative_to(csv_root_dir)),
                    "class": 1,
                    "split": s_label
                })
                # valid_num += 1

        # Make entries for COCO real data.
        real_coco_file: Path = s_dir / real_coco_filename
        coco_samples: list[Path] = find_coco_samples(real_coco_file, coco_dir, s_label)
        for p in coco_samples:
            entries.append({
                "image": str(p.relative_to(csv_root_dir)),
                "class": 0,
                "split": s_label
            })

        # Make entries for LSUN real data.
        real_lsun_file: Path = s_dir / real_lsun_filename
        lsun_samples: list[Path] = find_lsun_samples(real_lsun_file, lsun_dir, s_label)
        for p in lsun_samples:
            entries.append({
                "image": str(p.relative_to(csv_root_dir)),
                "class": 0,
                "split": s_label
            })

    if samples_num is not None:
        entries = random.sample(entries, samples_num)

    data_utils.write_csv_file(entries, output_csv, delimiter=output_csv_delimiter)
    print(f"Exported CSV to {output_csv}")


def find_coco_samples(coco_real_file: Path, coco_dir: Path, split: str) -> list[Path]:
    assert split in ["train", "val"]
    print("Loading COCO image paths.")
    with coco_real_file.open() as f:
        lines: list[str] = [l.rstrip() for l in f]
    coco_files: list[Path] = [coco_dir / f"train2017" / l for l in lines]
    print("Loading of COCO image paths completed.")
    for f in tqdm(coco_files, "Checking existence of COCO images", unit="image"):
        assert f.exists()
    return coco_files


def find_lsun_samples(lsun_real_file: Path, cnndetect_dir: Path, split: str) -> list[Path]:
    assert split in ["train", "val"]
    print("Loading LSUN image paths.")
    with lsun_real_file.open() as f:
        lines: list[str] = [l.rstrip() for l in f]
    lsun_files: list[Path] = []
    for l in lines:
        l_parts: list[str] = l.split("_")
        category: str = l_parts[0]
        filename: str = l_parts[1]
        lsun_files.append(cnndetect_dir/"train"/category/"0_real"/ filename)
    print("Loading of LSUN image paths completed.")
    for f in tqdm(lsun_files, "Checking existence of LSUN images", unit="image"):
        assert f.exists()
    return lsun_files


if __name__ == "__main__":
    main()
