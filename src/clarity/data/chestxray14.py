import os
import hashlib
from natsort import natsorted
from pathlib import Path
import pandas as pd
from typing import Literal

from clarity.data.utils import read_file_lines

## Global constants
# list of tuples from md5 hash to gzip
MD5_FILE_HASHES = [
    (
        "fe8ed0a6961412fddcbb3603c11b3698",
        "vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    ),  # images_001.tar.gz
    (
        "ab07a2d7cbe6f65ddd97b4ed7bde10bf",
        "i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    ),  # images_002.tar.gz
    (
        "2301d03bde4c246388bad3876965d574",
        "f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    ),  # images_003.tar.gz
    (
        "9f1b7f5aae01b13f4bc8e2c44a4b8ef6",
        "0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    ),  # images_004.tar.gz
    (
        "1861f3cd0ef7734df8104f2b0309023b",
        "v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    ),  # images_005.tar.gz
    (
        "456b53a8b351afd92a35bc41444c58c8",
        "asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    ),  # images_006.tar.gz
    (
        "1075121ea20a137b87f290d6a4a5965e",
        "jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    ),  # images_007.tar.gz
    (
        "b61f34cec3aa69f295fbb593cbd9d443",
        "tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    ),  # images_008.tar.gz
    (
        "442a3caa61ae9b64e61c561294d1e183",
        "upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    ),  # images_009.tar.gz
    (
        "09ec81c4c31e32858ad8cf965c494b74",
        "l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    ),  # images_010.tar.gz
    (
        "499aefc67207a5a97692424cf5dbeed5",
        "hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    ),  # images_011.tar.gz
    (
        "dc9fda1757c2de0032b63347a7d2895c",
        "ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
    ),  # images_012.tar.gz
]

# images with issues that are ignored (eg. completely white)
IGNORED_IMAGES = [
    "00007160_002.png",
    "00010007_121.png",
]

# column information
DROPPED_COLUMNS = ["OriginalImagePixelSpacing[x", "y]", "OriginalImage[Width", "Height]"]

RENAME_COLUMN_MAPPING = {"Image Index": "Filename", "Patient ID": "PatientID"}

CXR_14_ABNORMALITIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]
CXR_14_ABNORMALITY_REGIONS = {
    "Atelectasis": ["Left Lung", "Right Lung"],
    "Cardiomegaly": ["Heart"],
    "Consolidation": ["Left Lung", "Right Lung"],
    "Edema": ["Left Lung", "Right Lung"],
    "Effusion": ["Left Lung", "Right Lung"],
    "Emphysema": ["Left Lung", "Right Lung"],
    "Fibrosis": ["Left Lung", "Right Lung"],
    "Hernia": ["Facies Diaphragmatica", "Mediastinum"],
    "Infiltration": ["Left Lung", "Right Lung"],
    "Mass": ["Left Lung", "Right Lung", "Mediastinum"],
    "Nodule": ["Left Lung", "Right Lung"],
    "Pleural Thickening": ["Left Lung", "Right Lung"],
    "Pneumonia": ["Left Lung", "Right Lung"],
    "Pneumothorax": ["Left Lung", "Right Lung"],
}
CXR_14_ABNORMALITIES_BBOX_ANNOTATED = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]


## Data management functions
def get_chestxray14_dataframe(chestxray14_path: Path) -> pd.DataFrame:
    # load csv of dataset info
    csv_path = chestxray14_path / "Data_Entry_2017_v2020.csv"
    df_cxr14 = pd.read_csv(csv_path)

    # drop and rename columns
    df_cxr14.drop(columns=DROPPED_COLUMNS, inplace=True)
    df_cxr14.rename(columns=RENAME_COLUMN_MAPPING, inplace=True)

    # add image row and column info
    df_cxr14["Rows"] = 1024
    df_cxr14["Columns"] = 1024

    # remove corrupt images
    df_cxr14 = df_cxr14[~(df_cxr14["Filename"].isin(IGNORED_IMAGES))]

    # create Labels column as lists of strings
    df_cxr14["Labels"] = df_cxr14["Finding Labels"].apply(lambda labels_str: labels_str.split("|"))
    df_cxr14["Labels"] = df_cxr14["Labels"].apply(
        lambda labels: [" ".join(label.split("_")) for label in labels if label != "No Finding"]
    )
    df_cxr14.drop(columns=["Finding Labels"], inplace=True)

    # reset index
    df_cxr14.reset_index(drop=True, inplace=True)

    return df_cxr14


def get_chestxray14_bbox_dataframe(chestxray14_path: Path, image_size: int = 1024) -> pd.DataFrame:
    bbox_list_path = chestxray14_path / "BBox_List_2017.csv"
    df_bbox = pd.read_csv(bbox_list_path)

    # rename and drop collmns
    df_bbox.drop(columns=["Unnamed: 6", "Unnamed: 7", "Unnamed: 8"], inplace=True)
    df_bbox.rename(
        columns={
            "Bbox [x": "col_start",
            "y": "row_start",
            "h]": "h",
            "Image Index": "Filename",
            "Finding Label": "Label",
        },
        inplace=True,
    )

    # convert bbox size to image size
    df_bbox["row_start"] *= image_size / 1024
    df_bbox["col_start"] *= image_size / 1024
    df_bbox["h"] = df_bbox["h"] * image_size / 1024
    df_bbox["w"] = df_bbox["w"] * image_size / 1024

    # add row/col end fields
    df_bbox["row_end"] = df_bbox["row_start"] + df_bbox["h"]
    df_bbox["col_end"] = df_bbox["col_start"] + df_bbox["w"]
    df_bbox.drop(columns=["w", "h"], inplace=True)

    # rename infiltrate to infiltration
    df_bbox.loc[df_bbox["Label"] == "Infiltrate", "Label"] = "Infiltration"

    return df_bbox


def get_chestxray14_bbox_filenames(chestxray14_path: Path) -> list[str]:
    df_bbox = get_chestxray14_bbox_dataframe(chestxray14_path)
    filenames = df_bbox["Filename"].unique().tolist()

    return filenames


def verify_md5_hashes(chestxray14_gzip_path: Path) -> None:
    """Verify the MD5 hashes of the Chest X-ray 14 gzip files

    Args:
        chestxray14_gzip_path (Path): Path to Chest X-ray 14 gzips
    """
    # sort filenames using natsorted
    md5_info = natsorted(MD5_FILE_HASHES, key=lambda x: x[1])

    for expected_md5, filename in md5_info:
        filepath = chestxray14_gzip_path / filename

        # calculate the MD5 hash of the file
        md5 = hashlib.md5()

        print(filepath, end="", flush=True)
        try:
            with open(filepath, "rb") as file:
                # read the file in chunks to avoid memory issues with large files
                for chunk in iter(lambda: file.read(4096), b""):
                    md5.update(chunk)

            # get the calculated MD5 hash
            calculated_md5 = md5.hexdigest()

            # compare the calculated MD5 hash with the expected one
            if calculated_md5 == expected_md5:
                print(" ..okay")
            else:
                print(
                    f": DIFFERENT MD5 HASH\n[Expected: {expected_md5}, Calculated: {calculated_md5}]"
                )

        except FileNotFoundError:
            print(": File not found")

        except Exception as e:
            print(f": Error occurred - {e}")


def get_official_split_filenames(
    chestxray14_path: Path, available_images_path: Path | None = None
) -> tuple[list[str], list[str]]:
    if available_images_path is not None:
        available_image_filenames = set(os.listdir(available_images_path))

    train_filenames_path = chestxray14_path / "train_val_list.txt"
    train_filenames = read_file_lines(train_filenames_path)
    train_filenames = [filename for filename in train_filenames if filename not in IGNORED_IMAGES]
    if available_images_path is not None:
        train_filenames = [
            filename for filename in train_filenames if filename in available_image_filenames
        ]

    test_filenames_path = chestxray14_path / "test_list.txt"
    test_filenames = read_file_lines(test_filenames_path)
    test_filenames = [filename for filename in test_filenames if filename not in IGNORED_IMAGES]
    if available_images_path is not None:
        test_filenames = [
            filename for filename in test_filenames if filename in available_image_filenames
        ]

    return train_filenames, test_filenames


def get_localization_split_filenames(chestxray14_path: Path) -> tuple[list[str], list[str]]:
    val_filenames_path = chestxray14_path / "localization_val_list.txt"
    val_filenames = read_file_lines(val_filenames_path)

    test_filenames_path = chestxray14_path / "localization_test_list.txt"
    test_filenames = read_file_lines(test_filenames_path)

    return val_filenames, test_filenames


def get_classification_split_filenames(
    chestxray14_path: Path,
    available_images_path: Path | None = None,
    train_size: Literal[70, 75] = 70,
) -> tuple[list[str], list[str]]:
    # get available images to filter by
    if available_images_path is not None:
        available_image_filenames = set(os.listdir(available_images_path))

    # get filename for train set size
    match train_size:
        case 70:
            val = "_70"
        case 75:
            val = "_75"

    # get training filenames
    train_filenames_path = chestxray14_path / f"classification_train_list{val}.txt"
    train_filenames = read_file_lines(train_filenames_path)
    train_filenames = [filename for filename in train_filenames if filename not in IGNORED_IMAGES]
    # filter training filenames by available images
    if available_images_path is not None:
        train_filenames = [
            filename for filename in train_filenames if filename in available_image_filenames
        ]

    # get validation filenames
    val_filenames_path = chestxray14_path / f"classification_val_list{val}.txt"
    val_filenames = read_file_lines(val_filenames_path)
    val_filenames = [filename for filename in val_filenames if filename not in IGNORED_IMAGES]
    # filter validation filenames by available images
    if available_images_path is not None:
        val_filenames = [
            filename for filename in val_filenames if filename in available_image_filenames
        ]

    return train_filenames, val_filenames
