from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetIdentifier:
    """This dataclass is used to store the unique information of a dataset,"""

    camera_name: str
    root_dir: str
    dataset_name: str
    sql_filter: str = ""
    dataset_idx: int = -1
    data_len: int = -1
