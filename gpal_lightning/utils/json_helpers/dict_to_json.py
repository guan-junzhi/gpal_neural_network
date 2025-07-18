import json
from typing import Union

from gpal_lightning.utils.json_helpers.json_serializer import JsonSerializer


def dict_to_json(path: str, dict_object: Union[dict, list], indent: int = None) -> None:
    with open(path, "w") as outfile:
        json.dump(dict_object, outfile, cls=JsonSerializer, indent=indent)
