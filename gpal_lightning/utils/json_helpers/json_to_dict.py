import json


def json_to_dict(file):
    with open(file, "r") as infile:
        data = json.load(infile)
    return data
