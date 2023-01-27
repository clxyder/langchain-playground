import os
import json
from configparser import ConfigParser

from constants import (
    CONFIG_DEFAULT_KEY,
    CONFIG_OPENAI_API_KEY,
    CONFIG_SERPAPI_API_KEY,
)

def intialize_api_keys():
    # Initialize API Key
    config = ConfigParser()
    config.read("config.ini")
    default_config = config[CONFIG_DEFAULT_KEY]
    os.environ[CONFIG_OPENAI_API_KEY] = default_config.get(CONFIG_OPENAI_API_KEY, "")
    os.environ[CONFIG_SERPAPI_API_KEY] = default_config.get(CONFIG_SERPAPI_API_KEY, "")

def load_json(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath: str, payload: dict) -> None:
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)
