import os
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
    os.environ[CONFIG_OPENAI_API_KEY] = default_config[CONFIG_OPENAI_API_KEY]
    os.environ[CONFIG_SERPAPI_API_KEY] = default_config[CONFIG_SERPAPI_API_KEY]