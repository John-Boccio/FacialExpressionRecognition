import json


class ConfigParser:
    __config = None

    @staticmethod
    def get_config():
        if ConfigParser.__config is None:
            ConfigParser()
        return ConfigParser.__config

    def __init__(self):
        # If there already exists an instance of __config, something went wrong
        if ConfigParser.__config is not None:
            raise Exception("Improper usage of the singleton class ConfigParser. "
                            "Use ConfigParser.getConfig to get the config.json as a dict")
        # Create the instance of __config
        # This path must remain the same across all projects, do not change without moving the config.json file
        with open('config.json') as config_file:
            ConfigParser.__config = json.load(config_file)
