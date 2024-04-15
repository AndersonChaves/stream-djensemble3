import json

class Config():
    def __init__(self, config_file):
        self.config_file = config_file
        self.data = self.load_config_file()
        
    def load_config_file(self):
        with open(self.config_file, encoding='UTF-8') as json_file:
            return json.load(json_file)