import json

class Utils:
    def write_json(self,file_name,L):
        with open(file_name, mode="w") as f:
            d = json.dumps(L)
            f.write(d)

    def load_json(self,file_name):
        """
        this function reads prompts as a list
        """
        d = []
        with open(file_name, mode="r") as f:
            d = json.load(f)
        return d