import os
from typing import Dict

class EmailTranslator:

    def __init__(self, template_file: str, tags: Dict[str, str]) -> None:
        self.template_file = template_file
        self.tags = tags
        self.lines = None
        self._read_template_file()

    def _read_template_file(self) -> None:
        full_path = os.path.join(os.path.dirname(__file__), self.template_file)
        print(full_path)
        #if not os.path.isfile(self.template_file):
        if not os.path.isfile(full_path):
            print(f"ERROR: Did not find {self.template_file}.")
            # raise
            return None
        else:
            with open(full_path, "r", encoding="utf-8") as fp:
            #with open(self.template_file, "r", encoding="utf-8") as fp:
                self.lines = fp.readlines()

    def replace_tags(self) -> str:
        for key, value in self.tags.items():
            for i in range(len(self.lines)):
                self.lines[i] = self.lines[i].replace(key, value)
        return "".join(self.lines)
