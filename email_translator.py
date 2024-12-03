import os

class EmailTranslator:

    def __init__(self, template_file: str, tags: dict[str, str]):
        self.template_file = template_file
        self.tags = tags
        self.lines = None
        self._read_template_file()

    def _read_template_file(self) -> None:
        if not os.path.isfile(self.template_file):
            print("ERROR: Did not find email/zero_util_gpu_hours.txt for --zero_util_gpu_hours.")
            # raise 
            return None
        else:
            with open(self.template_file, "r", encoding="utf-8") as fp:
                self.lines = fp.readlines()
            # self.lines.remove(8 * "0123456789")

    def translate(self) -> str:
        for key, value in self.tags.items():
            for i in range(len(self.lines)):
                self.lines[i] = self.lines[i].replace(key, value)
        return "".join(self.lines)
