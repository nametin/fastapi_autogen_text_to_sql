class Helpers:
    def __init__(self):
        pass

    # def write_txt(filename: str, data: str) -> None:
    #     with open(filename, "w") as file:
    #         file.write(data)

    def get_sql_part_from_str(text):
        start = text.find("```") + 3
        text = text[start:]
        end = text.find("```")
        return text[:end].strip()
