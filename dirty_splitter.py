# open file utf-8

import io
import json


def replace_thai_number(string: str) -> str:
    string = string.replace("๐", "0")
    string = string.replace("๑", "1")
    string = string.replace("๒", "2")
    string = string.replace("๓", "3")
    string = string.replace("๔", "4")
    string = string.replace("๕", "5")
    string = string.replace("๖", "6")
    string = string.replace("๗", "7")
    string = string.replace("๘", "8")
    string = string.replace("๙", "9")
    return string


with io.open("raw/dirty.txt", "r", encoding="utf-8") as input:
    text = input.read()

    text = replace_thai_number(text)
    texts = text.split("---")
    res = []
    for t in texts:
        t = t.strip()
        if t:
            res.append(t)

    print(len(res))
    # save as json
    with io.open("raw/dirty.json", "w", encoding="utf-8") as output:
        json.dump(res, output, indent=4, ensure_ascii=False)
