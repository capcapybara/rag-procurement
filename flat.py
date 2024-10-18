import io
import json

with io.open("raw/cleaned.json", "r", encoding="utf-8") as input:
    data = json.load(input)

    res = "\n\n---\n\n\n".join(data)

    with io.open("raw/cleaned.txt", "w", encoding="utf-8") as output:
        output.write(res)
