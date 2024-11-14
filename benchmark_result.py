import json
import os

files = os.listdir("./result_bench")
files.sort()
for file in files:
    with open(f"./result_bench/{file}", "r", encoding="utf8") as f:
        data = json.load(f)

        result = {
            "-1": 0,
            "0": 0,
            "1": 0,
            "unknown": 0,
        }

        for d in data:
            if d["result"] not in ("-1", "0", "1"):
                result["unknown"] += 1
            else:
                result[d["result"]] += 1
        print(f"Result of {file}:")
        print(f"Good (+1): {result['1']}")
        print(f"Neutral (0): {result['0']}")
        print(f"Bad (-1): {result['-1']}")
        if result["unknown"] != 0:
            print(f"Unknown: {result['unknown']}")
        print("\n---\n")
