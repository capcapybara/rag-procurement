import json
import os


def add(map, type, data):
    if type not in map:
        map[type] = {
            "-1": 0,
            "0": 0,
            "1": 0,
            "unknown": 0,
        }

    if data["result"] not in ("-1", "0", "1"):
        map[type]["unknown"] += 1
    else:
        map[type][data["result"]] += 1


files = os.listdir("./result_bench")
files.sort()
for file in files:
    with open(f"./result_bench/{file}", "r", encoding="utf8") as f:
        data = json.load(f)

        map = {}

        for d in data:
            add(map, "normal", d)
        print(f"Result of {file}:")
        for type, result in map.items():
            print(f"LEVEL: {type}")
            print(f"Good (+1): {result['1']}")
            print(f"Neutral (0): {result['0']}")
            print(f"Bad (-1): {result['-1']}")
            if result["unknown"] != 0:
                print(f"Unknown: {result['unknown']}")
            print("---\n")

        print("\n------\n")
