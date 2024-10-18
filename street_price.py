with open("./raw/streets_prices.csv", "r", encoding="utf-8") as f:
    codes = f.read().split("\n")

    # split to chunks of 20
    chunk = [codes[i : i + 20] for i in range(0, len(codes), 20)]
    print(len(chunk))
    print(len(codes))
    res = []
    for i, c in enumerate(chunk):
        idx = i + 1
        header = f"""
{"{"}"name": "ราคาประเมินที่ดินในกรุงเทพมหานคร ในช่วงปีพ.ศ. 2566 ถึง ปี 2569 ({idx})"{"}"}

ราคาประเมินที่ดินในกรุงเทพมหานคร ในช่วงปีพ.ศ. 2566 ถึง ปี 2569 ({idx})
Index,Street Name,Price,Range
"""

        body = "\n".join(c)

        res.append(header + body)

    out = "\n\n---\n\n\n".join(res)
    with open("./raw/streets_prices.label.txt", "w", encoding="utf-8") as f:
        f.write(out)
