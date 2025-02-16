import io
import os
import pymupdf


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


path = "./raw/"

# get files in the directory
header_percent_map = {
    "พรบ-จัดซื้อจัดจ้าง-2560.label.pdf": 0.13,
    "ระเบียบกระทรวงการคลัง-การจัดซื้อจัดจ้าง.label.pdf": 0.13,
}


files = os.listdir(path)

for file in files:
    if not file.endswith(".label.pdf"):
        continue

    file_path = os.path.join(path, file)
    doc = pymupdf.open(file_path)
    header_percent = header_percent_map.get(file, 0)
    print(f"ignore header from {file}: {header_percent}")

    all = ""

    for page in doc:  # iterate the document pages
        rect = page.rect
        header_height = rect.height * header_percent  # Calculate header height

        # Define the area excluding the header
        text_area = pymupdf.Rect(rect.x0, rect.y0 + header_height, rect.x1, rect.y1)

        text: str = page.get_text("text", clip=text_area)  # type: ignore
        all = all + replace_thai_number(text.strip())

    with io.open(
        f"./raw/tmp.{file.replace(".pdf",".txt")}", "w", encoding="utf-8"
    ) as output:
        output.write(all)
