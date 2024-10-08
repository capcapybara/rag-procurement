import json


# Function to convert unicode escape sequences in JSON fields to actual UTF-8
def decode_unicode_in_json(json_obj):
    if isinstance(json_obj, dict):
        return {k: decode_unicode_in_json(v) for k, v in json_obj.items()}
    elif isinstance(json_obj, list):
        return [decode_unicode_in_json(i) for i in json_obj]
    elif isinstance(json_obj, str):
        return json_obj.encode("utf-8").decode("utf-8")
    else:
        return json_obj


with open("result.json", "r") as f:
    parsed_json = json.loads(f.read())

    # Convert unicode escape sequences in all fields
    decoded_json = decode_unicode_in_json(parsed_json)

    # Print the decoded JSON
    # print(json.dumps(decoded_json, ensure_ascii=False, indent=4))
    with open("result_decoded.json", "w", encoding="utf-8") as f:
        json.dump(decoded_json, f, ensure_ascii=False, indent=4)
