from typing import List, Optional


def sign_nbt(
    line1:   Optional[str] = None,
    line2:   Optional[str] = None,
    line3:   Optional[str] = None,
    line4:   Optional[str] = None,
    color:   Optional[str] = None,
):
    nbt_fields: List[str] = []

    for i, line in enumerate([line1, line2, line3, line4]):
        if line is not None:
            nbt_fields.append(f"Text{i+1}: '{{\"text\":\"{line}\"}}'")

    if color is not None:
        nbt_fields.append(f"Color: \"{color}\"")

    return ",".join(nbt_fields)
