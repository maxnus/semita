from collections.abc import Buffer

def find_path(
    passable: Buffer, start_i: int, start_j: int, end_i: int, end_j: int, costs: Buffer | None, /
) -> tuple[float, float, bytes | None]: ...
def distances_from(
    passable: Buffer, start_i: int, start_j: int, distances: Buffer, costs: Buffer | None, /
) -> None: ...
