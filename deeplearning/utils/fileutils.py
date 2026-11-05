import typing
import csv
import io
import zlib
import os


class Compression:
    @staticmethod
    def decompress(data: bytes) -> bytes:
        decompressed = zlib.decompressobj()
        inflated = decompressed.decompress(data)
        inflated += decompressed.flush()
        return inflated

    @staticmethod
    def compress(data: bytes) -> bytes:
        compressed = zlib.compressobj()
        deflated = compressed.compress(data)
        deflated += compressed.flush()
        return deflated


class CsvIO:
    @staticmethod
    def to_records(csv_string: str) -> typing.List[dict]:
        reader = csv.DictReader(io.StringIO(csv_string))
        return list(reader)

    @staticmethod
    def to_csv_string(records: typing.List[dict], with_header: bool = True) -> str:
        if not records:
            return ""
        writer = csv.DictWriter((output := io.StringIO()), fieldnames=records[0].keys())
        if with_header:
            writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

    @staticmethod
    def read_from_csvfile(filepath: str, to_records: bool = True) -> typing.List[dict] | str:
        with open(filepath, 'r', encoding="utf-8") as f:
            csv_string = f.read()
        if to_records:
            return CsvIO.to_records(csv_string)
        return csv_string

    @staticmethod
    def write_to_csvfile(filepath: str, records: typing.List[dict], append: bool = False):
        if append:
            csv_string = CsvIO.to_csv_string(records, False)
            with open(filepath, 'a', encoding="utf-8") as f:
                f.write(csv_string)
            return

        csv_string = CsvIO.to_csv_string(records)
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(csv_string)

    @staticmethod
    def read_from_zlib_csvfile(filepath: str, to_records: bool = True) -> typing.List[dict] | str:
        with open(filepath, 'rb') as f:
            csv_string = Compression.decompress(f.read()).decode("utf-8")
        if to_records:
            return CsvIO.to_records(csv_string)
        return csv_string

    @staticmethod
    def write_to_zlib_csvfile(filepath: str, records: typing.List[dict]):
        csv_string = CsvIO.to_csv_string(records)
        with open(filepath, 'wb') as f:
            f.write(Compression.compress(csv_string.encode("utf-8")))


def flattened_directory(path: str) -> typing.List[str]:
    if not os.path.isdir(path):
        return [path]
    filepaths = []
    for sub_path in os.listdir(path):
        filepaths.extend(flattened_directory(os.path.join(path, sub_path)))
    return filepaths


def to_readable_size(byte_size: float, n_digits: int = 0) -> str | None:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if byte_size < 1024:
            return f"{byte_size:.{n_digits}f} {unit}"
        byte_size /= 1024
    return None


def to_byte_size(readable_size: str) -> int | None:
    size, unit = readable_size.split(" ")
    try:
        index = ("B", "KB", "MB", "GB", "TB", "PB").index(unit)
    except ValueError:
        return None
    try:
        return int(size) * 1024 ** index
    except ValueError:
        return None


if __name__ == '__main__':
    ...
