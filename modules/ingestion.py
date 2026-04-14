import csv
import io
import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_FILE_SIZE_MB = 25
ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1"]


def _serialize_row(row):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(row)
    return buffer.getvalue().strip()


def _get_file_size_bytes(uploaded_file):
    if hasattr(uploaded_file, "size") and uploaded_file.size is not None:
        return int(uploaded_file.size)

    current_position = uploaded_file.tell()
    uploaded_file.seek(0, io.SEEK_END)
    size_bytes = uploaded_file.tell()
    uploaded_file.seek(current_position)
    return int(size_bytes)


def _read_uploaded_bytes(uploaded_file):
    uploaded_file.seek(0)
    raw_data = uploaded_file.read()

    if isinstance(raw_data, str):
        return raw_data.encode("utf-8")

    return raw_data


def _decode_csv_bytes(raw_bytes):
    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Unable to decode the uploaded CSV file: {last_error}")


def _normalize_header(header_row):
    normalized_header = []
    seen_names = {}

    for index, value in enumerate(header_row, start=1):
        column_name = str(value).strip() if value is not None else ""
        if not column_name:
            column_name = f"column_{index}"

        if column_name in seen_names:
            seen_names[column_name] += 1
            column_name = f"{column_name}_{seen_names[column_name]}"
        else:
            seen_names[column_name] = 1

        normalized_header.append(column_name)

    return normalized_header


def ingest_csv(uploaded_file, max_size_mb=DEFAULT_MAX_FILE_SIZE_MB):
    filename = getattr(uploaded_file, "name", "uploaded.csv")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only .csv files are supported.")

    file_size_bytes = _get_file_size_bytes(uploaded_file)
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    if file_size_bytes <= 0:
        raise ValueError("The uploaded CSV file is empty.")

    if file_size_bytes > max_size_bytes:
        actual_size_mb = file_size_bytes / (1024 * 1024)
        raise ValueError(
            f"File is {actual_size_mb:.1f} MB. InsightFlow currently accepts CSV files up to {max_size_mb} MB."
        )

    raw_bytes = _read_uploaded_bytes(uploaded_file)
    if not raw_bytes:
        raise ValueError("The uploaded CSV file is empty.")

    decoded_text, encoding_used = _decode_csv_bytes(raw_bytes)
    reader = csv.reader(io.StringIO(decoded_text))

    try:
        header_row = next(reader)
    except StopIteration as exc:
        raise ValueError("The uploaded CSV file is empty.") from exc

    if not header_row or not any(str(value).strip() for value in header_row):
        raise ValueError("The uploaded CSV file does not contain a valid header row.")

    normalized_header = _normalize_header(header_row)
    valid_rows = []
    repaired_row_numbers = []
    skipped_row_numbers = []
    skipped_rows = []
    blank_row_count = 0

    for line_number, row in enumerate(reader, start=2):
        if not row or not any(str(cell).strip() for cell in row):
            blank_row_count += 1
            continue

        if len(row) < len(normalized_header):
            row = row + [""] * (len(normalized_header) - len(row))
            repaired_row_numbers.append(line_number)
            valid_rows.append(row)
            continue

        if len(row) > len(normalized_header):
            skipped_row_numbers.append(line_number)
            skipped_rows.append(
                {
                    "line_number": line_number,
                    "reason": "Too many fields for the detected header",
                    "field_count": len(row),
                    "row_text": _serialize_row(row),
                }
            )
            continue

        valid_rows.append(row)

    df = pd.DataFrame(valid_rows, columns=normalized_header)
    metadata = {
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "max_size_mb": max_size_mb,
        "encoding": encoding_used,
        "blank_row_count": blank_row_count,
        "repaired_row_count": len(repaired_row_numbers),
        "repaired_row_numbers": repaired_row_numbers[:10],
        "skipped_row_count": len(skipped_row_numbers),
        "skipped_row_numbers": skipped_row_numbers[:10],
        "skipped_rows": skipped_rows,
        "skipped_rows_preview": skipped_rows[:25],
        "empty_dataset": df.empty,
    }

    LOGGER.info(
        "Ingested CSV '%s' with %s row(s), encoding=%s, repaired_rows=%s, skipped_rows=%s.",
        filename,
        len(df),
        encoding_used,
        metadata["repaired_row_count"],
        metadata["skipped_row_count"],
    )

    if skipped_row_numbers:
        LOGGER.warning(
            "Skipped %s malformed row(s) in '%s'. Sample line numbers: %s",
            len(skipped_row_numbers),
            filename,
            metadata["skipped_row_numbers"],
        )

    return df, metadata


def load_csv(uploaded_file, max_size_mb=DEFAULT_MAX_FILE_SIZE_MB):
    df, _ = ingest_csv(uploaded_file, max_size_mb=max_size_mb)
    return df
