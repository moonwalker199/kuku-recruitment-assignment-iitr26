
import json
import re
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TXT_DIR = BASE_DIR / "data" / "raw"
JSON_DIR = BASE_DIR / "data" / "processed"

VALID_KINDS = {"dialogue", "action", "camera", "sound_fx"}

TIME_RE = re.compile(r"^\d{1,2}:\d{2}(?:\.\d+)?$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*$")
RANGE_RE = re.compile(
    r"Start:\s*\((\d{1,2}:\d{2}(?:\.\d+)?)\)\s*\|\s*"
    r"End:\s*\((\d{1,2}:\d{2}(?:\.\d+)?)\)",
    re.I
)

errors = []
warnings = []


# ============================================================
# BASIC UTILITIES
# ============================================================

def fail(filename, message):
    errors.append(f"[ERROR] {filename}: {message}")


def warn(filename, message):
    warnings.append(f"[WARNING] {filename}: {message}")


def to_seconds(value):
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + float(seconds)


def clean_body(lines):
    text = "\n".join(line.strip() for line in lines).strip()

    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]

    return text


def parse_header(lines, label):
    prefix = label.upper() + ":"

    for line in lines:
        if line.strip().upper().startswith(prefix):
            return line.split(":", 1)[1].strip()

    return None


def classify(label):
    upper = label.upper()

    if upper == "CAMERA":
        return "camera", None, None

    if upper == "ACTION":
        return "action", None, None

    if upper in ("SOUND FX", "SFX", "SOUND EFFECTS"):
        return "sound_fx", None, None

    speaker = re.sub(r"\s*\([^)]*\)", "", label).strip()
    delivery_parts = re.findall(r"\(([^)]*)\)", label)

    delivery = ", ".join(delivery_parts) if delivery_parts else None

    return "dialogue", speaker, delivery


# ============================================================
# INDEPENDENT SOURCE TXT SCANNER
# ============================================================

def scan_source(txt_path):

    lines = txt_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    result = {
        "episode": parse_header(lines, "EPISODE NUMBER"),
        "logline": parse_header(lines, "EPISODE LOGLINE"),
        "continuity_notes": parse_header(lines, "CONTINUITY NOTES"),
        "scenes": [],
        "segments": [],
        "issues": []
    }

    current_scene = None
    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if line in ("--- SCRIPT START ---", "--- SCRIPT END ---"):
            i += 1
            continue

        # Scene detection
        scene_match = SCENE_RE.match(line)

        if scene_match:

            current_scene = {
                "scene_number": int(scene_match.group(1)),
                "heading": None,
                "start_seconds": None,
                "end_seconds": None
            }

            result["scenes"].append(current_scene)

            i += 1
            continue

        # Detect malformed block-like headers
        if line.startswith("**") and line.endswith("**:"):
            if not BLOCK_RE.match(line):
                result["issues"].append(
                    f"Malformed block header at line {i + 1}"
                )

        # Scene metadata
        if current_scene is not None:

            if line.lower().startswith("scene heading:"):
                current_scene["heading"] = line.split(":", 1)[1].strip()
                i += 1
                continue

            if line.lower().startswith("scene start:"):

                raw = line.split(":", 1)[1].strip()

                if TIME_RE.fullmatch(raw):
                    current_scene["start_seconds"] = to_seconds(raw)
                else:
                    result["issues"].append(
                        f"Invalid scene start at line {i + 1}"
                    )

                i += 1
                continue

            if line.lower().startswith("scene end:"):

                raw = line.split(":", 1)[1].strip()

                if TIME_RE.fullmatch(raw):
                    current_scene["end_seconds"] = to_seconds(raw)
                else:
                    result["issues"].append(
                        f"Invalid scene end at line {i + 1}"
                    )

                i += 1
                continue

        # Segment detection
        block_match = BLOCK_RE.match(line)

        if not block_match:
            i += 1
            continue

        label = block_match.group(1).strip()
        source_line = i + 1

        j = i + 1

        while j < len(lines) and not lines[j].strip():
            j += 1

        if j >= len(lines):
            result["issues"].append(
                f"Missing timestamp at line {source_line}"
            )
            break

        range_match = RANGE_RE.search(lines[j].strip())

        if not range_match:
            result["issues"].append(
                f"Invalid timestamp at line {j + 1} "
                f"for block at line {source_line}"
            )

            i = j + 1
            continue

        start_raw, end_raw = range_match.groups()

        start_s = to_seconds(start_raw)
        end_s = to_seconds(end_raw)

        # Extract original segment body
        body_lines = []
        k = j + 1

        while k < len(lines):

            candidate = lines[k].strip()

            if BLOCK_RE.match(candidate) or SCENE_RE.match(candidate):
                break

            if candidate in (
                "--- SCRIPT START ---",
                "--- SCRIPT END ---"
            ):
                break

            body_lines.append(lines[k])
            k += 1

        kind, speaker, delivery = classify(label)

        segment = {
            "label": label,
            "kind": kind,
            "speaker": speaker,
            "delivery": delivery,
            "start": start_raw,
            "end": end_raw,
            "start_seconds": start_s,
            "end_seconds": end_s,
            "duration_seconds": round(end_s - start_s, 3),
            "text": clean_body(body_lines),
            "source_line": source_line,
            "scene_number": (
                current_scene["scene_number"]
                if current_scene else None
            )
        }

        if end_s < start_s:
            result["issues"].append(
                f"End precedes start at line {source_line}"
            )

        if not segment["text"]:
            result["issues"].append(
                f"Empty segment at line {source_line}"
            )

        result["segments"].append(segment)

        i = k

    return result


# ============================================================
# JSON VALIDATION
# ============================================================

def compare_value(filename, segment_id, field, expected, actual):

    if expected != actual:
        fail(
            filename,
            f"{segment_id}: {field} mismatch | "
            f"Expected={expected!r} | Actual={actual!r}"
        )


def validate_episode(txt_path, json_path):

    filename = txt_path.name

    try:
        source = scan_source(txt_path)

        data = json.loads(
            json_path.read_text(encoding="utf-8-sig")
        )

    except Exception as e:
        fail(filename, f"Could not read/parse file: {e}")
        return

    # --------------------------------------------------------
    # 1. Source scanning errors
    # --------------------------------------------------------

    for issue in source["issues"]:
        fail(filename, f"Source parsing: {issue}")

    # --------------------------------------------------------
    # 2. Required top-level fields
    # --------------------------------------------------------

    required_fields = [
        "source_file",
        "episode",
        "logline",
        "continuity_notes",
        "episode_duration_seconds",
        "scene_count",
        "segment_count",
        "counts_by_kind",
        "scenes",
        "segments",
        "parse_errors"
    ]

    for field in required_fields:

        if field not in data:
            fail(filename, f"Missing top-level field: {field}")

    if any(field not in data for field in required_fields):
        return

    # --------------------------------------------------------
    # 3. Metadata comparison
    # --------------------------------------------------------

    compare_value(
        filename, "METADATA", "source_file",
        txt_path.name, data["source_file"]
    )

    compare_value(
        filename, "METADATA", "episode",
        source["episode"], data["episode"]
    )

    compare_value(
        filename, "METADATA", "logline",
        source["logline"], data["logline"]
    )

    compare_value(
        filename, "METADATA", "continuity_notes",
        source["continuity_notes"], data["continuity_notes"]
    )

    # --------------------------------------------------------
    # 4. Parser-reported errors
    # --------------------------------------------------------

    if not isinstance(data["parse_errors"], list):
        fail(filename, "parse_errors must be a list")

    elif data["parse_errors"]:
        for issue in data["parse_errors"]:
            fail(filename, f"Parser reported: {issue}")

    # --------------------------------------------------------
    # 5. Scene validation
    # --------------------------------------------------------

    if not isinstance(data["scenes"], list):
        fail(filename, "scenes must be a list")
        return

    if len(data["scenes"]) != len(source["scenes"]):
        fail(
            filename,
            f"Scene count mismatch: source={len(source['scenes'])}, "
            f"JSON={len(data['scenes'])}"
        )

    if data["scene_count"] != len(source["scenes"]):
        fail(filename, "Top-level scene_count mismatch")

    for index, expected in enumerate(source["scenes"]):

        if index >= len(data["scenes"]):
            break

        actual = data["scenes"][index]

        for field in (
            "scene_number",
            "heading",
            "start_seconds",
            "end_seconds"
        ):

            compare_value(
                filename,
                f"SCENE_{expected['scene_number']}",
                field,
                expected[field],
                actual.get(field)
            )

        if actual.get("segments") is None:
            fail(filename, f"Scene {expected['scene_number']} missing segments")

    # --------------------------------------------------------
    # 6. Flatten nested JSON segments
    # --------------------------------------------------------

    flat_segments = data.get("segments")

    if not isinstance(flat_segments, list):
        fail(filename, "Top-level segments must be a list")
        return

    nested_segments = []

    for scene in data["scenes"]:
        if isinstance(scene.get("segments"), list):
            nested_segments.extend(scene["segments"])

    if nested_segments != flat_segments:
        fail(
            filename,
            "Nested scene segments do not match top-level segments"
        )

    # --------------------------------------------------------
    # 7. Segment count comparison
    # --------------------------------------------------------

    expected_segments = source["segments"]

    if len(flat_segments) != len(expected_segments):
        fail(
            filename,
            f"Segment extraction mismatch: "
            f"TXT={len(expected_segments)}, JSON={len(flat_segments)}"
        )

    if data["segment_count"] != len(expected_segments):
        fail(filename, "Top-level segment_count mismatch")

    # --------------------------------------------------------
    # 8. Compare every original TXT block with JSON
    # --------------------------------------------------------

    fields_to_compare = [
        "label",
        "kind",
        "speaker",
        "delivery",
        "start",
        "end",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "text",
        "source_line",
        "scene_number"
    ]

    for index, expected in enumerate(expected_segments):

        if index >= len(flat_segments):
            break

        actual = flat_segments[index]

        if not isinstance(actual, dict):
            fail(filename, f"Segment {index + 1} is not an object")
            continue

        segment_id = actual.get(
            "segment_id",
            f"SEGMENT_{index + 1}"
        )

        for field in fields_to_compare:

            compare_value(
                filename,
                segment_id,
                field,
                expected[field],
                actual.get(field)
            )

    # --------------------------------------------------------
    # 9. Segment IDs and required fields
    # --------------------------------------------------------

    ids = []

    for segment in flat_segments:

        if not isinstance(segment, dict):
            continue

        required_segment_fields = [
            "segment_id",
            "scene_number",
            "kind",
            "label",
            "speaker",
            "delivery",
            "start",
            "end",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "text",
            "source_line"
        ]

        for field in required_segment_fields:
            if field not in segment:
                fail(filename, f"Missing segment field: {field}")

        if "segment_id" in segment:
            ids.append(segment["segment_id"])

    if len(ids) != len(set(ids)):
        fail(filename, "Duplicate segment IDs")

    # --------------------------------------------------------
    # 10. Verify segment ID generation
    # --------------------------------------------------------

    episode_match = re.search(r"\d+", source["episode"] or "")
    episode_id = episode_match.group() if episode_match else "UNK"

    for index, segment in enumerate(flat_segments):

        if not isinstance(segment, dict):
            continue

        scene_number = segment.get("scene_number")
        expected_scene = scene_number if scene_number is not None else "UNK"

        expected_id = (
            f"E{episode_id}_S{expected_scene}_{index + 1:03d}"
        )

        compare_value(
            filename,
            f"SEGMENT_{index + 1}",
            "segment_id",
            expected_id,
            segment.get("segment_id")
        )

    # --------------------------------------------------------
    # 11. Kind/count verification
    # --------------------------------------------------------

    expected_counts = Counter(
        s["kind"] for s in expected_segments
    )

    actual_counts = Counter(
        s.get("kind") for s in flat_segments
        if isinstance(s, dict)
    )

    for kind in VALID_KINDS:

        compare_value(
            filename,
            "COUNTS",
            kind,
            expected_counts.get(kind, 0),
            actual_counts.get(kind, 0)
        )

    expected_kind_counts = {
        kind: expected_counts.get(kind, 0)
        for kind in VALID_KINDS
    }

    compare_value(
        filename,
        "COUNTS",
        "counts_by_kind",
        expected_kind_counts,
        data["counts_by_kind"]
    )

    for segment in flat_segments:

        if isinstance(segment, dict):
            if segment.get("kind") not in VALID_KINDS:
                fail(
                    filename,
                    f"Invalid segment kind: {segment.get('kind')}"
                )

    # --------------------------------------------------------
    # 12. Timestamp and duration validation
    # --------------------------------------------------------

    for segment in flat_segments:

        if not isinstance(segment, dict):
            continue

        sid = segment.get("segment_id", "UNKNOWN")

        try:
            start = segment["start_seconds"]
            end = segment["end_seconds"]
            duration = segment["duration_seconds"]

            if not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in (start, end, duration)
            ):
                raise ValueError("Timestamp values must be numeric")

            if start < 0 or end < start:
                fail(filename, f"{sid}: Invalid time range")

            if abs((end - start) - duration) > 0.001:
                fail(filename, f"{sid}: Duration calculation incorrect")

            if abs(to_seconds(segment["start"]) - start) > 0.000001:
                fail(filename, f"{sid}: Start seconds conversion incorrect")

            if abs(to_seconds(segment["end"]) - end) > 0.000001:
                fail(filename, f"{sid}: End seconds conversion incorrect")

        except (KeyError, ValueError, TypeError) as e:
            fail(filename, f"{sid}: Timestamp validation failed: {e}")

    # --------------------------------------------------------
    # 13. Episode duration verification
    # --------------------------------------------------------

    expected_duration = max(
        (s["end_seconds"] for s in expected_segments),
        default=0.0
    )

    compare_value(
        filename,
        "EPISODE",
        "episode_duration_seconds",
        expected_duration,
        data["episode_duration_seconds"]
    )

    # --------------------------------------------------------
    # 14. Final per-file status
    # --------------------------------------------------------

    print(f"Checked: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    global errors, warnings

    errors = []
    warnings = []

    print("\n========== FULL PARSER VALIDATION ==========\n")

    if not TXT_DIR.is_dir():
        print(f"FAIL: TXT directory not found: {TXT_DIR}")
        return

    if not JSON_DIR.is_dir():
        print(f"FAIL: JSON directory not found: {JSON_DIR}")
        return

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    json_files = sorted(JSON_DIR.glob("*.json"))

    print(f"TXT files: {len(txt_files)}")
    print(f"JSON files: {len(json_files)}\n")

    if not txt_files or not json_files:
        print("RESULT: FAIL - No files available for validation.")
        return

    # Match by original filename, ignoring _parsed suffix
    json_lookup = {}

    for path in json_files:

        stem = path.stem.removesuffix("_parsed")

        if stem in json_lookup:
            fail(path.name, "Duplicate JSON output for same TXT file")
        else:
            json_lookup[stem] = path

    txt_stems = {p.stem for p in txt_files}

    # Check missing outputs
    for txt in txt_files:

        output = json_lookup.get(txt.stem)

        if output is None:
            fail(txt.name, "Missing matching JSON output")
            continue

        validate_episode(txt, output)

    # Check extra JSON outputs
    for output in json_files:

        stem = output.stem.removesuffix("_parsed")

        if stem not in txt_stems:
            fail(output.name, "No matching TXT source")

    # ========================================================
    # REPORT
    # ========================================================

    print("\n========== VALIDATION REPORT ==========\n")

    print(f"TXT files found:       {len(txt_files)}")
    print(f"JSON files found:      {len(json_files)}")
    print(f"Files expected:        {len(txt_files)}")
    print(f"Files matched:         {len(txt_stems & set(json_lookup))}")
    print(f"Total errors:          {len(errors)}")
    print(f"Total warnings:        {len(warnings)}")

    if warnings:
        print("\n---------- WARNINGS ----------")
        for item in warnings:
            print(item)

    if errors:
        print("\n---------- ERRORS ----------")
        for item in errors:
            print(item)

        print("\nRESULT: FAIL")
        print("Fix the errors and run validation again.")

    else:
        print("\nRESULT: PASS")
        print("All source blocks match their JSON representations.")
        print("No structural or content mismatches detected.")


if __name__ == "__main__":
    main()
