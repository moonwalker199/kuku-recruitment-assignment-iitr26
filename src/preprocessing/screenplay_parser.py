
import argparse
import json
import re
from pathlib import Path

TIME_RE = re.compile(r"^\d{1,2}:\d{2}(?:\.\d+)?$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*$")
RANGE_RE = re.compile(
    r"Start:\s*\((\d{1,2}:\d{2}(?:\.\d+)?)\)\s*\|\s*"
    r"End:\s*\((\d{1,2}:\d{2}(?:\.\d+)?)\)",
    re.I
)


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


def parse_script(path):

    lines = path.read_text(encoding="utf-8-sig").splitlines()

    episode = parse_header(lines, "EPISODE NUMBER")
    logline = parse_header(lines, "EPISODE LOGLINE")
    continuity = parse_header(lines, "CONTINUITY NOTES")

    episode_match = re.search(r"\d+", episode or "")
    episode_id = episode_match.group() if episode_match else "UNK"

    scenes = []
    segments = []
    errors = []

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
                "end_seconds": None,
                "segments": []
            }

            scenes.append(current_scene)
            i += 1
            continue

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
                    errors.append(f"Invalid scene start at line {i + 1}")

                i += 1
                continue

            if line.lower().startswith("scene end:"):
                raw = line.split(":", 1)[1].strip()

                if TIME_RE.fullmatch(raw):
                    current_scene["end_seconds"] = to_seconds(raw)
                else:
                    errors.append(f"Invalid scene end at line {i + 1}")

                i += 1
                continue

        # Segment detection
        block_match = BLOCK_RE.match(line)

        if not block_match:
            i += 1
            continue

        label = block_match.group(1).strip()
        header_line = i + 1

        j = i + 1

        while j < len(lines) and not lines[j].strip():
            j += 1

        if j >= len(lines):
            errors.append(f"Missing timestamp at line {header_line}")
            break

        range_match = RANGE_RE.search(lines[j].strip())

        if not range_match:
            errors.append(f"Invalid timestamp at line {j + 1}")
            i = j + 1
            continue

        start_raw, end_raw = range_match.groups()

        start_s = to_seconds(start_raw)
        end_s = to_seconds(end_raw)

        # Extract segment body
        body_lines = []
        k = j + 1

        while k < len(lines):

            candidate = lines[k].strip()

            if BLOCK_RE.match(candidate) or SCENE_RE.match(candidate):
                break

            if candidate in ("--- SCRIPT START ---", "--- SCRIPT END ---"):
                break

            body_lines.append(lines[k])
            k += 1

        # Classify segment
        upper_label = label.upper()

        if upper_label == "CAMERA":
            kind = "camera"
            speaker = None
            delivery = None

        elif upper_label == "ACTION":
            kind = "action"
            speaker = None
            delivery = None

        elif upper_label in ("SOUND FX", "SFX", "SOUND EFFECTS"):
            kind = "sound_fx"
            speaker = None
            delivery = None

        else:
            kind = "dialogue"
            speaker = re.sub(r"\s*\([^)]*\)", "", label).strip()

            delivery_parts = re.findall(r"\(([^)]*)\)", label)

            delivery = ", ".join(delivery_parts) if delivery_parts else None

        scene_number = (
            current_scene["scene_number"]
            if current_scene else None
        )

        segment = {
            "segment_id": (
                f"E{episode_id}_S{scene_number or 'UNK'}_"
                f"{len(segments) + 1:03d}"
            ),
            "scene_number": scene_number,
            "kind": kind,
            "label": label,
            "speaker": speaker,
            "delivery": delivery,
            "start": start_raw,
            "end": end_raw,
            "start_seconds": start_s,
            "end_seconds": end_s,
            "duration_seconds": round(end_s - start_s, 3),
            "text": clean_body(body_lines),
            "source_line": header_line
        }

        if end_s < start_s:
            errors.append(f"End precedes start at line {header_line}")

        if not segment["text"]:
            errors.append(f"Empty text at line {header_line}")

        segments.append(segment)

        if current_scene is not None:
            current_scene["segments"].append(segment)

        i = k

    # Scene validation
    for scene in scenes:

        start = scene["start_seconds"]
        end = scene["end_seconds"]

        if start is not None and end is not None and end < start:
            errors.append(
                f"Scene {scene['scene_number']} ends before it starts"
            )

    # Episode duration
    duration = max(
        (s["end_seconds"] for s in segments),
        default=0.0
    )

    return {
        "source_file": path.name,
        "episode": episode,
        "logline": logline,
        "continuity_notes": continuity,
        "episode_duration_seconds": duration,
        "scene_count": len(scenes),
        "segment_count": len(segments),
        "counts_by_kind": {
            kind: sum(1 for s in segments if s["kind"] == kind)
            for kind in ("dialogue", "action", "camera", "sound_fx")
        },
        "scenes": scenes,
        "segments": segments,
        "parse_errors": errors
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="parsed")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_dir():
        raise SystemExit(f"Directory not found: {input_path}")

    files = sorted(input_path.glob("*.txt"))

    if not files:
        raise SystemExit("No TXT files found in input directory.")

    output_path.mkdir(parents=True, exist_ok=True)

    summary = []

    for file_path in files:

        try:
            result = parse_script(file_path)

            destination = output_path / f"{file_path.stem}_parsed.json"

            destination.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            summary.append({
                "file": file_path.name,
                "scenes": result["scene_count"],
                "segments": result["segment_count"],
                "errors": len(result["parse_errors"]),
                "output": destination.name
            })

        except Exception as error:

            summary.append({
                "file": file_path.name,
                "error": str(error)
            })

    print(json.dumps({
        "files_found": len(files),
        "files_processed": sum("error" not in x for x in summary),
        "results": summary
    }, indent=2))


if __name__ == "__main__":
    main()
