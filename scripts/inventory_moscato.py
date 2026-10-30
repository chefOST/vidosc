"""Inventory local MOSCATO metadata; never download or launch compute."""
import collections
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "data/moscato"
    annotations = json.loads((data / "annotations/gt_annotations_cmu.json").read_text())
    files = json.loads((data / "vocabulary/files_dict.json").read_text())
    mapping = json.loads((ROOT / "configs/changeit_moscato_alias_mapping.json").read_text())
    supported = {s for values in mapping["source_index_to_states"].values() for s in values}
    rows = []
    for video_id, annotation in sorted(annotations.items()):
        source = files.get(video_id, {}).get("video_path")
        video = data / "videos" / f"{video_id}.avi"
        objects = []
        for name, timeline in annotation["state"].items():
            labels = sorted({s for frame in timeline for s in frame})
            objects.append({"object": name, "annotation_count": len(timeline),
                "nonempty_frames": sum(bool(f) for f in timeline), "states": labels,
                "mapped_states": sorted(set(labels) & supported),
                "unmapped_states": sorted(set(labels) - supported)})
        row = {"video_id": video_id, "activity": video_id.split("_")[1],
            "local_video": str(video), "available": video.is_file(),
            "annotation_lengths_agree": len({o["annotation_count"] for o in objects}) == 1,
            "objects": objects, "original_video_path": source,
            "candidate_archive_url": (f"http://kitchen.cs.cmu.edu/Main/{Path(source).parent.name}.zip" if source else None),
            "archive_url_verified": False, "duration_seconds": None,
            "timing_verified": False}
        if video.is_file():
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error",
                "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,nb_frames:format=duration",
                "-of", "json", str(video)], text=True))
            stream = probe["streams"][0]
            n, d = map(float, stream["avg_frame_rate"].split("/"))
            row.update(duration_seconds=float(probe["format"]["duration"]), fps=n/d,
                       metadata_frame_count=int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else None)
        rows.append(row)
    result = {"videos": rows, "video_count": len(rows),
        "available_videos": sum(r["available"] for r in rows),
        "activities": dict(collections.Counter(r["activity"] for r in rows)),
        "note": "Candidate archive URLs are inferred from released file paths, not checked. Missing-video durations remain unknown. Annotation counts are not verified durations."}
    out = ROOT / "runs/batch-preparation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# MOSCATO inventory", "", result["note"], "",
        "| Video | Local | Duration (s) | Objects: mapped/total distinct states |", "|---|---|---|---|"]
    for row in rows:
        summary = "; ".join(f"{o['object']}: {len(o['mapped_states'])}/{len(o['states'])}" for o in row["objects"])
        lines.append(f"| {row['video_id']} | {row['available']} | {row['duration_seconds'] or 'unknown'} | {summary} |")
    (ROOT / "docs/moscato_inventory.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "videos"}, indent=2))


if __name__ == "__main__":
    main()
