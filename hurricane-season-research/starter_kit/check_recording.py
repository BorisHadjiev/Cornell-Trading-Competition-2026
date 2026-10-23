"""Compare a testnet recording with the research data. The check shows if your bot got the feed
exactly as the exchange published it.

    python check_recording.py recording.jsonl ../history/

Each news message from the testnet is also a line in the `feed.jsonl` of one research season. This
script finds the season of each recorded message from its date. It ignores snapshots, because they
have no `seq`. It also ignores duplicate messages, because the exchange can send a message two times
after a retry. For each season, it reports the missing, different and unexpected messages. If you
started to record during a season, the check starts at the first message that you received.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from history import load


def compare(recorded: list[dict[str, Any]], published: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the recorded news messages of one season with its published feed, by `seq`."""
    want = {m["seq"]: m for m in published}
    got: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for m in recorded:
        if m["seq"] in got:
            duplicates += 1
        got.setdefault(m["seq"], m)
    seen = [s for s in got if s in want]
    return {
        "published": len(want),
        "received": len(got),
        "duplicates": duplicates,
        "missing": sorted(set(want) - set(got)),
        "different": sorted(s for s in seen if got[s] != want[s]),
        "unexpected": sorted(set(got) - set(want)),
        "first": min(seen, default=None),
    }


def news(path: Path) -> list[dict[str, Any]]:
    """Get the news messages that have a `seq` from a recording of record_feed.py."""
    out = []
    with path.open() as f:
        for line in f:
            frame = json.loads(line)
            message = frame.get("message")
            if frame.get("type") == "news" and isinstance(message, dict) and "seq" in message:
                out.append(message)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("history", type=Path, help="the research data folder, with manifest.json")
    args = parser.parse_args()
    data = load(args.history)
    by_season: dict[int, list[dict[str, Any]]] = {}
    for m in news(args.recording):
        season = data.season_of(m["time"][:10])
        if season is None:
            print(f"seq {m['seq']} at {m['time']} matches no research season", file=sys.stderr)
            continue
        by_season.setdefault(season.number, []).append(m)
    ok = True
    for number, recorded in sorted(by_season.items()):
        season = data.seasons[number - 1]
        r = compare(recorded, season.feed)
        late = r["first"] or 0
        missing = [s for s in r["missing"] if s >= late]
        print(f"season {number:3d} ({season.folder.name}): received {r['received']} of {r['published']}"
              f"{f', joined at seq {late}' if late else ''}; missing {len(missing)}, different {len(r['different'])},"
              f" unexpected {len(r['unexpected'])}, duplicates dropped {r['duplicates']}")
        for label, seqs in (("missing", missing), ("different", r["different"]), ("unexpected", r["unexpected"])):
            if seqs:
                print(f"    {label}: seq {', '.join(map(str, seqs[:20]))}{' ...' if len(seqs) > 20 else ''}")
        ok = ok and not (missing or r["different"] or r["unexpected"])
    print("recording matches the published feed" if ok and by_season else "differences found" if by_season
          else "no news messages in the recording")
    return 0 if ok and by_season else 1


if __name__ == "__main__":
    raise SystemExit(main())
