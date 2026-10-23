"""Record the market-data stream of the exchange to a file, with one JSON frame on each line.

    pip install "websockets>=14"
    python record_feed.py --url wss://hurricane.cornelltradingcompetition.org --key BOT_KEY --out recording.jsonl
    python record_feed.py --url wss://hurricane.cornelltradingcompetition.org --key BOT_KEY --symbols PORT,HAVEN

The recorder subscribes to the `news` channel (the storm feed). With `--symbols`, it also
subscribes to the quotes and trades of those symbols. Each line is the frame as received, plus
`"received_at"` (unix seconds). If the connection stops, the recorder connects again after a delay.
The exchange does not send old news again. A gap in the connection shows as missing `seq` numbers,
and `check_recording.py` reports them. Push Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time

import websockets


async def record(url: str, key: str, out: str, symbols: list[str]) -> None:
    """Write each frame to `out` until cancelled. Connect again when the connection stops."""
    channels = ["news", "quotes", "trades"] if symbols else ["news"]
    subscribe = {"action": "subscribe", "symbols": symbols, "channels": channels}
    backoff = 1.0
    with open(out, "a") as f:
        while True:
            try:
                async with websockets.connect(f"{url.rstrip('/')}/ws/v1/market-data",
                                              additional_headers={"X-API-Key": key}) as ws:
                    await ws.send(json.dumps(subscribe))
                    print("connected", flush=True)
                    backoff = 1.0
                    async for frame in ws:
                        f.write(json.dumps({**json.loads(frame), "received_at": time.time()}) + "\n")
                        f.flush()
            except (OSError, websockets.WebSocketException) as exc:
                print(f"disconnected ({exc}); next try in {backoff:.0f} s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the address of the exchange, for example wss://hurricane.cornelltradingcompetition.org")
    parser.add_argument("--key", required=True, help="a bot key from your Team page")
    parser.add_argument("--out", default="recording.jsonl")
    parser.add_argument("--symbols", default="", help="symbols for quotes and trades, with commas between them")
    args = parser.parse_args()
    symbols = [s for s in args.symbols.split(",") if s]
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(record(args.url, args.key, args.out, symbols))


if __name__ == "__main__":
    main()
