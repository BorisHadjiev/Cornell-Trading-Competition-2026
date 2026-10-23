"""A simple Hurricane Season bot. It shows the full loop. It is not a strategy.

    pip install "websockets>=14"
    python example_bot.py --url https://hurricane.cornelltradingcompetition.org --key BOT_KEY [--dry-run]

The bot reads the public case data, follows the competition clock, and reads the feed and the
quotes. An official advisory can give an island a probability of hurricane conditions of `--odds`
or more. When this occurs, the bot sells a small quantity of each island business that has a site
on that island. It sells each company one time in each storm, with a limit order at the bid. It
does not send market orders. Replace `decide` with your own model. You can use the other parts
(connection, clock, quotes and orders) again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from typing import Any

import websockets

# the network in front of the exchange refuses the default urllib user agent
HEADERS = {"User-Agent": "hurricane-starter-kit/1.0"}


def decide(advisory: dict[str, Any], case: dict[str, Any], sold: set[str], odds: float) -> list[str]:
    """Symbols to sell: the island businesses on islands at `odds` or more in the advisory."""
    hit = {island for island, p in advisory.get("hurricane_conditions", {}).items() if p >= odds}
    return [sym for sym, c in case["companies"].items()
            if c["kind"] == "physical" and sym not in sold and any(s["island"] in hit for s in c["sites"])]


class Bot:
    """The connection, clock, quotes and orders around `decide`."""

    def __init__(self, url: str, key: str, clip: float, odds: float, dry_run: bool) -> None:
        self.url, self.key, self.clip, self.odds, self.dry_run = url.rstrip("/"), key, clip, odds, dry_run
        self.case = self.get("/cases/hurricane.json", api=False)
        self.quotes: dict[str, dict[str, Any]] = {}
        self.sold: set[str] = set()
        self.storm = ""

    def get(self, path: str, api: bool = True) -> Any:
        req = urllib.request.Request(self.url + ("/api/v1" if api else "") + path,
                                     headers={**HEADERS, "X-API-Key": self.key})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)

    def sell(self, sym: str) -> None:
        """Send a limit sell order at the current bid, for `clip` of the position limit."""
        bid = (self.quotes.get(sym) or {}).get("bid")
        if bid is None:
            print(f"no bid for {sym}; no order")
            return
        qty = max(1, round(self.clip * self.case["limits"][sym]))
        order = {"symbol": sym, "side": "sell", "order_type": "limit", "quantity": qty, "price": bid}
        if self.dry_run:
            print("would send", order)
            return
        req = urllib.request.Request(self.url + "/api/v1/orders", data=json.dumps(order).encode(), method="POST",
                                     headers={**HEADERS, "X-API-Key": self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print("sent", order, "->", json.load(resp).get("status"))
        except urllib.error.HTTPError as exc:
            print("rejected", order, exc.code, exc.read().decode()[:200])

    def on_news(self, m: dict[str, Any]) -> None:
        if m.get("storm") and m["storm"] != self.storm:
            self.storm, self.sold = m["storm"], set()
            print(f"new storm: {self.storm}")
        if m.get("type") == "advisory" and m.get("source") == "official":
            if self.get("/competitions/time").get("state") != "running":
                return
            for sym in decide(m, self.case, self.sold, self.odds):
                self.sold.add(sym)
                self.sell(sym)

    async def run(self) -> None:
        subscribe = {"action": "subscribe", "symbols": list(self.case["limits"]), "channels": ["news", "quotes"]}
        ws_url = self.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/v1/market-data"
        while True:
            try:
                async with websockets.connect(ws_url, additional_headers={"X-API-Key": self.key}) as ws:
                    await ws.send(json.dumps(subscribe))
                    async for frame in ws:
                        data = json.loads(frame)
                        if data["type"] == "quote":
                            self.quotes[data["symbol"]] = data
                        elif data["type"] == "news" and isinstance(data.get("message"), dict):
                            await asyncio.to_thread(self.on_news, data["message"])
            except (OSError, websockets.WebSocketException) as exc:
                print(f"disconnected ({exc}); connecting again")
                await asyncio.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the address of the exchange, for example https://hurricane.cornelltradingcompetition.org")
    parser.add_argument("--key", required=True, help="a bot key from your Team page")
    parser.add_argument("--clip", type=float, default=0.1, help="order size, as a fraction of the position limit")
    parser.add_argument("--odds", type=float, default=0.8, help="probability of hurricane conditions that starts a sale")
    parser.add_argument("--dry-run", action="store_true", help="show the orders, but do not send them")
    args = parser.parse_args()
    asyncio.run(Bot(args.url, args.key, args.clip, args.odds, args.dry_run).run())


if __name__ == "__main__":
    main()
