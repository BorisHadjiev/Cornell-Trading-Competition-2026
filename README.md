# Cornell Trading Competition 2026

Systematic Equities case: one hurricane season on the Archipelago Exchange (10 stocks and 2 ETFs).

The research pack is a Google Drive zip of about 34MB (about 209MB uncompressed). It is not in git.

```bash
python3 scripts/fetch_research_data.py
```

That writes `hurricane-season-research/`. These paths are committed because they are source, not dumps:

- `hurricane-season-research/case.json` — map, sites, companies, ETF baskets, limits, outlooks, timing
- `hurricane-season-research/starter_kit/` — data loader, feed recorder, recording check, example bot
- `hurricane-season-research/README.md` — data dictionary

`hurricane-season-research/history/` (600 seasons) and the zip stay local. See the data dictionary after fetch. The starter kit expects Python 3.10+. `record_feed.py` and `example_bot.py` need `websockets>=14`. `history.py` uses the standard library only.
