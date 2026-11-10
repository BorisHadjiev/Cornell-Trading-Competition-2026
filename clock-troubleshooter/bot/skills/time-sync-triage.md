---
description: Use when choosing a clock verdict from inspect_local_clock and probe_time_server results.
---

# Time sync triage

Use this order. Stop at the first match.

1. No address for the time server: `dns_failed`.
2. `time.windows.com` did not answer, and `time.nist.gov` did: `windows_time_unreachable`.
3. No NTP server answered, and DNS returned an address: `udp_123_blocked`.
4. Windows Time service state is stopped: `time_service_stopped`. Mention this together with the NTP result when both are true. Prefer `time_service_stopped` when NTP also failed.
5. Best absolute NTP or HTTP delta is 2 seconds or more: `clock_offset`.
6. Best absolute NTP offset is under 2 seconds, and the reported zone is not the zone the user expects: `clock_ok_zone_display`. A UTC clock looks five hours ahead of New York from November through early March, and four hours ahead during daylight saving time.
7. Otherwise: `insufficient_evidence`.

A stratum value, a reference id, and a round-trip time show that the server answered. They are not by themselves a clock error.

Do not recommend a clock change from this skill. Report the verdict and the evidence.
