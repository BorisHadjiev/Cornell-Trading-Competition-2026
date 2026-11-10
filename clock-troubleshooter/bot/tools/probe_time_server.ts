import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

import { probeNtp } from "../lib/ntp.js";

export default defineTool({
  description:
    "Send one NTP query to a time server and report reachability, stratum, and clock offset in milliseconds. Defaults to time.windows.com. Does not change the clock.",
  effect: "read",
  inputSchema: z.object({
    host: z
      .string()
      .min(1)
      .max(253)
      .default("time.windows.com")
      .describe("NTP host name or IP address. Use time.windows.com unless a second opinion is needed."),
    timeoutMs: z.number().int().min(500).max(10000).default(4000),
  }),
  async execute({ host, timeoutMs }) {
    return probeNtp(host, timeoutMs);
  },
});
