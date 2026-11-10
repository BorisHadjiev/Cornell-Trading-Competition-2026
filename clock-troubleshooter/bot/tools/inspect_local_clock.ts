import { defineTool } from "@grok/agent-sdk/tools";

import { inspectClock } from "../lib/clock.js";

export default defineTool({
  description:
    "Read this computer's clock, time zone, and time-service status. Also compare the clock with HTTP Date headers. Does not change the clock.",
  effect: "read",
  async execute() {
    return inspectClock();
  },
});
