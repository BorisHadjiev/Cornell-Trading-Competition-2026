import { defineAgent } from "@grok/agent-sdk";

export default defineAgent({
  name: "Clock troubleshooter",
  description:
    "Read-only diagnosis of the local clock, the time zone, and Windows time server connectivity.",
  model: {
    id: "grok-4.7",
    params: [
      { id: "reasoning_effort", value: "high" },
      { id: "fast", value: "true" },
    ],
  },
  // Built-in shell and edit tools stay off. Diagnosis uses the read-only server tools.
  tools: ["read"],
});
