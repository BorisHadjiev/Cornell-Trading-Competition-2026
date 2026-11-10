import { defineEval, includes } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["smoke"],
  description: "A wrong-clock report reads the local clock and probes the Windows time server.",
  async test(t) {
    await t.send(
      "My computer shows the wrong time and cannot connect to the Windows time server. Find out why. Do not change the clock.",
    );
    t.succeeded();
    t.calledTool("inspect_local_clock");
    t.calledTool("probe_time_server", { input: { host: "time.windows.com" } });
    t.check(t.reply, includes(/Verdict:/));
  },
});
