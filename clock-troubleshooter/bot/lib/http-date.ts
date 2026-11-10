import https from "node:https";

export type HttpDateCheck = {
  host: string;
  httpDate: string | null;
  localUtc: string;
  deltaMs: number | null;
  error: string | null;
};

const REFERENCE_HOSTS = ["www.microsoft.com", "www.google.com"] as const;

export async function checkReferenceDates(timeoutMs = 5000): Promise<HttpDateCheck[]> {
  return Promise.all(REFERENCE_HOSTS.map((host) => fetchHttpDate(host, timeoutMs)));
}

function fetchHttpDate(host: string, timeoutMs: number): Promise<HttpDateCheck> {
  return new Promise((resolve) => {
    const localUtc = new Date().toISOString();
    const req = https.request(
      {
        host,
        method: "HEAD",
        path: "/",
        timeout: timeoutMs,
        headers: { "User-Agent": "clock-troubleshooter", Connection: "close" },
      },
      (res) => {
        res.resume();
        const httpDate = typeof res.headers.date === "string" ? res.headers.date : null;
        const remote = httpDate ? Date.parse(httpDate) : Number.NaN;
        const deltaMs = Number.isNaN(remote) ? null : Date.now() - remote;
        resolve({
          host,
          httpDate,
          localUtc,
          deltaMs,
          error: httpDate ? null : "missing_date_header",
        });
      },
    );
    req.on("timeout", () => {
      req.destroy();
      resolve({ host, httpDate: null, localUtc, deltaMs: null, error: "timeout" });
    });
    req.on("error", (error) => {
      resolve({ host, httpDate: null, localUtc, deltaMs: null, error: error.message });
    });
    req.end();
  });
}
