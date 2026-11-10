import dgram from "node:dgram";
import dns from "node:dns/promises";

export type NtpProbe = {
  host: string;
  ok: boolean;
  address: string | null;
  family: 4 | 6 | null;
  rttMs: number | null;
  offsetMs: number | null;
  stratum: number | null;
  referenceId: string | null;
  serverUtc: string | null;
  dnsAddresses: string[];
  error: string | null;
};

const NTP_UNIX_DELTA = 22_089_888_00;
const HOST_NAME =
  /^(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)(?:\.(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?))*$/;
const IPV4 = /^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$/;
const IPV6 = /^[0-9a-fA-F:.]{2,45}$/;

export function isSafeTimeHost(host: string): boolean {
  return HOST_NAME.test(host) || IPV4.test(host) || (IPV6.test(host) && host.includes(":"));
}

export async function probeNtp(host: string, timeoutMs: number): Promise<NtpProbe> {
  const empty: NtpProbe = {
    host,
    ok: false,
    address: null,
    family: null,
    rttMs: null,
    offsetMs: null,
    stratum: null,
    referenceId: null,
    serverUtc: null,
    dnsAddresses: [],
    error: null,
  };
  if (!isSafeTimeHost(host)) {
    return { ...empty, error: "invalid_host" };
  }

  let records: Array<{ address: string; family: 4 | 6 }>;
  try {
    const lookedUp = await dns.lookup(host, { all: true, verbatim: true });
    records = lookedUp.map((entry) => ({
      address: entry.address,
      family: entry.family === 6 ? 6 : 4,
    }));
  } catch (error) {
    return { ...empty, error: error instanceof Error ? `dns_failed: ${error.message}` : "dns_failed" };
  }
  if (records.length === 0) {
    return { ...empty, error: "dns_failed: no addresses" };
  }

  const ordered = [...records].sort((a, b) => a.family - b.family);
  const errors: string[] = [];
  for (const record of ordered.slice(0, 4)) {
    const result = await queryOnce(record.address, record.family, timeoutMs);
    if (result.ok) {
      return {
        ...empty,
        ...result,
        host,
        dnsAddresses: ordered.map((item) => item.address),
      };
    }
    errors.push(`${record.address}: ${result.error ?? "failed"}`);
  }
  return {
    ...empty,
    dnsAddresses: ordered.map((item) => item.address),
    error: errors.join("; "),
  };
}

type OnceResult = {
  ok: boolean;
  address: string | null;
  family: 4 | 6 | null;
  rttMs: number | null;
  offsetMs: number | null;
  stratum: number | null;
  referenceId: string | null;
  serverUtc: string | null;
  error: string | null;
};

function queryOnce(address: string, family: 4 | 6, timeoutMs: number): Promise<OnceResult> {
  return new Promise((resolve) => {
    const socket = dgram.createSocket(family === 6 ? "udp6" : "udp4");
    const packet = Buffer.alloc(48);
    packet[0] = 0x23;
    const sentAt = Date.now();
    stampPacket(packet, sentAt);
    let settled = false;
    const finish = (result: OnceResult) => {
      if (settled) return;
      settled = true;
      socket.close();
      resolve(result);
    };
    const timer = setTimeout(() => {
      finish({
        ok: false,
        address,
        family,
        rttMs: null,
        offsetMs: null,
        stratum: null,
        referenceId: null,
        serverUtc: null,
        error: "timeout",
      });
    }, timeoutMs);

    socket.once("error", (error) => {
      clearTimeout(timer);
      finish({
        ok: false,
        address,
        family,
        rttMs: null,
        offsetMs: null,
        stratum: null,
        referenceId: null,
        serverUtc: null,
        error: error.message,
      });
    });
    socket.once("message", (message) => {
      clearTimeout(timer);
      const receivedMs = Date.now();
      if (message.length < 48) {
        finish({
          ok: false,
          address,
          family,
          rttMs: null,
          offsetMs: null,
          stratum: null,
          referenceId: null,
          serverUtc: null,
          error: "short_packet",
        });
        return;
      }
      const transmitSeconds = message.readUInt32BE(40);
      const transmitFraction = message.readUInt32BE(44);
      const serverMs = ntpToUnixMs(transmitSeconds, transmitFraction);
      const stratum = message[1] ?? 0;
      const reference = message.subarray(12, 16);
      const usable = Number.isFinite(serverMs);
      finish({
        ok: usable,
        address,
        family,
        rttMs: usable ? Math.round(receivedMs - sentAt) : null,
        offsetMs: usable ? Math.round(serverMs - (sentAt + receivedMs) / 2) : null,
        stratum: usable ? stratum : null,
        referenceId: usable ? decodeReferenceId(reference, stratum) : null,
        serverUtc: usable ? new Date(serverMs).toISOString() : null,
        error: usable ? null : "bad_timestamp",
      });
    });

    socket.send(packet, 123, address, (error) => {
      if (!error) return;
      clearTimeout(timer);
      finish({
        ok: false,
        address,
        family,
        rttMs: null,
        offsetMs: null,
        stratum: null,
        referenceId: null,
        serverUtc: null,
        error: error.message,
      });
    });
  });
}

function stampPacket(packet: Buffer, now: number): void {
  const seconds = Math.floor(now / 1000) + NTP_UNIX_DELTA;
  const fraction = Math.round(((now % 1000) / 1000) * 2 ** 32);
  packet.writeUInt32BE(seconds >>> 0, 40);
  packet.writeUInt32BE(fraction >>> 0, 44);
}

function ntpToUnixMs(seconds: number, fraction: number): number {
  if (seconds === 0) return Number.NaN;
  return (seconds - NTP_UNIX_DELTA) * 1000 + Math.round((fraction / 2 ** 32) * 1000);
}

function decodeReferenceId(reference: Buffer, stratum: number): string {
  const ascii = reference.toString("ascii").replace(/\0/g, "");
  if (stratum <= 1 || /^[\x20-\x7e]{1,4}$/.test(ascii)) return ascii;
  return reference.toString("hex");
}
