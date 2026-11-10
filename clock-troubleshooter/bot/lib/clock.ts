import os from "node:os";

import { runCommand, type CommandResult } from "./commands.js";
import { checkReferenceDates, type HttpDateCheck } from "./http-date.js";

export type ClockInspection = {
  platform: NodeJS.Platform;
  hostname: string;
  utcNow: string;
  localIso: string;
  timeZoneOffsetMinutes: number;
  intlTimeZone: string | null;
  envTz: string | null;
  timeZoneId: string | null;
  windowsTimeService: string | null;
  windowsTimeSource: string | null;
  commands: CommandResult[];
  referenceHttpDates: HttpDateCheck[];
};

export async function inspectClock(): Promise<ClockInspection> {
  const now = new Date();
  const intl = Intl.DateTimeFormat().resolvedOptions().timeZone ?? null;
  const commands = await collectCommands();
  const referenceHttpDates = await checkReferenceDates();
  return {
    platform: process.platform,
    hostname: os.hostname(),
    utcNow: now.toISOString(),
    localIso: localIso(now),
    timeZoneOffsetMinutes: -now.getTimezoneOffset(),
    intlTimeZone: intl,
    envTz: process.env.TZ ?? null,
    timeZoneId: extractTimeZone(commands),
    windowsTimeService: extractServiceState(commands),
    windowsTimeSource: extractSource(commands),
    commands,
    referenceHttpDates,
  };
}

function localIso(now: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  const offsetMinutes = -now.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  return (
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` +
    `T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}` +
    `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`
  );
}

async function collectCommands(): Promise<CommandResult[]> {
  if (process.platform === "win32") {
    return Promise.all([
      runCommand("tzutil", ["/g"]),
      runCommand("sc", ["query", "w32time"]),
      runCommand("w32tm", ["/query", "/status"]),
      runCommand("w32tm", ["/query", "/source"]),
      runCommand("w32tm", ["/query", "/configuration"]),
    ]);
  }
  return Promise.all([
    runCommand("timedatectl", ["show"]),
    runCommand("chronyc", ["tracking"]),
  ]);
}

function commandStdout(commands: CommandResult[], file: string, includesArg?: string): string | null {
  const match = commands.find(
    (command) =>
      command.file === file &&
      command.ok &&
      (includesArg === undefined || command.args.includes(includesArg)),
  );
  const text = match?.stdout.trim();
  return text ? text : null;
}

function extractTimeZone(commands: CommandResult[]): string | null {
  const windowsZone = commandStdout(commands, "tzutil");
  if (windowsZone) return windowsZone.split(/\r?\n/)[0] ?? windowsZone;
  const show = commandStdout(commands, "timedatectl");
  const zone = show?.match(/^Timezone=(.+)$/m)?.[1];
  return zone ?? null;
}

function extractServiceState(commands: CommandResult[]): string | null {
  if (process.platform !== "win32") return null;
  const query = commands.find((command) => command.file === "sc");
  if (!query || query.error === "not_installed") return "not_installed";
  const state = query.stdout.match(/STATE\s+:\s+\d+\s+(\w+)/)?.[1];
  return state ?? (query.ok ? "unknown" : "query_failed");
}

function extractSource(commands: CommandResult[]): string | null {
  const source = commandStdout(commands, "w32tm", "/source");
  if (!source) return null;
  const line = source.split(/\r?\n/).find((item) => item.trim().length > 0);
  return line ?? null;
}
