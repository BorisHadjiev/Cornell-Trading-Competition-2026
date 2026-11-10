import { execFile } from "node:child_process";

export type CommandResult = {
  file: string;
  args: string[];
  ok: boolean;
  code: number | null;
  stdout: string;
  stderr: string;
  error: string | null;
};

const MAX_STDOUT = 8000;
const MAX_STDERR = 2000;

export function runCommand(file: string, args: string[], timeoutMs = 8000): Promise<CommandResult> {
  return new Promise((resolve) => {
    execFile(
      file,
      args,
      { timeout: timeoutMs, windowsHide: true, maxBuffer: 64 * 1024 },
      (error, stdout, stderr) => {
        const err = error as NodeJS.ErrnoException | null;
        const missing = err?.code === "ENOENT";
        const code = typeof err?.code === "number" ? err.code : error ? 1 : 0;
        resolve({
          file,
          args,
          ok: error === null,
          code: error === null ? 0 : code,
          stdout: stdout.slice(0, MAX_STDOUT),
          stderr: stderr.slice(0, MAX_STDERR),
          error: error === null ? null : missing ? "not_installed" : err?.message ?? "failed",
        });
      },
    );
  });
}
