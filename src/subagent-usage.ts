/**
 * Cumulative token usage of one subagent, read from its transcript (issue
 * #1427).
 *
 * The Agent tool's own `usage` (`AgentOutput`) covers only the subagent's
 * last API call, and the SDK stream lacks the figures to sum the calls
 * itself. Claude Code does hold every call with its final usage in the
 * subagent's transcript, `…/projects/<project>/<session>/subagents/agent-<agentId>.jsonl`,
 * so the adapter reads it when the subagent ends and reports the sum as
 * `_meta.claudeCode.subagentUsage` on the update that ends the subagent.
 *
 * The report breaks part of `_meta.quota.model_usage` down per subagent;
 * that turn total already counts subagent spend, so a client must not add
 * the two.
 */

import { readFileSync } from "node:fs";
import * as path from "node:path";

import { findTranscriptSync } from "./resumed-session.js";
import { subagentsDirectory } from "./subagent-history.js";

/** The cumulative usage of one subagent, split by model, in ACP `Usage` style. */
export type SubagentModelUsage = {
  model: string;
  inputTokens: number;
  outputTokens: number;
  /** Part of `outputTokens`, as Claude counts it. Omitted when no call of
   *  this model reported a thinking breakdown. */
  thoughtTokens?: number;
  cachedReadTokens: number;
  /** The 5m and 1h cache writes added together. The two TTLs are priced
   *  differently; `cachedWrite5mTokens`/`cachedWrite1hTokens` carry the
   *  split. */
  cachedWriteTokens: number;
  /** Omitted unless every write of this model reported the TTL split. */
  cachedWrite5mTokens?: number;
  cachedWrite1hTokens?: number;
  /** input + cache write + cache read + output. */
  totalTokens: number;
};

/** The `_meta.claudeCode.subagentUsage` value: one ended subagent's spend.
 *  A nested subagent's report covers only its own API calls. */
export type SubagentUsageReport = {
  /** The Agent/Task tool call that spawned (or resumed) the subagent. */
  toolUseId?: string;
  agentId: string;
  agentType?: string;
  models: SubagentModelUsage[];
};

/** Only ids that cannot escape the subagents directory are looked up. */
const AGENT_ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/;

/** One `usage` object as the transcript lines carry it (Anthropic Messages
 *  API shape), plus the advisor iterations some calls embed. */
type TranscriptUsage = {
  input_tokens?: unknown;
  output_tokens?: unknown;
  cache_read_input_tokens?: unknown;
  cache_creation_input_tokens?: unknown;
  cache_creation?: {
    ephemeral_5m_input_tokens?: unknown;
    ephemeral_1h_input_tokens?: unknown;
  };
  output_tokens_details?: { thinking_tokens?: unknown };
  iterations?: unknown;
};

type Tally = {
  input: number;
  output: number;
  thought: number | null;
  read: number;
  write: number;
  write5m: number | null;
  write1h: number | null;
};

function count(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function tallyUsage(models: Map<string, Tally>, model: string, usage: TranscriptUsage): void {
  let tally = models.get(model);
  if (!tally) {
    tally = { input: 0, output: 0, thought: null, read: 0, write: 0, write5m: null, write1h: null };
    models.set(model, tally);
  }
  tally.input += count(usage.input_tokens);
  tally.output += count(usage.output_tokens);
  const thinking = usage.output_tokens_details?.thinking_tokens;
  if (typeof thinking === "number" && Number.isFinite(thinking)) {
    tally.thought = (tally.thought ?? 0) + thinking;
  }
  tally.read += count(usage.cache_read_input_tokens);
  const creation = usage.cache_creation;
  if (creation && typeof creation === "object") {
    const five = count(creation.ephemeral_5m_input_tokens);
    const hour = count(creation.ephemeral_1h_input_tokens);
    tally.write += five + hour;
    tally.write5m = (tally.write5m ?? 0) + five;
    tally.write1h = (tally.write1h ?? 0) + hour;
  } else {
    // A flat write leaves the TTL split of this model unknown: a partial
    // split would let a client misprice the unsplit rest.
    tally.write += count(usage.cache_creation_input_tokens);
    tally.write5m = null;
    tally.write1h = null;
  }
}

/**
 * Sums every assistant call of a subagent transcript, split by model. An
 * assistant record's top-level `usage` is that call alone; its advisor
 * iterations are further calls it made, which the top-level figures do not
 * include (#914) and which carry their own model when the subagent switched
 * models mid-call. Malformed lines are skipped, like any other transcript
 * reader here does.
 */
export function aggregateSubagentUsage(lines: string): SubagentModelUsage[] {
  const models = new Map<string, Tally>();
  for (const line of lines.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let record: unknown;
    try {
      record = JSON.parse(trimmed);
    } catch {
      continue;
    }
    const message = (record as { message?: unknown } | null)?.message as
      { role?: unknown; model?: unknown; usage?: unknown } | undefined;
    if (!message || message.role !== "assistant") continue;
    const usage = message.usage;
    if (!usage || typeof usage !== "object") continue;
    const model = text(message.model) ?? "unknown";
    tallyUsage(models, model, usage as TranscriptUsage);
    const iterations = (usage as TranscriptUsage).iterations;
    if (!Array.isArray(iterations)) continue;
    for (const iteration of iterations) {
      if (!iteration || typeof iteration !== "object") continue;
      const entry = iteration as TranscriptUsage & { model?: unknown };
      tallyUsage(models, text(entry.model) ?? model, entry);
    }
  }
  return [...models].map(([model, tally]) => ({
    model,
    inputTokens: tally.input,
    outputTokens: tally.output,
    ...(tally.thought !== null ? { thoughtTokens: tally.thought } : {}),
    cachedReadTokens: tally.read,
    cachedWriteTokens: tally.write,
    ...(tally.write5m !== null ? { cachedWrite5mTokens: tally.write5m } : {}),
    ...(tally.write1h !== null ? { cachedWrite1hTokens: tally.write1h } : {}),
    totalTokens: tally.input + tally.write + tally.read + tally.output,
  }));
}

/**
 * The cumulative usage of the subagent `agentId` of session `sessionId`,
 * from its transcript. Null when there is no transcript, no readable
 * subagent transcript, or no usage in it — the field is optional and a
 * missing source reports nothing rather than a wrong total.
 *
 * Synchronous, and cheap: a directory listing and a stat per project
 * directory, then one small JSONL read, once per subagent end. Its callers
 * sit on the message pump and must not change its await structure.
 */
export function readSubagentUsage(options: {
  sessionId: string;
  agentId: string;
  toolUseId?: string;
  agentType?: string;
}): SubagentUsageReport | null {
  const { sessionId, agentId } = options;
  if (!sessionId || !AGENT_ID_PATTERN.test(agentId)) return null;
  const transcript = findTranscriptSync(sessionId);
  if (!transcript) return null;
  let lines: string;
  try {
    lines = readFileSync(
      path.join(subagentsDirectory(transcript, sessionId), `agent-${agentId}.jsonl`),
      "utf8",
    );
  } catch {
    return null;
  }
  const models = aggregateSubagentUsage(lines);
  if (models.length === 0) return null;
  const agentType = text(options.agentType);
  return {
    ...(options.toolUseId ? { toolUseId: options.toolUseId } : {}),
    agentId,
    ...(agentType ? { agentType } : {}),
    models,
  };
}
