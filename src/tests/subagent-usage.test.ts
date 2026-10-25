import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";

import { aggregateSubagentUsage, readSubagentUsage } from "../subagent-usage.js";

const assistantLine = (model: string, usage: Record<string, unknown>) =>
  JSON.stringify({ type: "assistant", message: { role: "assistant", model, usage } });

describe("aggregateSubagentUsage", () => {
  it("sums every call of a model, thinking and the TTL split included", () => {
    expect(
      aggregateSubagentUsage(
        [
          assistantLine("claude-sonnet-5-5", {
            input_tokens: 11,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: 301,
          }),
          assistantLine("claude-sonnet-5-5", {
            input_tokens: 12,
            cache_creation_input_tokens: 41,
            cache_read_input_tokens: 3001,
            output_tokens: 302,
            output_tokens_details: { thinking_tokens: 100 },
            cache_creation: { ephemeral_5m_input_tokens: 41, ephemeral_1h_input_tokens: 0 },
          }),
          "not json",
          JSON.stringify({ type: "user", message: { role: "user", content: "hi" } }),
        ].join("\n"),
      ),
    ).toEqual([
      {
        model: "claude-sonnet-5-5",
        inputTokens: 23,
        outputTokens: 603,
        thoughtTokens: 100,
        cachedReadTokens: 3001,
        cachedWriteTokens: 41,
        cachedWrite5mTokens: 41,
        cachedWrite1hTokens: 0,
        totalTokens: 3668,
      },
    ]);
  });

  it("sums advisor iterations on top of their message, under each one's model", () => {
    expect(
      aggregateSubagentUsage(
        assistantLine("claude-sonnet-5-5", {
          input_tokens: 10,
          output_tokens: 5,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
          iterations: [
            {
              model: "claude-haiku-4-5",
              input_tokens: 7,
              output_tokens: 3,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 9,
              cache_creation: { ephemeral_5m_input_tokens: 9, ephemeral_1h_input_tokens: 0 },
            },
            { input_tokens: 4, output_tokens: 2 },
          ],
        }),
      ),
    ).toEqual([
      {
        model: "claude-sonnet-5-5",
        inputTokens: 14,
        outputTokens: 7,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
        totalTokens: 21,
      },
      {
        model: "claude-haiku-4-5",
        inputTokens: 7,
        outputTokens: 3,
        cachedReadTokens: 0,
        cachedWriteTokens: 9,
        cachedWrite5mTokens: 9,
        cachedWrite1hTokens: 0,
        totalTokens: 19,
      },
    ]);
  });

  it("keeps a model-less call unknown and a flat cache write unsplit", () => {
    expect(
      aggregateSubagentUsage(
        JSON.stringify({
          message: {
            role: "assistant",
            usage: { input_tokens: 1, cache_creation_input_tokens: 5, output_tokens: 2 },
          },
        }),
      ),
    ).toEqual([
      {
        model: "unknown",
        inputTokens: 1,
        outputTokens: 2,
        cachedReadTokens: 0,
        cachedWriteTokens: 5,
        totalTokens: 8,
      },
    ]);
  });
});

describe("readSubagentUsage", () => {
  it("reads the cumulative usage of one agent transcript", async () => {
    const configDir = await mkdtemp(path.join(os.tmpdir(), "claude-acp-usage-"));
    const original = process.env.CLAUDE_CONFIG_DIR;
    process.env.CLAUDE_CONFIG_DIR = configDir;
    try {
      const sessionId = randomUUID();
      const project = path.join(configDir, "projects", "proj");
      await mkdir(path.join(project, sessionId, "subagents"), { recursive: true });
      await writeFile(path.join(project, `${sessionId}.jsonl`), "{}\n");
      await writeFile(
        path.join(project, sessionId, "subagents", "agent-agt_123.jsonl"),
        [
          assistantLine("claude-sonnet-5-5", {
            input_tokens: 11,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: 301,
          }),
          assistantLine("claude-sonnet-5-5", {
            input_tokens: 12,
            cache_creation_input_tokens: 41,
            cache_read_input_tokens: 3001,
            output_tokens: 302,
            cache_creation: { ephemeral_5m_input_tokens: 41, ephemeral_1h_input_tokens: 0 },
          }),
          "",
        ].join("\n"),
      );

      expect(
        readSubagentUsage({
          sessionId,
          agentId: "agt_123",
          toolUseId: "toolu_t",
          agentType: "Explore",
        }),
      ).toEqual({
        toolUseId: "toolu_t",
        agentId: "agt_123",
        agentType: "Explore",
        models: [
          {
            model: "claude-sonnet-5-5",
            inputTokens: 23,
            outputTokens: 603,
            cachedReadTokens: 3001,
            cachedWriteTokens: 41,
            cachedWrite5mTokens: 41,
            cachedWrite1hTokens: 0,
            totalTokens: 3668,
          },
        ],
      });
      expect(readSubagentUsage({ sessionId, agentId: "agt_missing" })).toBeNull();
      expect(readSubagentUsage({ sessionId, agentId: "../escape" })).toBeNull();
    } finally {
      if (original === undefined) delete process.env.CLAUDE_CONFIG_DIR;
      else process.env.CLAUDE_CONFIG_DIR = original;
      await rm(configDir, { recursive: true, force: true });
    }
  });

  it("reports nothing when the session has no transcript", async () => {
    const configDir = await mkdtemp(path.join(os.tmpdir(), "claude-acp-usage-"));
    const original = process.env.CLAUDE_CONFIG_DIR;
    process.env.CLAUDE_CONFIG_DIR = configDir;
    try {
      expect(readSubagentUsage({ sessionId: randomUUID(), agentId: "agt_1" })).toBeNull();
    } finally {
      if (original === undefined) delete process.env.CLAUDE_CONFIG_DIR;
      else process.env.CLAUDE_CONFIG_DIR = original;
      await rm(configDir, { recursive: true, force: true });
    }
  });
});
