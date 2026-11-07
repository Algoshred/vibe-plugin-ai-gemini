/**
 * vibe-plugin-ai-gemini SDK-mode prompt tests
 *
 * `@google/genai`'s `models.generateContentStream` returns
 * `Promise<AsyncGenerator<GenerateContentResponse>>`: the stream exists only
 * once the HTTP response has arrived. These tests give the SDK adapter a fake
 * client with exactly that contract, so iterating the un-awaited promise
 * (which fails with "iterable should have an iterator symbol") is caught.
 */
import { beforeEach, describe, expect, it } from "bun:test";

const { createPlugin } = await import("../index.js");

interface StreamChunk {
  type: string;
  content: string;
}

interface PromptResult {
  content: string;
  inputTokens: number;
  outputTokens: number;
  metadata?: Record<string, unknown>;
}

interface SdkAdapterLike {
  client: unknown;
  sendPrompt(
    prompt: string,
    model: string,
    config: Record<string, unknown>,
  ): Promise<PromptResult>;
  streamPrompt(
    prompt: string,
    model: string,
    config: Record<string, unknown>,
    onChunk: (chunk: StreamChunk) => void,
  ): Promise<PromptResult>;
}

interface GenerateCall {
  model: string;
  contents: string;
  config?: { maxOutputTokens?: number; temperature?: number };
}

/**
 * The plugin exports one shared provider instance. Switch it to SDK mode and
 * reach the adapter it builds; the private `getAdapter` is not on the public
 * provider type, so narrow structurally (the cast is test-only).
 */
function sdkAdapter(): SdkAdapterLike {
  const plugin = createPlugin({ name: "test", dataDir: "/tmp" });
  const provider = plugin.providers!.ai! as unknown as {
    setMode(mode: "sdk" | "cli"): void;
    getAdapter(): SdkAdapterLike;
  };
  provider.setMode("sdk");
  return provider.getAdapter();
}

function fakeClient(calls: GenerateCall[]): unknown {
  return {
    models: {
      generateContent: async (opts: GenerateCall) => {
        calls.push(opts);
        return {
          text: "hello world",
          usageMetadata: { promptTokenCount: 11, candidatesTokenCount: 7 },
        };
      },
      generateContentStream: async (opts: GenerateCall) => {
        calls.push(opts);
        async function* chunks(): AsyncGenerator<{ text?: string }> {
          yield { text: "hel" };
          yield { text: undefined };
          yield { text: "lo" };
        }
        return chunks();
      },
    },
  };
}

describe("gemini SDK adapter", () => {
  let calls: GenerateCall[];
  let adapter: SdkAdapterLike;

  beforeEach(() => {
    calls = [];
    adapter = sdkAdapter();
    adapter.client = fakeClient(calls);
  });

  it("awaits generateContentStream and forwards every text chunk", async () => {
    const chunks: StreamChunk[] = [];
    const result = await adapter.streamPrompt(
      "hi",
      "gemini-2.5-flash",
      { maxTokens: 10, temperature: 0.2 },
      (chunk) => chunks.push(chunk),
    );

    expect(result.content).toBe("hello");
    expect(result.metadata).toEqual({ mode: "sdk", provider: "gemini" });
    expect(chunks).toEqual([
      { type: "text", content: "hel" },
      { type: "text", content: "lo" },
      { type: "done", content: "" },
    ]);
    expect(calls).toEqual([
      {
        model: "gemini-2.5-flash",
        contents: "hi",
        config: { maxOutputTokens: 10, temperature: 0.2 },
      },
    ]);
  });

  it("returns the text and token usage from generateContent", async () => {
    const result = await adapter.sendPrompt("hi", "gemini-2.5-flash", {});

    expect(result).toEqual({
      content: "hello world",
      inputTokens: 11,
      outputTokens: 7,
      metadata: { mode: "sdk", provider: "gemini" },
    });
    expect(calls).toEqual([
      { model: "gemini-2.5-flash", contents: "hi", config: {} },
    ]);
  });
});
