/**
 * vibe-plugin-ai-gemini credential-resolution tests
 *
 * Verifies the provider resolves its API key from the agent config bag
 * (`hostServices.getConfig`) when it is NOT present in process.env — this is
 * the path the frontend writes to (PUT /api/config/GEMINI_API_KEY).
 */
import { beforeEach, describe, expect, it, mock } from "bun:test";
import type { HostServices } from "@vibecontrols/plugin-sdk";

// Stub the Google GenAI SDK so constructing a client never touches the network.
mock.module("@google/genai", () => {
  class MockGoogleGenAI {
    constructor(_opts: { apiKey: string }) {
      // no-op stub
    }
  }
  return { GoogleGenAI: MockGoogleGenAI };
});

const { createPlugin, permissionFlags } = await import("../index.js");

/**
 * The exported provider type (`AIAgentProvider`) does not surface the optional
 * `setHostServices` lifecycle method, but the concrete provider class
 * implements it. Narrow to a structural type that exposes it for the tests.
 */
interface ProviderWithHost {
  setHostServices(hs: HostServices): void;
  setMode(mode: "sdk" | "cli"): void;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
}

function getProvider(): ProviderWithHost {
  const plugin = createPlugin({ name: "test", dataDir: "/tmp" });
  return plugin.providers!.ai! as unknown as ProviderWithHost;
}

/**
 * The plugin exports a single shared provider instance, so credential state
 * (the warmed `cachedApiKey`, the resolved mode, and the cached adapter/client)
 * leaks between tests. Reset those private fields so each test starts from a
 * cold resolve and genuinely exercises the env → cache → config-bag chain.
 */
function resetProviderState(provider: ProviderWithHost): void {
  const internal = provider as unknown as Record<string, unknown>;
  internal["cachedApiKey"] = undefined;
  internal["hostServices"] = null;
  internal["adapter"] = null;
  internal["currentMode"] = null;
}

/**
 * Build a minimal HostServices whose `getConfig` returns `configValue` for
 * `configKey` (and undefined otherwise). Every HostServices field is optional,
 * so the provider's `setHostServices` (BoundLogger + ProviderRegistry +
 * getConfig) runs without throwing against this fake.
 */
function makeHostServices(
  configKey: string | null,
  configValue: string | undefined,
): { hs: HostServices; getConfig: ReturnType<typeof mock> } {
  const getConfig = mock((key: string): Promise<string | undefined> => {
    if (configKey !== null && key === configKey) {
      return Promise.resolve(configValue);
    }
    return Promise.resolve(undefined);
  });
  const hs: HostServices = {
    logger: {
      debug: () => {},
      info: () => {},
      warn: () => {},
      error: () => {},
    },
    getConfig,
  };
  return { hs, getConfig };
}

function clearEnv(): void {
  delete process.env["GEMINI_API_KEY"];
  delete process.env["GOOGLE_API_KEY"];
}

describe("gemini credential resolution", () => {
  beforeEach(() => {
    clearEnv();
    resetProviderState(getProvider());
  });

  it("resolves the key from the config bag (env cleared) and healthCheck is ok", async () => {
    const provider = getProvider();
    const { hs, getConfig } = makeHostServices(
      "GEMINI_API_KEY",
      "cfg-gemini-key",
    );

    provider.setHostServices(hs);
    // Let the fire-and-forget cache-warm promise settle.
    await new Promise((r) => setTimeout(r, 0));
    provider.setMode("sdk");

    const result = await provider.healthCheck();
    expect(result.ok).toBe(true);
    expect(getConfig).toHaveBeenCalled();
  });

  it("reports ok:false with a /required/ message when no key is available", async () => {
    const provider = getProvider();
    // getConfig returns undefined for every key → no key anywhere.
    const { hs } = makeHostServices(null, undefined);

    provider.setHostServices(hs);
    await new Promise((r) => setTimeout(r, 0));
    provider.setMode("sdk");

    const result = await provider.healthCheck();
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/required/i);
  });

  it("exports permissionFlags with the expected gemini mapping", () => {
    expect(permissionFlags("fullAuto")).toEqual(["--yolo"]);
    expect(permissionFlags("plan")).toEqual([]);
    expect(permissionFlags("acceptEdits")).toEqual([]);
    expect(permissionFlags(undefined)).toEqual([]);
  });
});
