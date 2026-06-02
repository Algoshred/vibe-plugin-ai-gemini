/**
 * vibe-plugin-gemini
 *
 * Google Gemini AI agent provider for VibeControls Agent.
 * Dual-mode: SDK (@google/genai) or CLI (`gemini` binary).
 * Auto-detects mode based on available resources.
 */

import { Elysia } from "elysia";
import type {
  HostServices,
  VibePlugin,
  ProfileContext,
} from "@vibecontrols/plugin-sdk";
import {
  BoundLogger,
  ProviderRegistry,
  TelemetryEmitter,
  createLifecycleHooks,
} from "@vibecontrols/plugin-sdk";

// ── AI Provider Contract Types ──────────────────────────────────────────
// (provider-specific contract — kept inline; not part of the SDK surface)

type ProviderMode = "sdk" | "cli";

interface AIModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsStreaming: boolean;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

interface AIProviderCapabilities {
  streaming: boolean;
  vision: boolean;
  fileAttachments: boolean;
  toolUse: boolean;
  mcpSupport: boolean;
  voiceMode: boolean;
  cancelSupport: boolean;
  modelListing: boolean;
}

interface AIFileAttachment {
  filename: string;
  mimeType: string;
  content: Buffer | string;
  size: number;
}

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

export type PermissionMode = "plan" | "acceptEdits" | "fullAuto";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  /** Autonomy level for CLI mode; ignored by the SDK adapter. */
  permissionMode?: PermissionMode;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  thinkingSteps?: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
}

interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}

interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  agentMetadata?: Record<string, unknown>;
  createdAt: string;
}

interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
  modelBreakdown?: Record<
    string,
    { inputTokens: number; outputTokens: number; requestCount: number }
  >;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancelRequest?(sessionId: string): Promise<void>;
  getCapabilities?(): AIProviderCapabilities;
  attachFiles?(sessionId: string, files: AIFileAttachment[]): Promise<void>;
  getMode?(): ProviderMode;
  setMode?(mode: ProviderMode): void;
  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null;
  sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }>;
}

interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
    agentMetadata?: Record<string, unknown>;
  }): unknown;
}

// ── Adapter Interface ───────────────────────────────────────────────────

interface ProviderAdapter {
  sendPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }>;
  streamPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancel?(abortController: AbortController): void;
}

// ── Constants ───────────────────────────────────────────────────────────

const PROVIDER_NAME = "gemini";
const CLI_COMMAND = "gemini";
/**
 * Resolve CLI binary path with platform-correct extension.
 * On Windows, Bun.spawn calls CreateProcess directly (no PATHEXT), so a bare
 * name won't find `name.exe`/`name.cmd`. Bun.which searches PATH like the shell.
 */
function platformExeName(base: string): string {
  return process.platform === "win32" ? `${base}.exe` : base;
}

function resolveCliBin(): string {
  const found =
    typeof Bun !== "undefined" && typeof Bun.which === "function"
      ? Bun.which(CLI_COMMAND, { PATH: process.env.PATH })
      : null;
  if (found) return found;
  return platformExeName(CLI_COMMAND);
}
const CLI_BIN = resolveCliBin();

const DISPLAY_NAME = "Google Gemini";
const DEFAULT_MODEL = "gemini-2.5-flash";
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk", "cli"];
const CLI_NPM_PACKAGE = "@google/gemini-cli";

const KNOWN_MODELS: AIModelInfo[] = [
  {
    id: "gemini-2.5-flash",
    name: "Gemini 2.5 Flash",
    provider: PROVIDER_NAME,
    contextWindow: 1_048_576,
    maxOutputTokens: 65_536,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 0.15,
    outputPricePerMToken: 0.6,
  },
  {
    id: "gemini-2.5-pro",
    name: "Gemini 2.5 Pro",
    provider: PROVIDER_NAME,
    contextWindow: 1_048_576,
    maxOutputTokens: 65_536,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 1.25,
    outputPricePerMToken: 10.0,
  },
  {
    id: "gemini-2.0-flash",
    name: "Gemini 2.0 Flash",
    provider: PROVIDER_NAME,
    contextWindow: 1_048_576,
    maxOutputTokens: 8_192,
    supportsVision: true,
    supportsStreaming: true,
    inputPricePerMToken: 0.1,
    outputPricePerMToken: 0.4,
  },
];

// ── SDK Adapter ─────────────────────────────────────────────────────────

class GeminiSdkAdapter implements ProviderAdapter {
  private client: unknown = null;
  private readonly resolveApiKey: () => Promise<string | undefined>;

  /**
   * Takes an async resolver rather than a static key so the adapter always
   * reads the freshest credential — env var OR the agent config bag the
   * frontend writes to (PUT /api/config/GEMINI_API_KEY). Resolving lazily at
   * first use (not at construction) means a key saved after the session was
   * created still takes effect without recreating the adapter.
   */
  constructor(resolveApiKey: () => Promise<string | undefined>) {
    this.resolveApiKey = resolveApiKey;
  }

  private async getClient(): Promise<{
    models: {
      generateContent(opts: {
        model: string;
        contents: string;
        config?: {
          maxOutputTokens?: number;
          temperature?: number;
          abortSignal?: AbortSignal;
        };
      }): Promise<{
        text: string;
        usageMetadata?: {
          promptTokenCount?: number;
          candidatesTokenCount?: number;
        };
      }>;
      generateContentStream(opts: {
        model: string;
        contents: string;
        config?: {
          maxOutputTokens?: number;
          temperature?: number;
          abortSignal?: AbortSignal;
        };
      }): AsyncIterable<{ text: string }> & {
        [Symbol.asyncIterator](): AsyncIterator<{ text: string }>;
      };
    };
  }> {
    if (this.client) {
      return this.client as ReturnType<typeof this.getClient> extends Promise<
        infer T
      >
        ? T
        : never;
    }

    const apiKey = (await this.resolveApiKey())?.trim();
    if (!apiKey) {
      throw new Error(
        "GOOGLE_API_KEY or GEMINI_API_KEY is required for SDK mode. Set it " +
          "in the AI provider credentials (it is stored in the agent config) " +
          "or export it in the agent environment.",
      );
    }

    let GoogleGenAI: new (opts: { apiKey: string }) => unknown;
    try {
      const mod = await import("@google/genai");
      GoogleGenAI = mod.GoogleGenAI;
    } catch {
      throw new Error(
        "Failed to load @google/genai SDK. Install it with: bun add @google/genai",
      );
    }
    this.client = new GoogleGenAI({ apiKey });
    return this.client as ReturnType<typeof this.getClient> extends Promise<
      infer T
    >
      ? T
      : never;
  }

  async sendPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }> {
    const ai = await this.getClient();

    const sdkConfig: {
      maxOutputTokens?: number;
      temperature?: number;
    } = {};
    if (config.maxTokens) sdkConfig.maxOutputTokens = config.maxTokens;
    if (config.temperature !== undefined)
      sdkConfig.temperature = config.temperature;

    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: sdkConfig,
    });

    const content = response.text ?? "";
    const inputTokens =
      response.usageMetadata?.promptTokenCount ?? Math.ceil(prompt.length / 4);
    const outputTokens =
      response.usageMetadata?.candidatesTokenCount ??
      Math.ceil(content.length / 4);

    return {
      content,
      inputTokens,
      outputTokens,
      metadata: { mode: "sdk", provider: PROVIDER_NAME },
    };
  }

  async streamPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }> {
    const ai = await this.getClient();

    const sdkConfig: {
      maxOutputTokens?: number;
      temperature?: number;
    } = {};
    if (config.maxTokens) sdkConfig.maxOutputTokens = config.maxTokens;
    if (config.temperature !== undefined)
      sdkConfig.temperature = config.temperature;

    const stream = ai.models.generateContentStream({
      model,
      contents: prompt,
      config: sdkConfig,
    });

    let fullContent = "";
    for await (const chunk of stream) {
      const text = chunk.text ?? "";
      if (text) {
        fullContent += text;
        onChunk({ type: "text", content: text });
      }
    }

    onChunk({ type: "done", content: "" });

    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(fullContent.length / 4);

    return {
      content: fullContent,
      inputTokens,
      outputTokens,
      metadata: { mode: "sdk", provider: PROVIDER_NAME },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      await this.getClient();
      return {
        ok: true,
        message: `${DISPLAY_NAME} SDK initialized (mode: sdk)`,
      };
    } catch (err) {
      return {
        ok: false,
        message: err instanceof Error ? err.message : "SDK init failed",
      };
    }
  }

  async listModels(): Promise<AIModelInfo[]> {
    return KNOWN_MODELS;
  }

  cancel(_abortController: AbortController): void {
    _abortController.abort();
  }
}

// ── CLI Adapter ─────────────────────────────────────────────────────────

class GeminiCliAdapter implements ProviderAdapter {
  private readonly resolveApiKey: () => Promise<string | undefined>;

  constructor(resolveApiKey: () => Promise<string | undefined>) {
    this.resolveApiKey = resolveApiKey;
  }

  /**
   * Build the spawn environment, layering the resolved API key on top of the
   * agent's own env. The Gemini CLI authenticates from GEMINI_API_KEY /
   * GOOGLE_API_KEY, so injecting the key the user saved in the agent config
   * makes CLI mode work without a separate `gemini` login on the sandbox.
   */
  private async spawnEnv(): Promise<Record<string, string>> {
    const env: Record<string, string> = {
      ...(process.env as Record<string, string>),
    };
    const apiKey = (await this.resolveApiKey())?.trim();
    if (apiKey) {
      env["GEMINI_API_KEY"] = apiKey;
      env["GOOGLE_API_KEY"] = apiKey;
    }
    return env;
  }

  async sendPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }> {
    const args = this.buildArgs(model, prompt, config);
    const proc = Bun.spawn([CLI_BIN, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      env: await this.spawnEnv(),
      timeout: (config.providerConfig?.timeoutMs as number) || 300_000,
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0 && !stdout) {
      throw new Error(
        `${DISPLAY_NAME} CLI exited with code ${exitCode}: ${stderr}`,
      );
    }

    const content = stdout.trim() || stderr.trim();
    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(content.length / 4);

    return {
      content,
      inputTokens,
      outputTokens,
      metadata: { mode: "cli", exitCode, provider: PROVIDER_NAME },
    };
  }

  async streamPrompt(
    prompt: string,
    model: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
  ): Promise<{
    content: string;
    inputTokens: number;
    outputTokens: number;
    metadata?: Record<string, unknown>;
  }> {
    const args = this.buildArgs(model, prompt, config);
    const proc = Bun.spawn([CLI_BIN, ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: config.workingDirectory || process.cwd(),
      env: await this.spawnEnv(),
      timeout: (config.providerConfig?.timeoutMs as number) || 300_000,
    });

    let fullContent = "";
    const reader = proc.stdout.getReader();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = new TextDecoder().decode(value);
        fullContent += text;
        onChunk({ type: "text", content: text });
      }
    } finally {
      reader.releaseLock();
    }

    await proc.exited;
    onChunk({ type: "done", content: "" });

    const inputTokens = Math.ceil(prompt.length / 4);
    const outputTokens = Math.ceil(fullContent.length / 4);

    return {
      content: fullContent,
      inputTokens,
      outputTokens,
      metadata: { mode: "cli", provider: PROVIDER_NAME },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const proc = Bun.spawnSync([CLI_BIN, "--version"], {
        timeout: 5000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0) {
        return {
          ok: true,
          message: `${DISPLAY_NAME} CLI ${proc.stdout.toString().trim()} (mode: cli)`,
        };
      }
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not available (exit code ${proc.exitCode})`,
      };
    } catch {
      return {
        ok: false,
        message: `${DISPLAY_NAME} CLI not installed or not in PATH`,
      };
    }
  }

  async listModels(): Promise<AIModelInfo[]> {
    return KNOWN_MODELS;
  }

  private buildArgs(
    model: string,
    prompt: string,
    config: AISessionConfig,
  ): string[] {
    const args: string[] = ["prompt"];
    if (model && model !== "default") args.push("--model", model);
    args.push(...permissionFlags(config.permissionMode));
    args.push(prompt);
    return args;
  }
}

/**
 * Map the provider-agnostic permission mode to Gemini CLI flags.
 * Gemini CLI exposes `--yolo` (auto-approve everything) and, on newer
 * versions, `--approval-mode`. Unknown/undefined → safe default
 * (acceptEdits). plan/acceptEdits emit nothing when the version lacks an
 * approval flag (the CLI then prompts/defaults). CLI-version-dependent.
 */
export function permissionFlags(mode: PermissionMode | undefined): string[] {
  switch (mode) {
    case "fullAuto":
      return ["--yolo"];
    case "plan":
    case "acceptEdits":
    default:
      return [];
  }
}

// ── Managed Session ─────────────────────────────────────────────────────

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  files: AIFileAttachment[];
  abortController: AbortController | null;
  createdAt: string;
  updatedAt: string;
}

// ── Provider Implementation ─────────────────────────────────────────────

class GeminiProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private logger: BoundLogger | null = null;
  private adapter: ProviderAdapter | null = null;
  private currentMode: ProviderMode | null = null;
  private cachedApiKey: string | undefined;

  setHostServices(hs: HostServices) {
    this.hostServices = hs;
    this.logger = new BoundLogger(hs.logger, `${PROVIDER_NAME}-provider`);
    const registry = new ProviderRegistry(hs);
    this.logIngester =
      registry.getProvider<LogIngester>("ai", "log-ingester") ?? null;

    // Warm the cache so autoDetectMode()/getCliLaunchSpec() (both sync) can
    // see a key the user stored in the agent config bag, not just env vars.
    void Promise.all([
      Promise.resolve(hs.getConfig?.("GEMINI_API_KEY")),
      Promise.resolve(hs.getConfig?.("GOOGLE_API_KEY")),
    ])
      .then(([gemini, google]) => {
        const key = gemini?.trim() || google?.trim();
        if (key) this.cachedApiKey = key;
      })
      .catch(() => {});
  }

  /**
   * Resolve the Gemini API key from, in order: the process env (operator
   * override always wins), the warmed cache, then the agent config bag the
   * frontend writes to. Mirrors the resolution every other provider uses so
   * SDK + CLI mode work with a key saved purely through the UI.
   */
  private async resolveApiKey(): Promise<string | undefined> {
    const envKey =
      process.env["GOOGLE_API_KEY"]?.trim() ||
      process.env["GEMINI_API_KEY"]?.trim();
    if (envKey) return envKey;

    if (this.cachedApiKey) return this.cachedApiKey;

    if (this.hostServices?.getConfig) {
      try {
        const gemini = (
          await this.hostServices.getConfig("GEMINI_API_KEY")
        )?.trim();
        if (gemini) {
          this.cachedApiKey = gemini;
          return gemini;
        }
        const google = (
          await this.hostServices.getConfig("GOOGLE_API_KEY")
        )?.trim();
        if (google) {
          this.cachedApiKey = google;
          return google;
        }
      } catch {
        return undefined;
      }
    }
    return undefined;
  }

  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }

  getDisplayName(): string {
    return DISPLAY_NAME;
  }

  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }

  getMode(): ProviderMode {
    if (!this.currentMode) {
      this.autoDetectMode();
    }
    return this.currentMode!;
  }

  setMode(mode: ProviderMode): void {
    if (!SUPPORTED_MODES.includes(mode)) {
      throw new Error(`${DISPLAY_NAME} does not support ${mode} mode`);
    }
    this.currentMode = mode;
    this.adapter = null;
    this.log("info", `Mode set to: ${mode}`);
  }

  getCapabilities(): AIProviderCapabilities {
    return {
      streaming: true,
      vision: true,
      fileAttachments: true,
      toolUse: true,
      mcpSupport: false,
      voiceMode: false,
      cancelSupport: true,
      modelListing: true,
    };
  }

  async listModels(): Promise<AIModelInfo[]> {
    const adapter = this.getAdapter();
    if (adapter.listModels) return adapter.listModels();
    return KNOWN_MODELS;
  }

  async cancelRequest(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.abortController) {
      session.abortController.abort();
      session.abortController = null;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Request cancelled for session ${sessionId}`);
    }
  }

  async attachFiles(
    sessionId: string,
    files: AIFileAttachment[],
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    session.files.push(...files);
    session.updatedAt = new Date().toISOString();
    this.log(
      "debug",
      `Attached ${files.length} file(s) to session ${sessionId}`,
    );
  }

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();

    // If session already exists, return it
    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return {
        id,
        name: existing.config.name,
        status: "active",
        agentType: existing.config.agentType,
        provider: PROVIDER_NAME,
        config: existing.config,
        stats: existing.stats,
        createdAt: existing.createdAt,
        updatedAt: now,
      };
    }

    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      files: [],
      abortController: null,
      createdAt: now,
      updatedAt: now,
    };

    this.sessions.set(id, session);
    this.log(
      "info",
      `Session created: ${id} (${config.name}) [${this.getMode()}]`,
    );

    return {
      id,
      name: config.name,
      status: "active",
      agentType: config.agentType,
      provider: PROVIDER_NAME,
      config,
      stats: session.stats,
      createdAt: now,
      updatedAt: now,
    };
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");

    session.status = "processing";
    session.updatedAt = new Date().toISOString();
    const startTime = Date.now();

    const fullPrompt = this.buildFullPrompt(prompt, context, session);
    const model = session.config.model || DEFAULT_MODEL;
    const adapter = this.getAdapter();

    this.logIngester?.append({ sessionId, type: "input", content: prompt });

    try {
      const result = await adapter.sendPrompt(
        fullPrompt,
        model,
        session.config,
      );
      const durationMs = Date.now() - startTime;

      this.updateStats(session, result.inputTokens, result.outputTokens, model);
      session.status = "active";
      session.updatedAt = new Date().toISOString();

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model,
        durationMs,
      });

      return {
        content: result.content,
        model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({ sessionId, type: "error", content: errorMsg });
      throw err;
    }
  }

  async streamPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");

    session.status = "processing";
    session.abortController = new AbortController();
    session.updatedAt = new Date().toISOString();
    const startTime = Date.now();

    const fullPrompt = this.buildFullPrompt(prompt, context, session);
    const model = session.config.model || DEFAULT_MODEL;
    const adapter = this.getAdapter();

    this.logIngester?.append({ sessionId, type: "input", content: prompt });

    try {
      const result = await adapter.streamPrompt(
        fullPrompt,
        model,
        session.config,
        onChunk ?? (() => {}),
      );
      const durationMs = Date.now() - startTime;

      this.updateStats(session, result.inputTokens, result.outputTokens, model);
      session.status = "active";
      session.abortController = null;
      session.updatedAt = new Date().toISOString();

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model,
        durationMs,
      });

      return {
        content: result.content,
        model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.abortController = null;
      session.updatedAt = new Date().toISOString();
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({ sessionId, type: "error", content: errorMsg });
      throw err;
    }
  }

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }

  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    const session = this.sessions.get(sessionId);
    return (
      session?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }

  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    Object.assign(session.config, config);
    session.updatedAt = new Date().toISOString();
  }

  async destroySession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.abortController) session.abortController.abort();
      session.status = "terminated";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Session terminated: ${sessionId}`);
    }
  }

  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => ({
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    }));
  }

  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    const adapter = this.getAdapter();
    return adapter.healthCheck();
  }

  // ── `vibe ai run` / `vibe ai sdk` integration ────────────────────────

  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null {
    const env: Record<string, string> = {};
    const apiKey =
      process.env["GOOGLE_API_KEY"]?.trim() ||
      process.env["GEMINI_API_KEY"]?.trim() ||
      this.cachedApiKey;
    if (apiKey) {
      env["GOOGLE_API_KEY"] = apiKey;
      env["GEMINI_API_KEY"] = apiKey;
    }
    return { binary: CLI_COMMAND, env };
  }

  async sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }> {
    const adapter = new GeminiSdkAdapter(() => this.resolveApiKey());
    const model = opts.model ?? DEFAULT_MODEL;
    const config: AISessionConfig = {
      name: "vibe-ai-sdk",
      agentType: PROVIDER_NAME,
      model,
      maxTokens: opts.maxTokens,
      providerConfig: opts.extras,
    };
    const result = await adapter.sendPrompt(opts.prompt, model, config);
    return {
      text: result.content,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        model,
      },
    };
  }

  // ── Private Helpers ─────────────────────────────────────────────────

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;

    const mode = this.getMode();
    // The adapters resolve the key lazily (env → cache → config bag), so we
    // no longer throw here for a missing key — that check moved into the SDK
    // adapter, which can read the config bag the FE writes to.
    this.adapter =
      mode === "sdk"
        ? new GeminiSdkAdapter(() => this.resolveApiKey())
        : new GeminiCliAdapter(() => this.resolveApiKey());

    return this.adapter;
  }

  private autoDetectMode(): void {
    const apiKey =
      process.env["GOOGLE_API_KEY"]?.trim() ||
      process.env["GEMINI_API_KEY"]?.trim() ||
      this.cachedApiKey;
    if (apiKey) {
      this.currentMode = "sdk";
      this.log("info", "Auto-detected SDK mode (API key found)");
      return;
    }

    try {
      // Cross-platform binary discovery via Bun.which (handles PATHEXT on Windows).
      if (Bun.which(CLI_COMMAND, { PATH: process.env.PATH })) {
        this.currentMode = "cli";
        this.log("info", "Auto-detected CLI mode (gemini binary found)");
        return;
      }
    } catch {
      // binary not found
    }

    this.currentMode = "cli";
    this.log(
      "info",
      "No API key or CLI binary found, defaulting to CLI mode (will error on use)",
    );
  }

  private buildFullPrompt(
    prompt: string,
    context: AIContext[] | undefined,
    session: ManagedSession,
  ): string {
    const parts: string[] = [];

    if (session.config.systemPrompt) {
      parts.push(`System: ${session.config.systemPrompt}\n`);
    }

    if (context && context.length > 0) {
      const contextStr = context
        .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
        .join("\n\n");
      parts.push(contextStr);
    }

    if (session.files.length > 0) {
      const fileContext = session.files
        .map((f) => {
          const text =
            typeof f.content === "string"
              ? f.content
              : f.content.toString("utf-8");
          return `--- File: ${f.filename} (${f.mimeType}) ---\n${text}`;
        })
        .join("\n\n");
      parts.push(fileContext);
    }

    parts.push(prompt);
    return parts.join("\n\n");
  }

  private updateStats(
    session: ManagedSession,
    inputTokens: number,
    outputTokens: number,
    model: string,
  ): void {
    session.stats.inputTokens += inputTokens;
    session.stats.outputTokens += outputTokens;
    session.stats.requestCount += 1;

    const modelInfo = KNOWN_MODELS.find((m) => m.id === model);
    if (modelInfo) {
      session.stats.estimatedCostUsd +=
        (inputTokens / 1_000_000) * modelInfo.inputPricePerMToken +
        (outputTokens / 1_000_000) * modelInfo.outputPricePerMToken;
    }

    if (!session.stats.modelBreakdown) session.stats.modelBreakdown = {};
    const breakdown = session.stats.modelBreakdown[model] ?? {
      inputTokens: 0,
      outputTokens: 0,
      requestCount: 0,
    };
    breakdown.inputTokens += inputTokens;
    breakdown.outputTokens += outputTokens;
    breakdown.requestCount += 1;
    session.stats.modelBreakdown[model] = breakdown;
  }

  private log(level: "info" | "error" | "debug", msg: string) {
    this.logger?.[level](msg);
  }
}

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([CLI_BIN, "--version"], {
      timeout: 5000,
      stdout: "pipe",
      stderr: "ignore",
    });
    if (proc.exitCode === 0) return proc.stdout.toString().trim();
  } catch {
    // Binary not found.
  }
  return null;
}

/**
 * Install a global npm CLI, runtime-resiliently. The agent always ships Bun
 * (it IS a Bun process) but NOT npm/node — the production agent image is Alpine
 * + Bun only — so a hard-coded `npm install -g` silently fails there. We try
 * each available global installer in turn and report the last error.
 */
function installGlobalNpmCli(pkgSpec: string): {
  ok: boolean;
  message: string;
} {
  const candidates: string[][] = [
    ["bun", "install", "-g", pkgSpec],
    ["npm", "install", "-g", pkgSpec],
  ];
  let lastError = "";
  for (const cmd of candidates) {
    const exe = cmd[0]!;
    if (!Bun.which(exe, { PATH: process.env.PATH })) continue;
    try {
      const proc = Bun.spawnSync(cmd, {
        timeout: 180_000,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode === 0) return { ok: true, message: cmd.join(" ") };
      lastError =
        proc.stderr.toString().trim() || `${exe} exited ${proc.exitCode}`;
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    }
  }
  return {
    ok: false,
    message:
      lastError ||
      `No global installer (bun/npm) found. Run manually: bun install -g ${pkgSpec}`,
  };
}

function createPrereqsRoutes() {
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => {
      const version = getCliVersion();
      return {
        satisfied: Boolean(version),
        missing: version
          ? []
          : [
              {
                name: CLI_COMMAND,
                kind: "npm" as const,
                requiresSudo: false,
                description: `${DISPLAY_NAME} CLI for CLI mode`,
              },
            ],
      };
    })
    .post("/install", () => {
      if (getCliVersion()) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }

      const result = installGlobalNpmCli(CLI_NPM_PACKAGE);
      if (result.ok) {
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      }
      return {
        ok: false,
        installed: [],
        pendingSudo: [],
        errors: [{ name: CLI_COMMAND, message: result.message }],
      };
    });
}

const PLUGIN_NAME = "gemini";
const PLUGIN_VERSION = "1.0.0";

const provider = new GeminiProvider();

const lifecycle = createLifecycleHooks({
  name: PLUGIN_NAME,
  telemetryEventName: "ai.provider.ready",
  onInit: (hostServices: HostServices) => {
    provider.setHostServices(hostServices);
    new TelemetryEmitter(PLUGIN_NAME, PLUGIN_VERSION, hostServices).emit(
      "ai.provider.ready",
      { provider: PLUGIN_NAME },
    );
  },
  onShutdown: () => {
    for (const [id] of (provider as GeminiProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
});

type GeminiVibePlugin = VibePlugin & {
  providers?: { ai?: AIAgentProvider };
};

export const createPlugin = (_ctx: ProfileContext): GeminiVibePlugin => ({
  capabilities: {
    secrets: "read",
    subprocess: true,
    gateway: false,
    telemetry: true,
  },
  name: PLUGIN_NAME,
  version: PLUGIN_VERSION,
  description:
    "Google Gemini AI agent provider for VibeControls (SDK + CLI dual-mode)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: "npm",
      requiresSudo: false,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),
  onServerStart: lifecycle.onServerStart,
  onServerStop: lifecycle.onServerStop,
});
