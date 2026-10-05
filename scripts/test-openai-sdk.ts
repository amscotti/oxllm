#!/usr/bin/env -S deno run --allow-net --allow-env

import OpenAI from "npm:openai@4";

// Same origin override as scripts/smoke-docker.sh. Runtime SDK checks need a
// gateway with these aliases (or a deterministic mock upstream) configured.
const PROXY_URL = (Deno.env.get("OXLLM_URL") || "http://127.0.0.1:4000").replace(/\/+$/, "");
const API_KEY = Deno.env.get("OXLLM_MASTER_KEY") || "sk-ox-local-smoke-master";
const CHAT_MODEL = Deno.env.get("OXLLM_SDK_CHAT_MODEL") || "claude-sonnet";
const CROSS_MODEL = Deno.env.get("OXLLM_SDK_CROSS_MODEL") || "grok";

const client = new OpenAI({
  baseURL: `${PROXY_URL}/v1`,
  apiKey: API_KEY,
  timeout: 60_000,
  maxRetries: 0,
});

let failures = 0;

function failed(error: unknown) {
  failures += 1;
  console.error("   ✗ FAILED:", error instanceof Error ? error.message : String(error));
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function assertText(value: unknown): asserts value is string {
  assert(typeof value === "string" && value.trim().length > 0, "Expected non-empty assistant text");
}

function write(text: string) {
  Deno.stdout.writeSync(new TextEncoder().encode(text));
}

console.log("=== OpenAI SDK Tests ===\n");

// Test 1: Basic chat completion (Anthropic via OpenAI SDK)
console.log(`1. Basic chat completion (${CHAT_MODEL} via OpenAI SDK)...`);
try {
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: "Say hello in exactly 3 words." }],
    max_tokens: 20,
  });
  assertText(resp.choices[0]?.message?.content);
  console.log("   ✓ Response:", resp.choices[0]?.message?.content);
  console.log("   ✓ Model:", resp.model);
  console.log("   ✓ Usage:", JSON.stringify(resp.usage));
} catch (e) {
  failed(e);
}

// Test 2: Streaming chat completion
console.log(`\n2. Streaming chat completion (${CHAT_MODEL})...`);
try {
  const stream = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: "Count from 1 to 5." }],
    max_tokens: 50,
    stream: true,
  }, { signal: AbortSignal.timeout(60_000) });
  let text = "";
  let finished = false;
  write("   Stream: ");
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || "";
    text += content;
    finished ||= chunk.choices.some((choice) => typeof choice.finish_reason === "string");
    write(content);
  }
  console.log();
  assertText(text);
  assert(finished, "Stream ended without a finish_reason");
} catch (e) {
  failed(e);
}

// Test 3: Grok (xAI) via OpenAI SDK
console.log(`\n3. Cross-provider ${CROSS_MODEL} via OpenAI SDK...`);
try {
  const resp = await client.chat.completions.create({
    model: CROSS_MODEL,
    messages: [{ role: "user", content: "Say hello in exactly 3 words." }],
    max_tokens: 20,
  });
  assertText(resp.choices[0]?.message?.content);
  console.log("   ✓ Response:", resp.choices[0]?.message?.content);
  console.log("   ✓ Model:", resp.model);
} catch (e) {
  failed(e);
}

// Test 4: Tool/function calling
console.log(`\n4. Tool/function calling (${CHAT_MODEL})...`);
try {
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: "What is the weather in Tokyo? Use the get_weather tool." }],
    max_tokens: 200,
    tools: [{
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather for a city",
        parameters: {
          type: "object",
          properties: { city: { type: "string", description: "City name" } },
          required: ["city"],
        },
      },
    }],
    tool_choice: { type: "function", function: { name: "get_weather" } },
  });
  const toolCall = resp.choices[0]?.message?.tool_calls?.[0];
  assert(toolCall, "Forced get_weather tool call is missing");
  assert(toolCall.function.name === "get_weather", "Unexpected tool name");
  const args = JSON.parse(toolCall.function.arguments);
  assert(typeof args?.city === "string", "Tool arguments must contain city:string");
  console.log("   ✓ Tool called:", toolCall.function.name);
  console.log("   ✓ Arguments:", toolCall.function.arguments);
} catch (e) {
  failed(e);
}

// Test 5: Multi-turn conversation
console.log("\n5. Multi-turn conversation...");
try {
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "user", content: "My name is TestUser." },
      { role: "assistant", content: "Nice to meet you, TestUser!" },
      { role: "user", content: "What is my name? Just the name, nothing else." },
    ],
    max_tokens: 20,
  });
  assertText(resp.choices[0]?.message?.content);
  console.log("   ✓ Response:", resp.choices[0]?.message?.content);
} catch (e) {
  failed(e);
}

// Test 6: System message
console.log("\n6. System message (Anthropic system extraction)...");
try {
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: "You are a pirate. Always respond in pirate speak." },
      { role: "user", content: "Hello." },
    ],
    max_tokens: 50,
  });
  assertText(resp.choices[0]?.message?.content);
  console.log("   ✓ Response:", resp.choices[0]?.message?.content);
} catch (e) {
  failed(e);
}

// Test 7: List models
console.log("\n7. List models...");
try {
  const models = await client.models.list();
  assert(models.data.length > 0, "Expected configured models in the model list");
  for (const m of models.data) {
    console.log(`   ✓ ${m.id} (owned by ${m.owned_by})`);
  }
} catch (e) {
  failed(e);
}

console.log(`\n=== OpenAI SDK Tests Complete: ${failures} failed ===`);
Deno.exitCode = failures > 0 ? 1 : 0;
