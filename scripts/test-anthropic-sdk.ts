#!/usr/bin/env -S deno run --allow-net --allow-env

import Anthropic from "npm:@anthropic-ai/sdk@^0.39.0";

// Match the deployment smoke client's origin override. These runtime checks
// call model endpoints; CI can point them at a gateway with mock providers.
const PROXY_URL = (Deno.env.get("OXLLM_URL") || "http://127.0.0.1:4000").replace(/\/+$/, "");
const API_KEY = Deno.env.get("OXLLM_MASTER_KEY") || "sk-ox-local-smoke-master";
const CHAT_MODEL = Deno.env.get("OXLLM_SDK_CHAT_MODEL") || "claude-sonnet";
const CROSS_MODEL = Deno.env.get("OXLLM_SDK_CROSS_MODEL") || "grok";

const client = new Anthropic({
  baseURL: PROXY_URL,
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

function assertText(message: Anthropic.Message) {
  assert(message.content.some((block) => block.type === "text" && block.text.trim().length > 0),
    "Expected non-empty assistant text");
}

function assertWeatherTool(message: Anthropic.Message) {
  const tool = message.content.find((block) => block.type === "tool_use");
  assert(tool?.type === "tool_use", "Forced get_weather tool call is missing");
  assert(tool.name === "get_weather", "Unexpected tool name");
  assert(typeof (tool.input as { city?: unknown })?.city === "string",
    "Tool input must contain city:string");
  console.log("   ✓ Tool called:", tool.name);
  console.log("   ✓ Input:", JSON.stringify(tool.input));
}

function write(text: string) {
  Deno.stdout.writeSync(new TextEncoder().encode(text));
}

console.log("=== Anthropic SDK Tests ===\n");

// Test 1: Basic message (Anthropic native format via proxy)
console.log(`1. Basic message (${CHAT_MODEL} via Anthropic SDK)...`);
try {
  const resp = await client.messages.create({
    model: CHAT_MODEL,
    max_tokens: 20,
    messages: [{ role: "user", content: "Say hello in exactly 3 words." }],
  });
  assertText(resp);
  console.log("   ✓ Content:", resp.content[0]?.type === "text" ? resp.content[0].text : "(non-text)");
  console.log("   ✓ Model:", resp.model);
  console.log("   ✓ Usage:", JSON.stringify(resp.usage));
  console.log("   ✓ Stop reason:", resp.stop_reason);
} catch (e) {
  failed(e);
}

// Test 2: Streaming message
console.log(`\n2. Streaming message (${CHAT_MODEL})...`);
try {
  const stream = client.messages.stream({
    model: CHAT_MODEL,
    max_tokens: 50,
    messages: [{ role: "user", content: "Count from 1 to 5." }],
  }, { signal: AbortSignal.timeout(60_000) });
  write("   Stream: ");
  for await (const event of stream) {
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
      write(event.delta.text);
    }
  }
  console.log();
  assertText(await stream.finalMessage());
} catch (e) {
  failed(e);
}

// Test 3: System message
console.log("\n3. System message...");
try {
  const resp = await client.messages.create({
    model: CHAT_MODEL,
    max_tokens: 50,
    system: "You are a pirate. Always respond in pirate speak.",
    messages: [{ role: "user", content: "Hello." }],
  });
  assertText(resp);
  console.log("   ✓ Response:", resp.content[0]?.type === "text" ? resp.content[0].text : "(non-text)");
} catch (e) {
  failed(e);
}

// Test 4: Tool use
console.log(`\n4. Tool use (${CHAT_MODEL})...`);
try {
  const resp = await client.messages.create({
    model: CHAT_MODEL,
    max_tokens: 200,
    messages: [{ role: "user", content: "What is the weather in Tokyo? Use the get_weather tool." }],
    tools: [{
      name: "get_weather",
      description: "Get weather for a city",
      input_schema: {
        type: "object",
        properties: { city: { type: "string", description: "City name" } },
        required: ["city"],
      },
    }],
    tool_choice: { type: "tool", name: "get_weather" },
  });
  assertWeatherTool(resp);
} catch (e) {
  failed(e);
}

// Test 5: Multi-turn conversation
console.log("\n5. Multi-turn conversation...");
try {
  const resp = await client.messages.create({
    model: CHAT_MODEL,
    max_tokens: 20,
    messages: [
      { role: "user", content: "My name is TestUser." },
      { role: "assistant", content: "Nice to meet you, TestUser!" },
      { role: "user", content: "What is my name? Just the name." },
    ],
  });
  assertText(resp);
  console.log("   ✓ Response:", resp.content[0]?.type === "text" ? resp.content[0].text : "(non-text)");
} catch (e) {
  failed(e);
}

// Test 6: xAI Grok via Anthropic SDK (cross-provider via /v1/messages)
// This tests that oxllm can accept Anthropic format and forward to a non-Anthropic provider
console.log(`\n6. Cross-provider: ${CROSS_MODEL} via Anthropic SDK (/v1/messages)...`);
try {
  const resp = await client.messages.create({
    model: CROSS_MODEL,
    max_tokens: 20,
    messages: [{ role: "user", content: "Say hello in exactly 3 words." }],
  });
  assertText(resp);
  console.log("   ✓ Content:", resp.content[0]?.type === "text" ? resp.content[0].text : "(non-text)");
  console.log("   ✓ Model:", resp.model);
} catch (e) {
  failed(e);
}

// Test 7: Streaming with tool use
console.log("\n7. Streaming with tool use...");
try {
  const stream = client.messages.stream({
    model: CHAT_MODEL,
    max_tokens: 200,
    messages: [{ role: "user", content: "Check the weather in Paris using the get_weather tool." }],
    tools: [{
      name: "get_weather",
      description: "Get weather for a city",
      input_schema: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
    }],
    tool_choice: { type: "tool", name: "get_weather" },
  }, { signal: AbortSignal.timeout(60_000) });
  const finalMessage = await stream.finalMessage();
  assertWeatherTool(finalMessage);
} catch (e) {
  failed(e);
}

console.log(`\n=== Anthropic SDK Tests Complete: ${failures} failed ===`);
Deno.exitCode = failures > 0 ? 1 : 0;
