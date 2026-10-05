#!/usr/bin/env -S deno run --allow-net --allow-env

import Anthropic from "npm:@anthropic-ai/sdk@^0.39.0";

const PROXY_URL = "http://127.0.0.1:4000";
const API_KEY = Deno.env.get("OXLLM_MASTER_KEY") || "sk-ox-local-smoke-master";
const client = new Anthropic({ baseURL: PROXY_URL, apiKey: API_KEY });

let passCount = 0, failCount = 0;
const ok = (m: string) => { console.log(`   ✓ ${m}`); passCount++; };
const fail = (m: string, e?: string) => { console.error(`   ✗ ${m}${e ? ": " + e : ""}`); failCount++; };

console.log("=== Thinking + Prompt Caching Tests ===\n");

// Large context for caching (>1024 tokens)
const CONTEXT = `You are an expert on the Apache License 2.0. Here is the full license text:

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions. "License" shall mean the terms and conditions for use, reproduction, and distribution. "Licensor" shall mean the copyright owner or entity authorized by the copyright owner. "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. "You" shall mean an individual or Legal Entity exercising permissions granted by this License. "Source" form shall mean the preferred form for making modifications. "Object" form shall mean any form resulting from mechanical transformation. "Work" shall mean the work of authorship. "Derivative Works" shall mean any work that is based on or derived from the Work.

2. Grant of Copyright License. Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work.

3. Grant of Patent License. Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work.

4. Redistribution. You may reproduce and distribute copies provided that: (a) You give recipients a copy of this License; (b) You cause modified files to carry prominent notices; (c) You retain all copyright, patent, trademark, and attribution notices.

5. Submission of Contributions. Any Contribution intentionally submitted shall be under the terms of this License.

6. Trademarks. This License does not grant permission to use trade names or trademarks.

7. Disclaimer of Warranty. Licensor provides the Work on an AS IS BASIS, WITHOUT WARRANTIES.

8. Limitation of Liability. In no event shall Licensor be liable for damages.

9. Accepting Warranty. While redistributing, You may choose to offer support or warranty for a fee.

APPENDIX: How to apply the Apache License to your work: Attach the following boilerplate notice, with the fields enclosed by brackets "[]" replaced with your own identifying information. (Don't include the brackets!) The text should be enclosed in the appropriate comment syntax for the file format.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.`;

// ============================================================
// THINKING TESTS
// ============================================================
console.log("--- Thinking Tests ---\n");

// Test 1: Adaptive thinking (correct format for Sonnet 5 - no budget_tokens)
console.log("1. Adaptive thinking (complex reasoning)...");
try {
  const resp = await client.messages.create({
    model: "claude-sonnet",
    max_tokens: 12000,
    thinking: { type: "adaptive" },
    messages: [{ role: "user", content: "I have 3 boxes. Box A has 2 red balls. Box B has 3 blue balls. Box C has 4 green balls. If I randomly pick one ball from each box, what are the odds all three are the same color? Think step by step." }],
  } as any);

  const thinkingBlock = resp.content.find((b: any) => b.type === "thinking");
  const textBlock = resp.content.find((b: any) => b.type === "text");
  ok(`Response has ${resp.content.length} content blocks`);
  ok(`Stop reason: ${resp.stop_reason}`);
  ok(`Usage: input=${resp.usage.input_tokens}, output=${resp.usage.output_tokens}`);
  if (thinkingBlock) {
    ok(`Thinking block present (${String(thinkingBlock.thinking).length} chars)`);
  } else {
    ok("No thinking block (model chose not to think — valid for adaptive)");
  }
  if (textBlock) ok(`Answer: ${textBlock.text?.slice(0, 100)}`);
} catch (e) { fail("Failed", e.message); }

// Test 2: Thinking streaming
console.log("\n2. Streaming with thinking...");
try {
  const stream = client.messages.stream({
    model: "claude-sonnet",
    max_tokens: 12000,
    thinking: { type: "adaptive" },
    messages: [{ role: "user", content: "Design a simple SQL schema for a blog with posts and comments. Think about normalization." }],
  } as any);

  let thinkingChars = 0, textChars = 0;
  for await (const event of stream) {
    if (event.type === "content_block_delta") {
      if (event.delta?.type === "thinking_delta") thinkingChars += event.delta.thinking?.length || 0;
      if (event.delta?.type === "text_delta") textChars += event.delta.text?.length || 0;
    }
  }
  ok(`Thinking: ${thinkingChars} chars, Text: ${textChars} chars`);
} catch (e) { fail("Stream failed", e.message); }

// Test 3: Thinking + tool use
console.log("\n3. Thinking + tool use...");
try {
  const resp = await client.messages.create({
    model: "claude-sonnet",
    max_tokens: 16000,
    thinking: { type: "adaptive" },
    messages: [{ role: "user", content: "What's the weather in Tokyo and London? Use the get_weather tool for each city." }],
    tools: [{ name: "get_weather", description: "Get weather", input_schema: { type: "object" as const, properties: { city: { type: "string" } }, required: ["city"] } }],
  } as any);
  const toolUses = resp.content.filter((b: any) => b.type === "tool_use");
  ok(`Stop reason: ${resp.stop_reason}`);
  ok(`Tool calls: ${toolUses.length}`);
  for (const tu of toolUses) ok(`  → ${tu.name}(${JSON.stringify(tu.input)})`);
} catch (e) { fail("Failed", e.message); }

// ============================================================
// PROMPT CACHING TESTS
// ============================================================
console.log("\n--- Prompt Caching Tests ---\n");

// Test 4: Cache via /v1/chat/completions (proven to work)
console.log("4. Cache write via /v1/chat/completions...");
try {
  const r1 = await fetch(`${PROXY_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet", max_tokens: 30,
      messages: [{ role: "user", content: [
        { type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } },
        { type: "text", text: "What license version? One word." },
      ]}],
    }),
  });
  const d1 = await r1.json();
  ok(`Response: ${d1.choices?.[0]?.message?.content}`);
  ok(`Cache creation: ${d1.usage?.cache_creation_input_tokens ?? 0}`);

  const r2 = await fetch(`${PROXY_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet", max_tokens: 30,
      messages: [{ role: "user", content: [
        { type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } },
        { type: "text", text: "Summarize in 3 words." },
      ]}],
    }),
  });
  const d2 = await r2.json();
  ok(`Response: ${d2.choices?.[0]?.message?.content}`);
  const cr = d2.usage?.cache_read_input_tokens ?? 0;
  ok(`Cache read: ${cr}`);
  if (cr > 0) ok("✓ CACHE HIT via /v1/chat/completions");
  else fail("Cache miss via /v1/chat/completions");
} catch (e) { fail("Failed", e.message); }

// Test 5: Cache via /v1/messages (Anthropic native)
console.log("\n5. Cache via /v1/messages (Anthropic SDK)...");
try {
  const r1 = await client.messages.create({
    model: "claude-sonnet", max_tokens: 30,
    system: [{ type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } }],
    messages: [{ role: "user", content: "What license version? One word." }],
  } as any);
  ok(`Response: ${r1.content.find((b: any) => b.type === "text")?.text}`);
  ok(`Cache creation: ${(r1.usage as any).cache_creation_input_tokens ?? 0}`);

  const r2 = await client.messages.create({
    model: "claude-sonnet", max_tokens: 30,
    system: [{ type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } }],
    messages: [{ role: "user", content: "Is it OSI approved? Yes or no." }],
  } as any);
  ok(`Response: ${r2.content.find((b: any) => b.type === "text")?.text}`);
  const cr = (r2.usage as any).cache_read_input_tokens ?? 0;
  ok(`Cache read: ${cr}`);
  if (cr > 0) ok("✓ CACHE HIT via /v1/messages");
  else fail("Cache miss via /v1/messages");
} catch (e) { fail("Failed", e.message); }

// Test 6: Cache via /v1/messages with user message content
console.log("\n6. Cache on user message content via /v1/messages...");
try {
  const r1 = await client.messages.create({
    model: "claude-sonnet", max_tokens: 30,
    messages: [{ role: "user", content: [
      { type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } },
      { type: "text", text: "Remember this. Say OK." },
    ]}],
  } as any);
  ok(`First: ${r1.content.find((b: any) => b.type === "text")?.text?.slice(0, 20)}`);

  const r2 = await client.messages.create({
    model: "claude-sonnet", max_tokens: 30,
    messages: [{ role: "user", content: [
      { type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } },
      { type: "text", text: "What license? One word." },
    ]}],
  } as any);
  ok(`Second: ${r2.content.find((b: any) => b.type === "text")?.text}`);
  const cr = (r2.usage as any).cache_read_input_tokens ?? 0;
  ok(`Cache read: ${cr}`);
  if (cr > 0) ok("✓ CACHE HIT on user message");
  else fail("Cache miss on user message");
} catch (e) { fail("Failed", e.message); }

console.log(`\n=== Results: ${passCount} passed, ${failCount} failed ===`);
if (failCount > 0) Deno.exit(1);
