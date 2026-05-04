const test = require("node:test");
const assert = require("node:assert/strict");

const { classifyDiffLine, createNdjsonAccumulator } = require("./stream-utils.js");

test("classifyDiffLine marks meta/add/remove/neutral", () => {
  assert.equal(classifyDiffLine("diff --git a/a b/a"), "diff-meta");
  assert.equal(classifyDiffLine("@@ -1,2 +1,3 @@"), "diff-meta");
  assert.equal(classifyDiffLine("+added"), "diff-addition");
  assert.equal(classifyDiffLine("-removed"), "diff-deletion");
  assert.equal(classifyDiffLine(" context"), "diff-neutral");
});

test("createNdjsonAccumulator parses fragmented chunks and skips malformed lines", () => {
  const malformed = [];
  const decoder = createNdjsonAccumulator((raw) => malformed.push(raw));

  const first = decoder.pushChunk('{"type":"log","text":"one"}\n{"type":"log"');
  assert.equal(first.length, 1);
  assert.equal(first[0].text, "one");

  const second = decoder.pushChunk(',"text":"two"}\nnot-json\n');
  assert.equal(second.length, 1);
  assert.equal(second[0].text, "two");
  assert.equal(malformed.length, 1);

  const trailing = decoder.pushChunk('{"type":"result","data":{"ok":true}}');
  assert.equal(trailing.length, 0);
  const flushed = decoder.flush();
  assert.equal(flushed.length, 1);
  assert.equal(flushed[0].type, "result");
});
