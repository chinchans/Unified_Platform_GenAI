(() => {
  function classifyDiffLine(line) {
    const value = String(line || "");
    if (
      value.startsWith("diff --git") ||
      value.startsWith("---") ||
      value.startsWith("+++") ||
      value.startsWith("@@")
    ) {
      return "diff-meta";
    }
    if (value.startsWith("+")) return "diff-addition";
    if (value.startsWith("-")) return "diff-deletion";
    return "diff-neutral";
  }

  function createNdjsonAccumulator(onMalformed) {
    let buffer = "";
    return {
      pushChunk(chunkText) {
        const parsedEvents = [];
        buffer += String(chunkText || "");
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";
        lines.forEach((line) => {
          const trimmed = line.trim();
          if (!trimmed) return;
          try {
            parsedEvents.push(JSON.parse(trimmed));
          } catch (_error) {
            if (typeof onMalformed === "function") {
              onMalformed(trimmed);
            }
          }
        });
        return parsedEvents;
      },
      flush() {
        const trailing = buffer.trim();
        buffer = "";
        if (!trailing) return [];
        try {
          return [JSON.parse(trailing)];
        } catch (_error) {
          if (typeof onMalformed === "function") {
            onMalformed(trailing);
          }
          return [];
        }
      }
    };
  }

  const api = { classifyDiffLine, createNdjsonAccumulator };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof window !== "undefined") {
    window.StreamUtils = api;
  }
})();
