const apiBaseUrl = window.platformConfig.apiBaseUrl;

const state = {
  sessionId: null,
  promptReady: false,
  codeReady: false,
  awaitingAmbiguityResolution: false,
  pollTimer: null,
  lastLogCount: 0,
  promptDisplayed: false,
  codeDisplayed: false,
  promptGroupExpanded: false,
  jenkinsGroupExpanded: false,
  promptContainer: null,
  promptTextNode: null,
  promptEditBtn: null,
  promptCancelBtn: null,
  generationInProgress: false,
  reviewInProgress: false,
  commitInProgress: false,
  pushInProgress: false,
  currentMilestones: [],
  gitHistoryLoaded: false,
  selectedHistoryBranch: "",
  selectedHistoryCommit: "",
  gitCommitLimit: 10,
  gitRenderedCommitCount: 0,
  gitBranchUpstreams: {},
  selectedBranchUpstream: "",
  gitDefaultRemote: "origin",
  selectedBranchUpstreamHead: "",
  cursorOutputContainer: null,
  cursorOutputBody: null,
  lastLogSignature: "",
  pipelineStarted: false
};

const splashScreen = document.getElementById("splash-screen");
const mainScreen = document.getElementById("main-screen");
const settingsScreen = document.getElementById("settings-screen");
const settingsBackBtn = document.getElementById("settings-back-btn");
const enterPlatformBtn = document.getElementById("enter-platform-btn");

const intentInput = document.getElementById("intent-input");
const primaryActionBtn = document.getElementById("primary-action-btn");
const generateCodeBtn = document.getElementById("generate-code-btn");
const commitBtn = document.getElementById("commit-btn");
const pushBtn = document.getElementById("push-btn");
const commitMessageGroup = document.getElementById("commit-message-group");
const commitMessageInput = document.getElementById("commit-message-input");
const acceptBtn = document.getElementById("accept-btn");
const rejectBtn = document.getElementById("reject-btn");
const reviewActions = document.getElementById("review-actions");
const milestoneList = document.getElementById("milestone-list");
const logStream = document.getElementById("log-stream");
const ambiguityCallout = document.getElementById("self-learning-ambiguity-callout");
const ambiguityItemList = document.getElementById("ambiguity-item-list");
const runningIndicator = document.getElementById("running-indicator");
const runningIndicatorText = document.getElementById("running-indicator-text");
const gitHistoryScreen = document.getElementById("git-history-screen");
const openGitHistoryBtn = document.getElementById("open-git-history-btn");
const closeGitHistoryBtn = document.getElementById("close-git-history-btn");
const gitBranchSelect = document.getElementById("git-branch-select");
const gitRefreshBtn = document.getElementById("git-refresh-btn");
const gitCommitTree = document.getElementById("git-commit-tree");
const gitCommitDetail = document.getElementById("git-commit-detail");
const settingsOpenButtons = document.querySelectorAll(".titlebar-settings-btn");
const pipelineTrack = document.getElementById("pipeline-track");
const pipelineEmptyState = document.getElementById("pipeline-empty-state");
const streamUtils = window.StreamUtils || {};
const CHEVRON_DOWN_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-down" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708"/>
</svg>`;
const CHEVRON_UP_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-up" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M7.646 4.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1-.708.708L8 5.707l-5.646 5.647a.5.5 0 0 1-.708-.708z"/>
</svg>`;

function setReviewActionsVisible(visible) {
  reviewActions.classList.toggle("hidden", !visible);
  refreshActionButtons();
}

function setCommitMessageVisible(visible, suggestedMessage = "") {
  commitMessageGroup.classList.toggle("hidden", !visible);
  if (!visible) {
    commitMessageInput.value = "";
    refreshActionButtons();
    return;
  }
  if (suggestedMessage && !commitMessageInput.value.trim()) {
    commitMessageInput.value = suggestedMessage;
  }
  refreshActionButtons();
}

function refreshActionButtons() {
  const isBusy =
    state.generationInProgress ||
    state.reviewInProgress ||
    state.commitInProgress ||
    state.pushInProgress;

  const canGeneratePrompt =
    !isBusy && (!state.sessionId || state.awaitingAmbiguityResolution) && !state.promptReady;

  const canGenerateCode =
    !isBusy &&
    state.promptReady &&
    !state.codeReady &&
    !generateCodeBtn.classList.contains("hidden");

  const canReview = !isBusy && state.codeReady && !reviewActions.classList.contains("hidden");

  const canCommit =
    !isBusy &&
    !commitBtn.classList.contains("hidden") &&
    !commitMessageGroup.classList.contains("hidden");

  const canPush = !isBusy && !pushBtn.classList.contains("hidden");

  primaryActionBtn.disabled = !canGeneratePrompt;
  generateCodeBtn.disabled = !canGenerateCode;
  acceptBtn.disabled = !canReview;
  rejectBtn.disabled = !canReview;
  commitBtn.disabled = !canCommit;
  pushBtn.disabled = !canPush;
}

function collectAmbiguityItems(session) {
  if (!session) return [];
  const raw =
    session.ambiguity_items ??
    session.ambiguityItems ??
    session.ambiguity_questions ??
    session.self_learning_ambiguities;
  if (Array.isArray(raw) && raw.length) {
    return raw
      .map((entry) =>
        typeof entry === "string"
          ? entry
          : entry?.text ??
              entry?.question ??
              entry?.message ??
              entry?.description ??
              ""
      )
      .filter(Boolean);
  }
  if (session.ambiguityQuestion) return [session.ambiguityQuestion];
  return [];
}

/** Open points from self-learning (not a milestone): shown before final prompt composition. */
function updateSelfLearningAmbiguityPanel(session) {
  const items = collectAmbiguityItems(session);
  if (!items.length) {
    ambiguityCallout.classList.add("hidden");
    ambiguityItemList.innerHTML = "";
    return;
  }
  ambiguityCallout.classList.remove("hidden");
  ambiguityItemList.innerHTML = "";
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    ambiguityItemList.appendChild(li);
  });
}

function getMilestoneStatusMap(milestones) {
  return milestones.reduce((acc, item) => {
    acc[item.id] = item.status;
    return acc;
  }, {});
}

function aggregateStatus(statuses) {
  if (statuses.some((s) => s === "failed")) return "failed";
  if (statuses.some((s) => s === "in_progress")) return "in_progress";
  if (statuses.every((s) => s === "completed")) return "completed";
  return "not_completed";
}

function getInitialMilestones() {
  return [
    { id: "feature_validation", status: "not_completed" },
    { id: "knowledge_retrieval", status: "not_completed" },
    { id: "template_orchestrator", status: "not_completed" },
    { id: "self_learning_agent", status: "not_completed" },
    { id: "prompt_generation", status: "not_completed" },
    { id: "code_generation", status: "not_completed" },
    { id: "commit_push", status: "not_completed" },
    { id: "jenkins_checkout", status: "not_completed" },
    { id: "test_script_generation", status: "not_completed" },
    { id: "build_compile", status: "not_completed" },
    { id: "rca_build_fix", status: "not_completed" },
    { id: "runtime_execute", status: "not_completed" },
    { id: "rca_runtime_fix", status: "not_completed" },
    { id: "test_scoring", status: "not_completed" }
  ];
}

function setPipelineVisibility(visible) {
  state.pipelineStarted = visible;
  if (pipelineTrack) pipelineTrack.classList.toggle("hidden", !visible);
  if (pipelineEmptyState) pipelineEmptyState.classList.toggle("hidden", visible);
}

function renderPipelinePhases(milestones) {
  if (!pipelineTrack) return;
  if (!state.pipelineStarted) {
    setPipelineVisibility(false);
    return;
  }
  const statusById = getMilestoneStatusMap(milestones || []);
  const resolve = (ids) => aggregateStatus(ids.map((id) => statusById[id] || "not_completed"));

  const phases = [
    {
      title: "Intent Decomposition",
      subtitle: "Requirement integrity and expansion checks",
      status: resolve(["feature_validation", "knowledge_retrieval", "template_orchestrator", "self_learning_agent"])
    },
    { title: "Prompt Generation", status: resolve(["prompt_generation"]) },
    { title: "Code Generation", status: resolve(["code_generation"]) },
    { title: "Review & Commit", status: resolve(["commit_push"]) },
    {
      title: "CI/CD Pipeline",
      status: resolve(["jenkins_checkout", "test_script_generation", "build_compile", "runtime_execute"])
    },
    { title: "Test Execution", status: resolve(["test_scoring"]) },
    { title: "RCA Fix Loop", status: resolve(["rca_build_fix", "rca_runtime_fix"]) }
  ];
  phases.push({
    title: "Done",
    status: aggregateStatus(phases.map((phase) => phase.status))
  });

  const inProgressIndex = phases.findIndex((phase) => phase.status === "in_progress");
  const notCompletedIndex = phases.findIndex((phase) => phase.status === "not_completed");
  const activeIndex = inProgressIndex >= 0 ? inProgressIndex : notCompletedIndex >= 0 ? notCompletedIndex : phases.length - 1;

  pipelineTrack.innerHTML = "";
  phases.forEach((phase, index) => {
    const item = document.createElement("div");
    item.className = "pipeline-item";
    if (index === activeIndex && phase.status !== "completed" && phase.status !== "failed") {
      item.classList.add("active");
    }
    if (phase.status === "completed") item.classList.add("completed");
    if (phase.subtitle) {
      item.innerHTML = `<div class="pipeline-title">${phase.title}</div><div class="pipeline-subtitle">${phase.subtitle}</div>`;
    } else {
      item.textContent = phase.title;
    }
    pipelineTrack.appendChild(item);
  });
}

function createMilestoneNode(label, status, isLast = false, extraClass = "") {
  const li = document.createElement("li");
  li.className = `milestone-item ${extraClass}`.trim();
  if (isLast) li.classList.add("is-last");
  li.innerHTML = `
    <span class="milestone-visual">
      <span class="status-ring status-${status}">
        <span class="status-inner"></span>
      </span>
      <span class="status-connector"></span>
    </span>
    <span class="milestone-label">${label}</span>
  `;
  return li;
}

function renderMilestones(milestones) {
  if (!Array.isArray(milestones)) return;
  state.currentMilestones = milestones.map((item) => ({ ...item }));
  updateRunningIndicator();
  renderPipelinePhases(state.currentMilestones);
  milestoneList.innerHTML = "";
  const statusById = getMilestoneStatusMap(milestones);

  const promptChildren = [
    { id: "feature_validation", label: "Intent Resolution" },
    { id: "knowledge_retrieval", label: "Knowledge Retrieval" },
    { id: "template_orchestrator", label: "Template Orchestrator" },
    { id: "self_learning_agent", label: "Self Learning Agent" },
    { id: "prompt_generation", label: "Generate Prompt" }
  ];

  const promptGroupStatus = aggregateStatus(
    promptChildren.map((c) => statusById[c.id] || "not_completed")
  );

  const promptParent = createMilestoneNode("Prompt Generation", promptGroupStatus, false, "parent");
  promptParent.classList.toggle("expanded", state.promptGroupExpanded);
  promptParent.classList.toggle("collapsed", !state.promptGroupExpanded);
  const promptLabel = promptParent.querySelector(".milestone-label");
  const labelText = promptLabel.textContent;
  promptLabel.textContent = "";
  promptLabel.classList.add("is-expandable");
  const textSpan = document.createElement("span");
  textSpan.className = "milestone-text";
  textSpan.textContent = labelText;
  const toggle = document.createElement("span");
  toggle.className = "milestone-toggle-indicator";
  toggle.textContent = state.promptGroupExpanded ? "▲" : "▼";
  promptLabel.append(textSpan, toggle);
  milestoneList.appendChild(promptParent);

  const childrenRow = document.createElement("li");
  childrenRow.className = `milestone-children-row ${state.promptGroupExpanded ? "expanded" : "collapsed"}`;
  const childrenList = document.createElement("ul");
  childrenList.className = "milestone-children-list";

  promptChildren.forEach((child, index) => {
    const childNode = createMilestoneNode(
      child.label,
      statusById[child.id] || "not_completed",
      index === promptChildren.length - 1,
      "child"
    );
    childrenList.appendChild(childNode);
  });

  childrenRow.appendChild(childrenList);
  milestoneList.appendChild(childrenRow);
  promptLabel.addEventListener("click", () => {
    state.promptGroupExpanded = !state.promptGroupExpanded;
    childrenRow.classList.toggle("collapsed", !state.promptGroupExpanded);
    childrenRow.classList.toggle("expanded", state.promptGroupExpanded);
    promptParent.classList.toggle("collapsed", !state.promptGroupExpanded);
    promptParent.classList.toggle("expanded", state.promptGroupExpanded);
    toggle.textContent = state.promptGroupExpanded ? "▲" : "▼";
  });

  const codeNode = createMilestoneNode(
    "Code Generation",
    statusById.code_generation || "not_completed",
    false,
    "parent"
  );
  milestoneList.appendChild(codeNode);

  const commitNode = createMilestoneNode(
    "Commit & Push",
    statusById.commit_push || "not_completed",
    false,
    "parent"
  );
  milestoneList.appendChild(commitNode);

  const jenkinsChildren = [
    { id: "jenkins_checkout", label: "Checkout Branch" },
    { id: "test_script_generation", label: "Test Script Generation" },
    { id: "build_compile", label: "Build / Compile" },
    { id: "rca_build_fix", label: "RCA Build Fix Loop" },
    { id: "runtime_execute", label: "Run / Execute" },
    { id: "rca_runtime_fix", label: "RCA Runtime Fix Loop" },
    { id: "test_scoring", label: "Test Execution & Scoring" }
  ];

  const jenkinsGroupStatus = aggregateStatus(
    jenkinsChildren.map((c) => statusById[c.id] || "not_completed")
  );
  const jenkinsParent = createMilestoneNode("CI/CD Pipeline", jenkinsGroupStatus, true, "parent");
  jenkinsParent.classList.toggle("expanded", state.jenkinsGroupExpanded);
  jenkinsParent.classList.toggle("collapsed", !state.jenkinsGroupExpanded);
  const jenkinsLabel = jenkinsParent.querySelector(".milestone-label");
  const jenkinsText = jenkinsLabel.textContent;
  jenkinsLabel.textContent = "";
  jenkinsLabel.classList.add("is-expandable");
  const jenkinsTextSpan = document.createElement("span");
  jenkinsTextSpan.className = "milestone-text";
  jenkinsTextSpan.textContent = jenkinsText;
  const jenkinsToggle = document.createElement("span");
  jenkinsToggle.className = "milestone-toggle-indicator";
  jenkinsToggle.textContent = state.jenkinsGroupExpanded ? "▲" : "▼";
  jenkinsLabel.append(jenkinsTextSpan, jenkinsToggle);
  milestoneList.appendChild(jenkinsParent);

  const jenkinsChildrenRow = document.createElement("li");
  jenkinsChildrenRow.className = `milestone-children-row ${state.jenkinsGroupExpanded ? "expanded" : "collapsed"}`;
  const jenkinsChildrenList = document.createElement("ul");
  jenkinsChildrenList.className = "milestone-children-list";
  jenkinsChildren.forEach((child, index) => {
    const childNode = createMilestoneNode(
      child.label,
      statusById[child.id] || "not_completed",
      index === jenkinsChildren.length - 1,
      "child"
    );
    jenkinsChildrenList.appendChild(childNode);
  });
  jenkinsChildrenRow.appendChild(jenkinsChildrenList);
  milestoneList.appendChild(jenkinsChildrenRow);

  jenkinsLabel.addEventListener("click", () => {
    state.jenkinsGroupExpanded = !state.jenkinsGroupExpanded;
    jenkinsChildrenRow.classList.toggle("collapsed", !state.jenkinsGroupExpanded);
    jenkinsChildrenRow.classList.toggle("expanded", state.jenkinsGroupExpanded);
    jenkinsParent.classList.toggle("collapsed", !state.jenkinsGroupExpanded);
    jenkinsParent.classList.toggle("expanded", state.jenkinsGroupExpanded);
    jenkinsToggle.textContent = state.jenkinsGroupExpanded ? "▲" : "▼";
  });
}

function statusRank(status) {
  const order = {
    not_completed: 0,
    in_progress: 1,
    completed: 2,
    failed: 3
  };
  return order[status] ?? 0;
}

function mergeStatus(current, incoming) {
  return statusRank(incoming) >= statusRank(current) ? incoming : current;
}

function updateMilestoneStatusLocal(milestoneId, status) {
  if (!Array.isArray(state.currentMilestones) || !state.currentMilestones.length) return;
  const nextMilestones = state.currentMilestones.map((item) => {
    if (item.id !== milestoneId) return item;
    return {
      ...item,
      status: mergeStatus(item.status || "not_completed", status)
    };
  });
  renderMilestones(nextMilestones);
}

function logLevelBadgeLabel(level) {
  switch (level) {
    case "warning":
      return "WARN";
    case "error":
      return "ERR";
    case "success":
      return "OK";
    default:
      return "INFO";
  }
}

/**
 * Renders a single log row with a level badge, optional [stage] tag, and message (colored in CSS).
 */
function buildStructuredLogLine(text, level = "info") {
  const row = document.createElement("div");
  row.className = `log-line log-level-${level}`;

  const raw = String(text ?? "");
  const sysMatch = raw.match(/^SYS\s+(.*)$/);
  const mainText = sysMatch ? sysMatch[1] : raw;

  const badge = document.createElement("span");
  if (sysMatch) {
    badge.className = "log-line-badge log-line-badge-sys";
    badge.textContent = "SYS";
  } else {
    badge.className = `log-line-badge log-line-badge-${level}`;
    badge.textContent = logLevelBadgeLabel(level);
  }

  const content = document.createElement("span");
  content.className = "log-line-content";

  const stageMatch = mainText.match(/^\[([^\]]+)\]\s*(.*)$/s);
  if (stageMatch) {
    const stage = document.createElement("span");
    stage.className = "log-line-stage";
    stage.textContent = `[${stageMatch[1]}]`;
    const msg = document.createElement("span");
    msg.className = "log-line-msg";
    const rest = stageMatch[2] ?? "";
    msg.textContent = rest.length ? ` ${rest}` : "";
    content.appendChild(stage);
    content.appendChild(msg);
  } else {
    const msg = document.createElement("span");
    msg.className = "log-line-msg";
    msg.textContent = mainText;
    content.appendChild(msg);
  }

  row.appendChild(badge);
  row.appendChild(content);
  return row;
}

function appendLog(text, type = "info") {
  const signature = `${type}::${String(text)}`;
  if (state.lastLogSignature === signature) return;
  state.lastLogSignature = signature;
  const div = buildStructuredLogLine(String(text), type);
  div.classList.add("log-item");
  logStream.appendChild(div);
  logStream.scrollTop = logStream.scrollHeight;
}

function renderInitialLogView() {
  if (!logStream || logStream.children.length > 0) return;
  const welcome = document.createElement("div");
  welcome.className = "log-item welcome-title";
  welcome.textContent = "WELCOME";
  logStream.appendChild(welcome);

  const ready = buildStructuredLogLine(
    "SYS  Platform ready. Enter your intent above and click Generate Prompt.",
    "info"
  );
  ready.classList.add("log-item", "welcome-line");
  logStream.appendChild(ready);
}

function appendCodePreview(previewText) {
  const wrapper = document.createElement("div");
  wrapper.className = "log-item code-preview";

  const lines = String(previewText || "").split("\n");
  lines.forEach((line) => {
    const lineDiv = document.createElement("div");
    lineDiv.className = "code-preview-line";
    if (line.startsWith("+")) {
      lineDiv.classList.add("diff-addition");
    } else if (line.startsWith("-")) {
      lineDiv.classList.add("diff-deletion");
    } else {
      lineDiv.classList.add("diff-neutral");
    }
    lineDiv.textContent = line;
    wrapper.appendChild(lineDiv);
  });

  logStream.appendChild(wrapper);
  logStream.scrollTop = logStream.scrollHeight;
}

function appendBackendLogs(logs = []) {
  logs.forEach((log) => {
    const rawMessage = String(log?.message || "");
    if (isCursorCliTranscriptLog(rawMessage)) {
      return;
    }
    const logType =
      log.type === "warning" ? "warning" : log.type === "error" ? "error" : "info";
    appendLog(`[${log.stage}] ${rawMessage}`, logType);
  });
}

function appendSectionHeader(title) {
  const target = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : logStream;
  const header = document.createElement("div");
  header.className = "log-item section-title";
  header.textContent = title;
  target.appendChild(header);
}

function ensureCursorOutputContainer() {
  if (state.cursorOutputContainer && state.cursorOutputBody) {
    return state.cursorOutputBody;
  }

  const container = document.createElement("div");
  container.className = "log-item cursor-output-box";
  container.innerHTML = `
    <div class="cursor-output-box-header">Cursor CLI Output</div>
    <div class="cursor-output-box-body"></div>
  `;

  const body = container.querySelector(".cursor-output-box-body");
  state.cursorOutputContainer = container;
  state.cursorOutputBody = body;
  logStream.appendChild(container);
  return body;
}

function clearCursorOutputContainer() {
  if (state.cursorOutputBody) {
    state.cursorOutputBody.innerHTML = "";
    return;
  }
  ensureCursorOutputContainer();
}

function isCursorCliTranscriptLog(text) {
  const raw = String(text || "").trim();
  const normalized = raw.toLowerCase();
  if (!normalized) return false;
  if (normalized.startsWith("starting cursor cli generation run.")) {
    return true;
  }
  if (
    normalized.startsWith("## ") ||
    normalized.startsWith("### ") ||
    normalized.startsWith("- ") ||
    normalized.startsWith("1. ") ||
    normalized.startsWith("2. ") ||
    normalized.startsWith("3. ") ||
    normalized.startsWith("4. ")
  ) {
    return true;
  }
  if (raw.includes("`") && (normalized.includes(".h") || normalized.includes(".c") || normalized.includes(".txt"))) {
    return true;
  }
  if (normalized.includes("here is what was implemented from `.unified_codegen_prompt.txt`")) {
    return true;
  }
  if (normalized.includes("### code changes") || normalized.includes("### required reports")) {
    return true;
  }
  if (normalized.includes("dependency_closure_report") || normalized.includes("coverage_gate_report")) {
    return true;
  }
  if (normalized.includes("note:** a full build was not run")) {
    return true;
  }
  return (
    normalized.startsWith("here is what was implemented") ||
    normalized.startsWith("### code changes") ||
    normalized.startsWith("### required reports") ||
    normalized.startsWith("### summary") ||
    normalized === "cursor text output"
  );
}

function appendOutputNote(text, target) {
  const note = document.createElement("div");
  note.className = "cursor-output-note";
  note.textContent = text;
  target.appendChild(note);
}

function appendCursorChatOutput(chatOutput) {
  const outputBody = ensureCursorOutputContainer();
  appendSectionHeader("Code generation chat output", outputBody);
  const block = document.createElement("pre");
  block.className = "log-item cursor-chat-output";
  block.textContent = String(chatOutput || "").trim() || "No output from Cursor CLI.";
  outputBody.appendChild(block);
  logStream.scrollTop = logStream.scrollHeight;
}

function appendCodeChanges(codeChanges, changedFiles) {
  const outputBody = ensureCursorOutputContainer();
  appendSectionHeader("Code changes by file", outputBody);
  if (!Array.isArray(codeChanges) || !codeChanges.length) {
    const untrackedCount = Array.isArray(changedFiles?.untracked_paths)
      ? changedFiles.untracked_paths.length
      : 0;
    if (untrackedCount > 0) {
      appendOutputNote(
        `No tracked file diffs were produced. ${untrackedCount} untracked file(s) were created.`,
        outputBody
      );
    } else {
      appendOutputNote(
        "No tracked file diffs were produced (chat output only, no code edits).",
        outputBody
      );
    }
    logStream.scrollTop = logStream.scrollHeight;
    return;
  }
  codeChanges.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "log-item code-change-card";

    const header = document.createElement("div");
    header.className = "code-change-header";
    const ins = Number(entry?.insertions || 0);
    const del = Number(entry?.deletions || 0);
    header.textContent = `${entry?.path || "unknown"} (+${ins} / -${del})`;
    card.appendChild(header);

    const diffBox = document.createElement("div");
    diffBox.className = "code-change-diff";
    const lines = String(entry?.diff || "").split("\n");
    if (!String(entry?.diff || "").trim()) {
      const empty = document.createElement("div");
      empty.className = "code-preview-line diff-neutral";
      empty.textContent = "(empty diff)";
      diffBox.appendChild(empty);
    } else {
      lines.forEach((line) => {
        const lineDiv = document.createElement("div");
        const lineStyle = streamUtils.classifyDiffLine
          ? streamUtils.classifyDiffLine(line)
          : "diff-neutral";
        lineDiv.className = `code-preview-line ${lineStyle}`;
        lineDiv.textContent = line;
        diffBox.appendChild(lineDiv);
      });
    }
    card.appendChild(diffBox);
    outputBody.appendChild(card);
  });
  logStream.scrollTop = logStream.scrollHeight;
}

function isMilestoneRunning() {
  return (state.currentMilestones || []).some((item) => item.status === "in_progress");
}

const STAGE_RUNNING_TEXT = {
  feature_validation: "Prompt generation in progress...",
  knowledge_retrieval: "Knowledge retrieval in progress...",
  template_orchestrator: "Template orchestration in progress...",
  self_learning_agent: "Self-learning analysis in progress...",
  prompt_generation: "Prompt generation in progress...",
  code_generation: "Code generation in progress...",
  commit_push: "Commit and push in progress...",
  jenkins_checkout: "Pipeline: checkout in progress...",
  test_script_generation: "Pipeline: test script generation in progress...",
  build_compile: "Pipeline: build/compile in progress...",
  rca_build_fix: "Pipeline: build RCA fix loop in progress...",
  runtime_execute: "Pipeline: runtime execution in progress...",
  rca_runtime_fix: "Pipeline: runtime RCA fix loop in progress...",
  test_scoring: "Pipeline: test scoring in progress..."
};

function getRunningStageMessage() {
  const runningMilestone = (state.currentMilestones || []).find((item) => item.status === "in_progress");
  if (!runningMilestone) return "";
  return STAGE_RUNNING_TEXT[runningMilestone.id] || "Pipeline in progress...";
}

function updateRunningIndicator() {
  if (!runningIndicator) return;
  const isRunning =
    isMilestoneRunning() || state.generationInProgress || state.reviewInProgress;
  runningIndicator.classList.toggle("hidden", !isRunning);
  if (!runningIndicatorText) return;
  if (state.generationInProgress) {
    runningIndicatorText.textContent = "Code generation in progress...";
    return;
  }
  if (state.reviewInProgress) {
    runningIndicatorText.textContent = "Applying review action...";
    return;
  }
  runningIndicatorText.textContent = getRunningStageMessage();
}

async function readNdjsonStream(response, onEvent) {
  if (!response.body) throw new Error("Streaming response body is unavailable.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  if (!streamUtils.createNdjsonAccumulator) {
    throw new Error("NDJSON stream decoder is unavailable.");
  }
  const accumulator = streamUtils.createNdjsonAccumulator((raw) => {
    appendLog(`Malformed stream event ignored: ${raw.slice(0, 120)}`, "warning");
  });
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunkText = decoder.decode(value, { stream: true });
    const events = accumulator.pushChunk(chunkText);
    events.forEach(onEvent);
  }
  const trailing = decoder.decode();
  const events = accumulator.pushChunk(trailing).concat(accumulator.flush());
  events.forEach(onEvent);
}

function setCodeGenerationBusy(isBusy) {
  state.generationInProgress = isBusy;
  refreshActionButtons();
  updateRunningIndicator();
}

function setReviewBusy(isBusy) {
  state.reviewInProgress = isBusy;
  refreshActionButtons();
  updateRunningIndicator();
}

function ensurePromptContainer() {
  if (state.promptContainer) return state.promptContainer;

  const container = document.createElement("div");
  container.className = "log-item prompt-output";
  container.innerHTML = `
    <div class="prompt-output-header">
      <span class="prompt-output-title">Enriched Prompt with Template</span>
      <div class="prompt-output-actions">
        <button class="danger prompt-cancel-btn hidden">Cancel</button>
        <button class="secondary prompt-edit-btn">Edit Prompt</button>
      </div>
    </div>
    <pre class="prompt-output-text"></pre>
  `;

  const editBtn = container.querySelector(".prompt-edit-btn");
  const cancelBtn = container.querySelector(".prompt-cancel-btn");
  state.promptEditBtn = editBtn;
  state.promptCancelBtn = cancelBtn;
  state.promptTextNode = container.querySelector(".prompt-output-text");
  state.promptTextNode.setAttribute("contenteditable", "false");

  cancelBtn.addEventListener("click", () => {
    if (!state.promptTextNode) return;
    state.promptTextNode.textContent = state.promptTextNode.dataset.originalPrompt || "";
    state.promptTextNode.setAttribute("contenteditable", "false");
    cancelBtn.classList.add("hidden");
    editBtn.textContent = "Edit Prompt";
  });

  editBtn.addEventListener("click", () => {
    if (!state.promptTextNode) return;

    if (state.promptTextNode.getAttribute("contenteditable") === "true") {
      const editedPrompt = state.promptTextNode.textContent.trim();
      if (!editedPrompt) {
        appendLog("Prompt cannot be empty.", "warning");
        state.promptTextNode.textContent = state.promptTextNode.dataset.originalPrompt || "";
        state.promptTextNode.setAttribute("contenteditable", "false");
        cancelBtn.classList.add("hidden");
        editBtn.textContent = "Edit Prompt";
        return;
      }

      (async () => {
        try {
          appendLog("Saving updated prompt...", "info");
          const result = await postJson("/api/codegen/update-prompt", {
            session_id: state.sessionId,
            prompt: editedPrompt
          });
          renderMilestones(result.milestones);
          const newLogs = (result.logs || []).slice(state.lastLogCount);
          appendBackendLogs(newLogs);
          state.lastLogCount = (result.logs || []).length;
          upsertPromptOutput(result.prompt || editedPrompt);
          state.promptTextNode.setAttribute("contenteditable", "false");
          cancelBtn.classList.add("hidden");
          editBtn.textContent = "Edit Prompt";
          appendLog("Prompt updated.", "info");
        } catch (error) {
          appendLog(`Error: ${error.message}`, "warning");
          state.promptTextNode.textContent = state.promptTextNode.dataset.originalPrompt || "";
          state.promptTextNode.setAttribute("contenteditable", "false");
          cancelBtn.classList.add("hidden");
          editBtn.textContent = "Edit Prompt";
        }
      })();
      return;
    }

    state.promptTextNode.dataset.originalPrompt = state.promptTextNode.textContent || "";
    state.promptTextNode.setAttribute("contenteditable", "true");
    cancelBtn.classList.remove("hidden");
    state.promptTextNode.focus();
    editBtn.textContent = "Save Prompt";
  });

  state.promptContainer = container;
  logStream.appendChild(container);
  return container;
}

function setPromptEditVisible(visible) {
  if (state.promptEditBtn) {
    state.promptEditBtn.classList.toggle("hidden", !visible);
    if (!visible) state.promptEditBtn.textContent = "Edit Prompt";
  }
  if (state.promptCancelBtn) {
    state.promptCancelBtn.classList.add("hidden");
  }
  if (!visible && state.promptTextNode) {
    state.promptTextNode.setAttribute("contenteditable", "false");
  }
}

function upsertPromptOutput(promptText) {
  const container = ensurePromptContainer();
  const normalizedPrompt = String(promptText || "").trim();
  if (state.promptTextNode) {
    state.promptTextNode.textContent = normalizedPrompt;
    state.promptTextNode.dataset.originalPrompt = normalizedPrompt;
  }
  setPromptEditVisible(true);
  state.promptDisplayed = true;
  logStream.appendChild(container);
  logStream.scrollTop = logStream.scrollHeight;
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  updateRunningIndicator();
}

function setPrimaryButtonLabel() {
  primaryActionBtn.textContent = state.awaitingAmbiguityResolution
    ? "Submit Resolution"
    : "Generate Prompt";
  refreshActionButtons();
}

function handleFailedState(result) {
  if (result.state !== "failed") return false;
  renderMilestones(result.milestones);
  const newLogs = (result.logs || []).slice(state.lastLogCount);
  appendBackendLogs(newLogs);
  state.lastLogCount = (result.logs || []).length;
  appendLog("Flow stopped due to failure. Resolve issue and retry.", "warning");
  updateSelfLearningAmbiguityPanel(null);
  state.promptReady = false;
  state.codeReady = false;
  state.awaitingAmbiguityResolution = false;
  setPrimaryButtonLabel();
  generateCodeBtn.classList.add("hidden");
  setReviewActionsVisible(false);
  commitBtn.classList.add("hidden");
  pushBtn.classList.add("hidden");
  setCommitMessageVisible(false);
  refreshActionButtons();
  stopPolling();
  updateRunningIndicator();
  return true;
}

async function getSessionSnapshot() {
  const response = await fetch(`${apiBaseUrl}/api/codegen/session/${state.sessionId}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Failed to fetch session");
  return data;
}

async function pollUntil(statesToStop) {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    try {
      const session = await getSessionSnapshot();
      renderMilestones(session.milestones);
      const newLogs = (session.logs || []).slice(state.lastLogCount);
      if (newLogs.length) {
        appendBackendLogs(newLogs);
        state.lastLogCount = session.logs.length;
      }

      if (handleFailedState(session)) return;

      if (session.state === "ambiguity_required" && statesToStop.includes("ambiguity_required")) {
        state.awaitingAmbiguityResolution = true;
        setPrimaryButtonLabel();
        updateSelfLearningAmbiguityPanel(session);
        appendLog(
          "Self-learning paused before final prompt: answer the open points in the panel, then submit.",
          "warning"
        );
        stopPolling();
        return;
      }

      if (session.state === "prompt_ready" && statesToStop.includes("prompt_ready")) {
        updateSelfLearningAmbiguityPanel(null);
        state.promptReady = true;
        generateCodeBtn.classList.remove("hidden");
        appendLog("Prompt ready. You can trigger code generation now.", "info");
        state.awaitingAmbiguityResolution = false;
        setPrimaryButtonLabel();
        upsertPromptOutput(session.prompt);
        refreshActionButtons();
        stopPolling();
        return;
      }

      if (session.state === "code_ready_for_review" && statesToStop.includes("code_ready_for_review")) {
        state.codeReady = true;
        setReviewActionsVisible(true);
        if (!state.codeDisplayed) {
          appendLog("----- CURSOR OUTPUT -----", "success");
          appendCodePreview(session.codePreview);
          state.codeDisplayed = true;
        }
        refreshActionButtons();
        stopPolling();
        return;
      }

      if (session.state === "pipeline_done" && statesToStop.includes("pipeline_done")) {
        appendLog("Flow complete. Jenkins pipeline finished successfully.", "info");
        stopPolling();
      }
    } catch (error) {
      appendLog(`Polling error: ${error.message}`, "warning");
      stopPolling();
    }
  }, 900);
}

async function postJson(path, body) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function getJson(path) {
  const response = await fetch(`${apiBaseUrl}${path}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.detail || "Request failed");
  }
  return data;
}

function showMainScreen() {
  gitHistoryScreen.classList.remove("active");
  settingsScreen.classList.remove("active");
  splashScreen.classList.remove("active");
  mainScreen.classList.add("active");
}

function showGitHistoryScreen() {
  splashScreen.classList.remove("active");
  mainScreen.classList.remove("active");
  settingsScreen.classList.remove("active");
  gitHistoryScreen.classList.add("active");
}

function showSettingsScreen() {
  splashScreen.classList.remove("active");
  mainScreen.classList.remove("active");
  gitHistoryScreen.classList.remove("active");
  settingsScreen.classList.add("active");
}

function buildLaneLayout(commits = []) {
  const hashToIndex = new Map();
  commits.forEach((c, i) => hashToIndex.set(c.hash, i));

  // Pin the "main" lane (first-parent chain) to lane 0 so the primary connector stays straight,
  // similar to VS Code's Git Graph.
  const mainChain = new Set();
  const head = commits?.[0]?.hash;
  if (head) {
    let cur = head;
    while (cur && hashToIndex.has(cur) && !mainChain.has(cur)) {
      mainChain.add(cur);
      const idx = hashToIndex.get(cur);
      const parents = Array.isArray(commits?.[idx]?.parents) ? commits[idx].parents : [];
      const next = parents?.[0];
      if (!next || !hashToIndex.has(next)) break;
      cur = next;
    }
  }

  const laneByHash = new Map();
  const active = [];
  let maxLane = 0;
  const rows = [];

  const ensureLaneForHash = (h) => {
    if (!h) return -1;
    let idx = active.indexOf(h);
    if (idx !== -1) return idx;
    // Never steal the main lane (0) for side branches.
    idx = active.slice(1).indexOf(null);
    idx = idx === -1 ? -1 : idx + 1;
    if (idx === -1) idx = active.length;
    active[idx] = h;
    return idx;
  };

  commits.forEach((c) => {
    const hash = c?.hash || "";
    const parents = Array.isArray(c?.parents) ? c.parents.filter(Boolean) : [];
    if (!hash) return;

    const incomingActive = active.map((v) => v != null);

    let lane = -1;
    if (mainChain.has(hash)) {
      lane = 0;
    } else {
      lane = active.indexOf(hash);
      if (lane === -1) {
        // Allocate side lanes from index 1+ so lane 0 stays reserved for main.
        const nullInSide = active.slice(1).indexOf(null);
        lane = nullInSide === -1 ? active.length : nullInSide + 1;
      }
    }
    active[lane] = hash;
    laneByHash.set(hash, lane);
    maxLane = Math.max(maxLane, lane);

    const mergeFromLanes = [];
    // Allocate merge-parent lanes (so they can be drawn as separate lines).
    parents.slice(1).forEach((p) => {
      const lp = ensureLaneForHash(p);
      mergeFromLanes.push(lp);
      maxLane = Math.max(maxLane, lp);
    });

    // Update active lanes for next row.
    const mainParent = parents[0] || null;
    active[lane] = mainParent;

    // End merge-parent lanes at this merge commit (they get absorbed).
    parents.slice(1).forEach((p) => {
      const lp = active.indexOf(p);
      if (lp !== -1 && lp !== 0) active[lp] = null;
    });

    // Clean duplicates of mainParent.
    if (mainParent) {
      for (let i = 0; i < active.length; i++) {
        if (i !== lane && i !== 0 && active[i] === mainParent) active[i] = null;
      }
    }

    const outgoingActive = active.map((v) => v != null);
    rows.push({
      hash,
      lane,
      mergeFromLanes: mergeFromLanes.filter((v) => v >= 0 && v !== lane),
      incomingActive,
      outgoingActive
    });
  });

  return { laneByHash, hashToIndex, laneCount: maxLane + 1, rows };
}

function renderGitCommitGraphOverlay(commits, listEl, svgEl) {
  const { laneByHash, hashToIndex, laneCount, rows } = buildLaneLayout(commits);
  const rowEls = Array.from(listEl.querySelectorAll(".git-commit-node"));
  const yCenters = rowEls.map((el) => Math.floor(el.offsetTop + el.offsetHeight / 2));
  const totalH = Math.max(1, listEl.scrollHeight);

  const graphColumnWidth = 64; // must match CSS `grid-template-columns` first column
  const lanePadLeft = 16;
  const maxLaneStep = 14;
  const minLaneStep = 8;
  const laneStep =
    laneCount <= 1
      ? maxLaneStep
      : Math.max(
          minLaneStep,
          Math.min(
            maxLaneStep,
            Math.floor((graphColumnWidth - lanePadLeft - 12) / Math.max(1, laneCount - 1))
          )
        );
  const xForLane = (lane) => lanePadLeft + lane * laneStep;
  const yForIndex = (idx) => yCenters[idx] ?? Math.floor(idx * 56 + 28);

  const palette = ["#1e104e", "#0ea5e9", "#22c55e", "#f97316", "#a855f7", "#ef4444", "#14b8a6"];
  const colorForLane = (lane) => palette[Math.abs(lane) % palette.length];

  const paths = [];
  const dots = [];

  // Draw continuous lane lines between rows (VS Code-like).
  for (let i = 0; i < rows.length - 1; i++) {
    const y1 = yForIndex(i);
    const y2 = yForIndex(i + 1);
    const out = rows[i]?.outgoingActive || [];
    for (let lane = 0; lane < out.length; lane++) {
      if (!out[lane]) continue;
      const x = xForLane(lane);
      paths.push(
        `<path d="M ${x} ${y1} L ${x} ${y2}" stroke="${colorForLane(
          lane
        )}" stroke-width="2" fill="none" stroke-linecap="round" />`
      );
    }
  }

  // Draw parent edges (commit -> each parent) for merges/branch joins.
  commits.forEach((c, idx) => {
    const hash = c?.hash || "";
    if (!hash) return;
    const lane = laneByHash.get(hash) ?? 0;
    const x1 = xForLane(lane);
    const y1 = yForIndex(idx);
    const parents = Array.isArray(c?.parents) ? c.parents.filter(Boolean) : [];

    // Dot
    dots.push(
      `<circle cx="${x1}" cy="${y1}" r="${parents.length > 1 ? 5.5 : 5}" fill="${colorForLane(
        lane
      )}" stroke="#ffffff" stroke-width="2" />`
    );

    parents.forEach((p, pIdx) => {
      const parentIndex = hashToIndex.get(p);
      if (parentIndex == null) return;
      const parentLane = laneByHash.get(p) ?? lane;
      const x2 = xForLane(parentLane);
      const y2 = yForIndex(parentIndex);
      const stroke = colorForLane(pIdx === 0 ? lane : parentLane);

      if (x1 === x2) {
        paths.push(
          `<path d="M ${x1} ${y1} L ${x2} ${y2}" stroke="${stroke}" stroke-width="2" fill="none" stroke-linecap="round" />`
        );
        return;
      }

      const midY = y1 + Math.max(12, Math.min(32, (y2 - y1) * 0.35));
      const c1x = x1;
      const c1y = midY;
      const c2x = x2;
      const c2y = y2 - Math.max(12, Math.min(32, (y2 - y1) * 0.35));
      paths.push(
        `<path d="M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}" stroke="${stroke}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" />`
      );
    });
  });

  const width = graphColumnWidth;
  svgEl.setAttribute("width", String(graphColumnWidth));
  svgEl.setAttribute("height", String(totalH));
  svgEl.setAttribute("viewBox", `0 0 ${width} ${totalH}`);

  svgEl.innerHTML = `${paths.join("")}${dots.join("")}`;
}

function appendGitCommitNodes(commits = [], listEl) {
  commits.forEach((commit) => {
    const node = document.createElement("div");
    node.className = "git-commit-node";
    node.dataset.hash = commit.hash;
    const shouldShowRemoteBadge =
      Boolean(state.selectedBranchUpstreamHead) && commit.hash === state.selectedBranchUpstreamHead;
    const remoteLabel = shouldShowRemoteBadge
      ? state.selectedBranchUpstream ||
        (state.selectedHistoryBranch && state.gitDefaultRemote
          ? `${state.gitDefaultRemote}/${state.selectedHistoryBranch}`
          : "")
      : "";
    const remoteBadge =
      shouldShowRemoteBadge && remoteLabel
        ? `<span class="git-remote-badge" title="Remote tracking HEAD">☁ ${remoteLabel}</span>`
        : "";
    node.innerHTML = `
      <span class="git-commit-graph"></span>
      <span>
        <div class="git-commit-subject-row">
          <div class="git-commit-subject">${commit.subject || "(no message)"}</div>
          ${remoteBadge}
        </div>
        <div class="git-commit-meta">${commit.short_hash} • ${commit.author} • ${commit.date}</div>
      </span>
    `;
    if (state.selectedHistoryCommit && state.selectedHistoryCommit === commit.hash) {
      node.classList.add("active");
    }
    node.addEventListener("click", () => {
      loadGitCommitDetails(commit.hash);
    });
    listEl.appendChild(node);
  });
}

function renderGitCommitTree(commits = []) {
  gitCommitTree.innerHTML = "";
  if (!Array.isArray(commits) || !commits.length) {
    gitCommitTree.innerHTML = `<div class="git-empty-state">No commits found for this branch.</div>`;
    state.gitRenderedCommitCount = 0;
    return false;
  }

  const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  overlay.classList.add("git-commit-graph-overlay");
  overlay.setAttribute("aria-hidden", "true");
  overlay.style.left = "8px";
  overlay.style.top = "8px";
  overlay.style.position = "absolute";

  const list = document.createElement("div");
  list.className = "git-commit-list";

  gitCommitTree.appendChild(overlay);
  gitCommitTree.appendChild(list);

  appendGitCommitNodes(commits, list);
  // After nodes are in the DOM, compute real row positions and paint connectors.
  requestAnimationFrame(() => renderGitCommitGraphOverlay(commits, list, overlay));

  state.gitRenderedCommitCount = commits.length;
  return true;
}

function renderGitCommitDetails(payload) {
  const commit = payload?.commit || {};
  const files = Array.isArray(payload?.files) ? payload.files : [];
  gitCommitDetail.innerHTML = "";

  const head = document.createElement("div");
  head.className = "git-commit-detail-head";
  head.innerHTML = `
    <div class="git-commit-detail-title">${commit.subject || "(no message)"}</div>
    <div class="git-commit-detail-meta">${commit.short_hash || ""} • ${commit.author || ""} • ${
      commit.date || ""
    }</div>
  `;
  gitCommitDetail.appendChild(head);

  if (!files.length) {
    const empty = document.createElement("div");
    empty.className = "git-empty-state";
    empty.textContent = "No file-level changes found in this commit.";
    gitCommitDetail.appendChild(empty);
    return;
  }

  files.forEach((file) => {
    const fileCard = document.createElement("div");
    fileCard.className = "git-file-card";
    const header = document.createElement("button");
    header.type = "button";
    header.className = "git-file-header";
    header.innerHTML = `
      <span class="git-file-title">${file.path} (+${Number(file.insertions || 0)} / -${Number(
      file.deletions || 0
    )})</span>
      <span class="git-file-toggle" aria-hidden="true">${CHEVRON_UP_SVG}</span>
    `;
    const toggleNode = header.querySelector(".git-file-toggle");
    fileCard.appendChild(header);

    const diffBox = document.createElement("div");
    diffBox.className = "git-file-diff";
    fileCard.appendChild(diffBox);

    header.addEventListener("click", () => {
      const isCollapsed = fileCard.classList.toggle("collapsed");
      if (toggleNode) {
        toggleNode.innerHTML = isCollapsed ? CHEVRON_DOWN_SVG : CHEVRON_UP_SVG;
      }
    });
    const lines = String(file.diff || "").split("\n");
    lines.forEach((line) => {
      const row = document.createElement("div");
      const lineStyle = streamUtils.classifyDiffLine
        ? streamUtils.classifyDiffLine(line)
        : "diff-neutral";
      row.className = `code-preview-line ${lineStyle}`;
      row.textContent = line;
      diffBox.appendChild(row);
    });
    gitCommitDetail.appendChild(fileCard);
  });
}

async function loadGitCommitDetails(commitHash) {
  if (!commitHash) return;
  state.selectedHistoryCommit = commitHash;
  Array.from(gitCommitTree.querySelectorAll(".git-commit-node")).forEach((node) => {
    node.classList.toggle("active", node.dataset.hash === commitHash);
  });
  gitCommitDetail.innerHTML = `<div class="git-empty-state">Loading commit details...</div>`;
  try {
    const payload = await getJson(`/api/codegen/git/commit/${encodeURIComponent(commitHash)}`);
    renderGitCommitDetails(payload);
  } catch (error) {
    gitCommitDetail.innerHTML = `<div class="git-empty-state">Failed to load commit details: ${error.message}</div>`;
  }
}

async function loadGitCommits(branchName, resetSelection = true, appendMode = false) {
  const branch = branchName || state.selectedHistoryBranch;
  if (!branch) return;
  if (!appendMode) {
    gitCommitTree.innerHTML = `<div class="git-empty-state">Loading commits...</div>`;
    gitCommitDetail.innerHTML = `<div class="git-empty-state">Select a commit to view file-wise changes.</div>`;
  }
  try {
    if (resetSelection) state.selectedHistoryCommit = "";
    const preserveScrollTop =
      appendMode && typeof gitCommitTree.scrollTop === "number" ? gitCommitTree.scrollTop : null;
    const payload = await getJson(
      `/api/codegen/git/commits?branch=${encodeURIComponent(branch)}&limit=${state.gitCommitLimit}`
    );
    const commits = payload.commits || [];
    state.selectedBranchUpstreamHead = payload?.upstream_head || "";
    const previousControls = gitCommitTree.querySelector(".git-commit-tree-controls");
    if (previousControls) previousControls.remove();

    let hasCommits = false;
    // For graph rendering we re-render the full list so lane computation stays consistent.
    hasCommits = renderGitCommitTree(commits);
    if (preserveScrollTop != null) {
      // Restore scroll position so "Show more" does not jump to top.
      gitCommitTree.scrollTop = preserveScrollTop;
      requestAnimationFrame(() => {
        gitCommitTree.scrollTop = preserveScrollTop;
      });
    }

    if (hasCommits && commits.length >= state.gitCommitLimit) {
      const controls = document.createElement("div");
      controls.className = "git-commit-tree-controls";
      const showMoreBtn = document.createElement("button");
      showMoreBtn.type = "button";
      showMoreBtn.className = "secondary git-show-more-btn";
      showMoreBtn.textContent = "Show more";
      showMoreBtn.addEventListener("click", async () => {
        state.gitCommitLimit += 10;
        // Use appendMode=true as "preserve scroll" mode (we still re-render fully).
        await loadGitCommits(branch, false, true);
      });
      controls.appendChild(showMoreBtn);
      gitCommitTree.appendChild(controls);
    }
  } catch (error) {
    gitCommitTree.innerHTML = `<div class="git-empty-state">Failed to load commits: ${error.message}</div>`;
  }
}

async function loadGitHistory() {
  gitCommitTree.innerHTML = `<div class="git-empty-state">Loading branches...</div>`;
  gitCommitDetail.innerHTML = `<div class="git-empty-state">Select a commit to view file-wise changes.</div>`;
  try {
    const payload = await getJson("/api/codegen/git/branches");
    const branches = Array.isArray(payload.branches) ? payload.branches : [];
    state.gitBranchUpstreams = payload?.upstreams || {};
    state.gitDefaultRemote = payload?.default_remote || "origin";
    gitBranchSelect.innerHTML = "";
    branches.forEach((branch) => {
      const option = document.createElement("option");
      option.value = branch;
      option.textContent = branch;
      gitBranchSelect.appendChild(option);
    });
    const selected = payload.current_branch && branches.includes(payload.current_branch)
      ? payload.current_branch
      : branches[0] || "";
    state.selectedHistoryBranch = selected;
    state.selectedBranchUpstream = state.gitBranchUpstreams[selected] || "";
    state.selectedBranchUpstreamHead = "";
    if (selected) {
      state.gitCommitLimit = 10;
      state.gitRenderedCommitCount = 0;
      gitBranchSelect.value = selected;
      await loadGitCommits(selected);
    } else {
      gitCommitTree.innerHTML = `<div class="git-empty-state">No branches found.</div>`;
    }
    state.gitHistoryLoaded = true;
  } catch (error) {
    gitCommitTree.innerHTML = `<div class="git-empty-state">Failed to load git history: ${error.message}</div>`;
  }
}

enterPlatformBtn.addEventListener("click", () => {
  splashScreen.classList.remove("active");
  mainScreen.classList.add("active");
});

openGitHistoryBtn.addEventListener("click", async () => {
  showGitHistoryScreen();
  if (!state.gitHistoryLoaded) {
    await loadGitHistory();
  }
});

gitRefreshBtn.addEventListener("click", async () => {
  state.gitHistoryLoaded = false;
  state.gitCommitLimit = 10;
  state.gitRenderedCommitCount = 0;
  state.selectedHistoryCommit = "";
  await loadGitHistory();
});

closeGitHistoryBtn.addEventListener("click", () => {
  showMainScreen();
});

settingsOpenButtons.forEach((button) => {
  button.addEventListener("click", () => {
    showSettingsScreen();
  });
});

settingsBackBtn.addEventListener("click", () => {
  showMainScreen();
});

gitBranchSelect.addEventListener("change", async (event) => {
  const selectedBranch = event.target.value;
  state.selectedHistoryBranch = selectedBranch;
  state.selectedBranchUpstream = state.gitBranchUpstreams[selectedBranch] || "";
  state.selectedBranchUpstreamHead = "";
  state.gitCommitLimit = 10;
  state.gitRenderedCommitCount = 0;
  await loadGitCommits(selectedBranch);
});

primaryActionBtn.addEventListener("click", async () => {
  const input = intentInput.value.trim();
  if (!input) {
    appendLog("Input required in intent text box.", "warning");
    return;
  }

  try {
    if (!state.awaitingAmbiguityResolution) {
      setPipelineVisibility(true);
      appendLog("Starting prompt generation flow...", "info");
      updateSelfLearningAmbiguityPanel(null);
      const result = await postJson("/api/codegen/generate", { intent: input });
      state.sessionId = result.session_id;
      state.lastLogCount = (result.logs || []).length;
      state.promptDisplayed = false;
      state.promptContainer = null;
      state.promptTextNode = null;
      state.promptEditBtn = null;
      state.promptCancelBtn = null;
      state.codeDisplayed = false;
      state.promptReady = false;
      state.codeReady = false;
      generateCodeBtn.classList.add("hidden");
      commitBtn.classList.add("hidden");
      pushBtn.classList.add("hidden");
      setCommitMessageVisible(false);
      setReviewActionsVisible(false);
      refreshActionButtons();
      renderMilestones(result.milestones);
      appendBackendLogs(result.logs);
      if (handleFailedState(result)) return;
      if (result.state === "ambiguity_required") {
        updateSelfLearningAmbiguityPanel(result);
      }
      refreshActionButtons();
      pollUntil(["ambiguity_required", "prompt_ready"]);
      return;
    }

    appendLog("Submitting ambiguity resolution...", "info");
    updateSelfLearningAmbiguityPanel(null);
    const result = await postJson("/api/codegen/resolve-ambiguities", {
      session_id: state.sessionId,
      resolution: input
    });
    if (handleFailedState(result)) return;
    renderMilestones(result.milestones);
    const newLogs = (result.logs || []).slice(state.lastLogCount);
    appendBackendLogs(newLogs);
    state.lastLogCount = (result.logs || []).length;
    state.awaitingAmbiguityResolution = false;
    setPrimaryButtonLabel();
    refreshActionButtons();
    pollUntil(["prompt_ready"]);
  } catch (error) {
    appendLog(`Error: ${error.message}`, "warning");
  }
});

generateCodeBtn.addEventListener("click", async () => {
  if (!state.sessionId || !state.promptReady) {
    appendLog("Prompt is not ready. Generate prompt first.", "warning");
    return;
  }
  if (state.generationInProgress) return;

  try {
    setPipelineVisibility(true);
    setCodeGenerationBusy(true);
    updateMilestoneStatusLocal("code_generation", "in_progress");
    clearCursorOutputContainer();
    appendLog("Triggering code generation stream...", "info");
    setPromptEditVisible(false);
    const input = intentInput.value.trim();
    const response = await fetch(`${apiBaseUrl}/api/code-generation/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        user_message: input || "Generate code changes based on the prepared prompt.",
        branch: "ltm_feature"
      })
    });
    if (!response.ok) {
      let detail = "Failed to start code generation stream.";
      try {
        const payload = await response.json();
        detail = payload.error || payload.detail || detail;
      } catch (_error) {
        // ignore json decoding errors
      }
      throw new Error(detail);
    }
    let finalResult = null;
    await readNdjsonStream(response, (event) => {
      if (event?.type === "log") {
        const eventText = event.text || "";
        if (!isCursorCliTranscriptLog(eventText)) {
          appendLog(eventText, "info");
        }
      } else if (event?.type === "error") {
        updateMilestoneStatusLocal("code_generation", "failed");
        appendLog(event.error || "Generation failed.", "warning");
      } else if (event?.type === "result") {
        finalResult = event.data || null;
      }
    });
    if (!finalResult) {
      updateMilestoneStatusLocal("code_generation", "failed");
      appendLog("Generation stream completed without a final payload.", "warning");
      return;
    }
    if (finalResult.type === "cursor_cli_generation") {
      appendCursorChatOutput(finalResult.chat_output);
      appendCodeChanges(finalResult.code_changes, finalResult.changed_files);
      if (finalResult.warning) {
        appendLog(finalResult.warning, "warning");
      }
      state.codeReady = Boolean(finalResult.review_available);
      setReviewActionsVisible(Boolean(finalResult.review_available));
      updateMilestoneStatusLocal(
        "code_generation",
        finalResult.success ? "completed" : "failed"
      );
      if (finalResult.error) {
        appendLog(finalResult.error, "warning");
      }
      refreshActionButtons();
    }
  } catch (error) {
    updateMilestoneStatusLocal("code_generation", "failed");
    appendLog(`Error: ${error.message}`, "warning");
  } finally {
    setCodeGenerationBusy(false);
  }
});

acceptBtn.addEventListener("click", async () => {
  if (!state.codeReady || state.reviewInProgress) return;
  try {
    setReviewBusy(true);
    const response = await fetch(`${apiBaseUrl}/api/code-generation/review/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "accept" })
    });
    if (!response.ok) throw new Error("Failed to submit accept review action.");
    let finalResult = null;
    await readNdjsonStream(response, (event) => {
      if (event?.type === "log") appendLog(event.text || "", "info");
      if (event?.type === "error") appendLog(event.error || "Review request failed.", "warning");
      if (event?.type === "result") finalResult = event.data || null;
    });
    if (finalResult?.success) {
      appendLog("Generated changes accepted.", "info");
      state.codeReady = false;
      generateCodeBtn.classList.add("hidden");
      setReviewActionsVisible(false);
      commitBtn.classList.remove("hidden");
      pushBtn.classList.add("hidden");
      setCommitMessageVisible(true, finalResult?.suggested_commit_message || "");
      refreshActionButtons();
    } else {
      appendLog(finalResult?.error || "Accept action failed.", "warning");
    }
  } catch (error) {
    appendLog(`Error: ${error.message}`, "warning");
  } finally {
    setReviewBusy(false);
  }
});

rejectBtn.addEventListener("click", async () => {
  if (!state.codeReady || state.reviewInProgress) return;
  try {
    setReviewBusy(true);
    const response = await fetch(`${apiBaseUrl}/api/code-generation/review/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "reject" })
    });
    if (!response.ok) throw new Error("Failed to submit reject review action.");
    let finalResult = null;
    await readNdjsonStream(response, (event) => {
      if (event?.type === "log") appendLog(event.text || "", "info");
      if (event?.type === "error") appendLog(event.error || "Review request failed.", "warning");
      if (event?.type === "result") finalResult = event.data || null;
    });
    if (finalResult?.success) {
      appendLog("Generated changes rejected and restored.", "warning");
      state.codeReady = false;
      generateCodeBtn.classList.remove("hidden");
      setReviewActionsVisible(false);
      commitBtn.classList.add("hidden");
      pushBtn.classList.add("hidden");
      setCommitMessageVisible(false);
      refreshActionButtons();
    } else {
      appendLog(finalResult?.error || "Reject action failed.", "warning");
    }
  } catch (error) {
    appendLog(`Error: ${error.message}`, "warning");
  } finally {
    setReviewBusy(false);
  }
});

commitBtn.addEventListener("click", async () => {
  try {
    setPipelineVisibility(true);
    appendLog("Committing accepted changes...", "info");
    const commitMessage = commitMessageInput.value.trim();
    if (!commitMessage) {
      appendLog("Commit message is required.", "warning");
      return;
    }
    state.commitInProgress = true;
    refreshActionButtons();
    const result = await postJson("/api/codegen/commit-push", {
      session_id: state.sessionId,
      commit_message: commitMessage
    });
    if (handleFailedState(result)) return;
    renderMilestones(result.milestones);
    const newLogs = (result.logs || []).slice(state.lastLogCount);
    appendBackendLogs(newLogs);
    state.lastLogCount = (result.logs || []).length;
    if (result.state === "commit_ready_for_push") {
      appendLog("Commit successful. Click Push to publish changes.", "info");
      commitBtn.classList.add("hidden");
      pushBtn.classList.remove("hidden");
      refreshActionButtons();
      return;
    }
  } catch (error) {
    appendLog(`Error: ${error.message}`, "warning");
  } finally {
    state.commitInProgress = false;
    refreshActionButtons();
  }
});

pushBtn.addEventListener("click", async () => {
  try {
    setPipelineVisibility(true);
    appendLog("Pushing committed changes...", "info");
    state.pushInProgress = true;
    refreshActionButtons();
    const result = await postJson("/api/codegen/push", {
      session_id: state.sessionId
    });
    if (handleFailedState(result)) return;
    renderMilestones(result.milestones);
    const newLogs = (result.logs || []).slice(state.lastLogCount);
    appendBackendLogs(newLogs);
    state.lastLogCount = (result.logs || []).length;
    if (result.state === "pipeline_done") {
      pushBtn.classList.add("hidden");
      setCommitMessageVisible(false);
      refreshActionButtons();
      pollUntil(["pipeline_done"]);
    }
  } catch (error) {
    appendLog(`Error: ${error.message}`, "warning");
  } finally {
    state.pushInProgress = false;
    refreshActionButtons();
  }
});

setReviewActionsVisible(false);
setCommitMessageVisible(false);
renderMilestones(getInitialMilestones());
setPipelineVisibility(false);
renderInitialLogView();
refreshActionButtons();
