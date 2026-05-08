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
const intentStatusPill = document.getElementById("intent-status-pill");
const ambiguityCallout = document.getElementById("self-learning-ambiguity-callout");
const ambiguityItemList = document.getElementById("ambiguity-item-list");
const runningIndicator = document.getElementById("running-indicator");
const runningIndicatorText = document.getElementById("running-indicator-text");
const centerTabs = document.querySelectorAll(".header-tab[data-center-pane]");
const centerPanes = {
  "logs-outputs": document.getElementById("pane-logs-outputs"),
  "audit-trail": document.getElementById("pane-audit-trail"),
  "eval-learning": document.getElementById("pane-eval-learning"),
  "validation-compilation": document.getElementById("pane-validation-compilation")
};
const headerStatusBadge = document.querySelector(".header-tab-status");
const panelHeadingTitle = document.querySelector(".right-panel-heading h3");
const panelHeadingBadge = document.querySelector(".pipeline-badge");
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
const pipelineRunMeta = document.getElementById("pipeline-run-meta");
const streamUtils = window.StreamUtils || {};
const CHEVRON_DOWN_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-down" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708"/>
</svg>`;
const CHEVRON_UP_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-up" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M7.646 4.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1-.708.708L8 5.707l-5.646 5.647a.5.5 0 0 1-.708-.708z"/>
</svg>`;

const CENTER_PANE_HEADER = {
  "logs-outputs": { title: "Live Output", badge: "SYS" },
  "audit-trail": { title: "Audit Trail", badge: "AUDIT" },
  "eval-learning": { title: "Eval & Learning", badge: "EVAL" },
  "validation-compilation": { title: "Validation & Compilation", badge: "VALIDATE" }
};

function setActiveCenterPane(paneId) {
  centerTabs.forEach((tab) => {
    const isActive = tab.dataset.centerPane === paneId;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", isActive ? "true" : "false");
  });

  Object.entries(centerPanes).forEach(([id, node]) => {
    if (!node) return;
    node.classList.toggle("hidden", id !== paneId);
  });

  const meta = CENTER_PANE_HEADER[paneId] || CENTER_PANE_HEADER["logs-outputs"];
  if (panelHeadingTitle) panelHeadingTitle.textContent = meta.title;
  if (panelHeadingBadge) panelHeadingBadge.textContent = meta.badge;
}

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
  if (pipelineRunMeta) pipelineRunMeta.classList.toggle("hidden", !visible);
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

  const phaseStatusText = (status) => {
    if (status === "in_progress") return "in progress";
    if (status === "completed") return "completed";
    if (status === "failed") return "failed";
    return "pending";
  };

  pipelineTrack.innerHTML = "";
  phases.forEach((phase, index) => {
    const item = document.createElement("div");
    item.className = "pipeline-item";
    if (index === activeIndex && phase.status !== "completed" && phase.status !== "failed") {
      item.classList.add("active");
    }
    if (phase.status === "completed") item.classList.add("completed");
    if (phase.status === "in_progress") item.classList.add("in-progress");
    if (phase.status === "failed") item.classList.add("failed");
    const statusLabel = phaseStatusText(phase.status);
    if (phase.subtitle) {
      item.innerHTML = `<div class="pipeline-item-head"><div class="pipeline-title">${phase.title}</div><div class="pipeline-phase-status">${statusLabel}</div></div><div class="pipeline-subtitle">${phase.subtitle}</div>`;
    } else {
      item.innerHTML = `<div class="pipeline-item-head"><div class="pipeline-title">${phase.title}</div><div class="pipeline-phase-status">${statusLabel}</div></div>`;
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
    refreshTopStatusBadges();
    return;
  }
  if (state.reviewInProgress) {
    runningIndicatorText.textContent = "Applying review action...";
    refreshTopStatusBadges();
    return;
  }
  runningIndicatorText.textContent = getRunningStageMessage();
  refreshTopStatusBadges();
}

function refreshTopStatusBadges() {
  const isRunning =
    isMilestoneRunning() || state.generationInProgress || state.reviewInProgress;
  const statusText = isRunning ? "Running" : state.promptReady ? "Prompt Ready" : "Idle";
  if (intentStatusPill) {
    intentStatusPill.textContent = statusText;
    intentStatusPill.classList.toggle("is-running", isRunning);
    intentStatusPill.classList.toggle("is-ready", !isRunning && state.promptReady);
  }
  if (headerStatusBadge) {
    headerStatusBadge.textContent = statusText;
    headerStatusBadge.classList.toggle("is-running", isRunning);
    headerStatusBadge.classList.toggle("is-ready", !isRunning && state.promptReady);
  }
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

const SETTINGS_STORAGE_KEY = "unifiedPlatform.settings.v1";

function readPlatformSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writePlatformSettings(data) {
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(data));
}

function initSettingsUi() {
  const paneAi = document.getElementById("settings-pane-ai-engine");
  const panePromptTemplates = document.getElementById("settings-pane-prompt-templates");
  const paneRepositoryConfig = document.getElementById("settings-pane-repository-config");
  const paneBranchStrategy = document.getElementById("settings-pane-branch-strategy");
  const paneCicdPipeline = document.getElementById("settings-pane-ci-cd-pipeline");
  const paneTestRunner = document.getElementById("settings-pane-test-runner");
  const paneRcaEngine = document.getElementById("settings-pane-rca-engine");
  const paneQualityGates = document.getElementById("settings-pane-quality-gates");
  const paneNotifications = document.getElementById("settings-pane-notifications");
  const paneWorkspacePaths = document.getElementById("settings-pane-workspace-paths");
  const paneAdvanced = document.getElementById("settings-pane-advanced");
  const paneGeneric = document.getElementById("settings-pane-generic");
  const genericTitle = document.getElementById("settings-generic-title");
  const genericSubtitle = document.getElementById("settings-generic-subtitle");
  const genericFields = document.getElementById("settings-generic-fields");
  const genericStatus = document.getElementById("settings-generic-status");
  const genericSaveBtn = document.getElementById("settings-generic-save");
  const genericResetBtn = document.getElementById("settings-generic-reset");
  const navButtons = document.querySelectorAll(".settings-sidebar .settings-nav-item[data-settings-pane]");
  const engineInputs = document.querySelectorAll('input[name="active-coding-engine"]');
  const cursorPanel = document.getElementById("settings-cursor-panel");
  const connPill = document.getElementById("settings-engine-connection-pill");
  const connLabel = connPill?.querySelector(".settings-connection-label");
  const engineRuntimeIcon = document.getElementById("settings-engine-runtime-icon");
  const engineSettingsTitle = document.getElementById("settings-cursor-heading");
  const enginePathLabel = document.getElementById("settings-engine-path-label");
  const enginePathHint = document.getElementById("settings-engine-path-hint");
  const pathInput = document.getElementById("settings-cursor-executable-path");
  const outSel = document.getElementById("settings-cursor-output-format");
  const trustSel = document.getElementById("settings-cursor-trust-mode");
  const forceChk = document.getElementById("settings-cursor-force-overwrite");
  const testBtn = document.getElementById("settings-cursor-test-connection");
  const saveBtn = document.getElementById("settings-cursor-save");
  const statusEl = document.getElementById("settings-cursor-status");

  const templatesDefaultSel = document.getElementById("settings-templates-default");
  const templatesDirInput = document.getElementById("settings-templates-directory");
  const templatesRoutes = document.getElementById("settings-templates-routes");
  const templatesAddBtn = document.getElementById("settings-templates-add");
  const templatesSaveBtn = document.getElementById("settings-templates-save");
  const templatesStatus = document.getElementById("settings-templates-status");

  const repoUrlInput = document.getElementById("settings-repo-url");
  const repoWorkspaceInput = document.getElementById("settings-repo-workspace");
  const repoStackSel = document.getElementById("settings-repo-stack");
  const repoAuthSel = document.getElementById("settings-repo-auth");
  const repoTestBtn = document.getElementById("settings-repo-test");
  const repoSaveBtn = document.getElementById("settings-repo-save");
  const repoReachablePill = document.getElementById("settings-repo-reachable-pill");
  const repoReachableLabel = repoReachablePill?.querySelector(".settings-connection-label");
  const repoStatus = document.getElementById("settings-repo-status");

  const branchBaseInput = document.getElementById("settings-branch-base");
  const branchPatternInput = document.getElementById("settings-branch-pattern");
  const branchAutoPush = document.getElementById("settings-branch-auto-push");
  const branchCommitTemplate = document.getElementById("settings-branch-commit-template");
  const branchSaveBtn = document.getElementById("settings-branch-save");
  const branchStatus = document.getElementById("settings-branch-status");
  const cicdInputs = document.querySelectorAll('input[name="cicd-system"]');
  const cicdPill = document.getElementById("settings-cicd-pill");
  const cicdPillLabel = document.getElementById("settings-cicd-pill-label");
  const cicdUrl = document.getElementById("settings-cicd-url");
  const cicdJob = document.getElementById("settings-cicd-job");
  const cicdToken = document.getElementById("settings-cicd-token");
  const cicdTrigger = document.getElementById("settings-cicd-trigger-on-push");
  const cicdPoll = document.getElementById("settings-cicd-poll");
  const cicdTestBtn = document.getElementById("settings-cicd-test");
  const cicdSaveBtn = document.getElementById("settings-cicd-save");
  const cicdStatus = document.getElementById("settings-cicd-status");
  const testRunnerInputs = document.querySelectorAll('input[name="test-runner-framework"]');
  const testRunnerCommand = document.getElementById("settings-test-runner-command");
  const testRunnerResultFormat = document.getElementById("settings-test-runner-result-format");
  const testRunnerTimeout = document.getElementById("settings-test-runner-timeout");
  const testRunnerSaveBtn = document.getElementById("settings-test-runner-save");
  const testRunnerStatus = document.getElementById("settings-test-runner-status");
  const rcaAutoTrigger = document.getElementById("settings-rca-auto-trigger");
  const rcaAutoApply = document.getElementById("settings-rca-auto-apply");
  const rcaReviewBelowThreshold = document.getElementById("settings-rca-review-below-threshold");
  const rcaSearchHistory = document.getElementById("settings-rca-search-history");
  const rcaAutoApplyThreshold = document.getElementById("settings-rca-auto-apply-threshold");
  const rcaMaxIterations = document.getElementById("settings-rca-max-iterations");
  const rcaMaxIterationAction = document.getElementById("settings-rca-max-iteration-action");
  const rcaSaveBtn = document.getElementById("settings-rca-save");
  const rcaStatus = document.getElementById("settings-rca-status");
  const qualityMinPassRate = document.getElementById("settings-quality-min-pass-rate");
  const qualityMinCoverage = document.getElementById("settings-quality-min-coverage");
  const qualityMaxWarnings = document.getElementById("settings-quality-max-warnings");
  const qualityBlockOnFail = document.getElementById("settings-quality-block-on-fail");
  const qualitySaveBtn = document.getElementById("settings-quality-save");
  const qualityStatus = document.getElementById("settings-quality-status");
  const notifyEmailCompletion = document.getElementById("settings-notify-email-completion");
  const notifyEmailFailure = document.getElementById("settings-notify-email-failure");
  const notifyTeamsEnabled = document.getElementById("settings-notify-teams-enabled");
  const notifySlackEnabled = document.getElementById("settings-notify-slack-enabled");
  const notifyEmailAddress = document.getElementById("settings-notify-email-address");
  const notificationsSaveBtn = document.getElementById("settings-notifications-save");
  const notificationsStatus = document.getElementById("settings-notifications-status");
  const pathPlatformRoot = document.getElementById("settings-path-platform-root");
  const pathPromptOutput = document.getElementById("settings-path-prompt-output");
  const pathLogOutput = document.getElementById("settings-path-log-output");
  const pathTestResults = document.getElementById("settings-path-test-results");
  const pathsSaveBtn = document.getElementById("settings-paths-save");
  const pathsStatus = document.getElementById("settings-paths-status");
  const advancedDryRun = document.getElementById("settings-advanced-dry-run");
  const advancedVerboseLogging = document.getElementById("settings-advanced-verbose-logging");
  const advancedPersistSession = document.getElementById("settings-advanced-persist-session");
  const advancedVersion = document.getElementById("settings-advanced-version");
  const advancedSaveBtn = document.getElementById("settings-advanced-save");
  const advancedResetBtn = document.getElementById("settings-advanced-reset");
  const advancedStatus = document.getElementById("settings-advanced-status");

  if (
    !paneAi ||
    !panePromptTemplates ||
    !paneRepositoryConfig ||
    !paneBranchStrategy ||
    !paneCicdPipeline ||
    !paneTestRunner ||
    !paneRcaEngine ||
    !paneQualityGates ||
    !paneNotifications ||
    !paneWorkspacePaths ||
    !paneAdvanced ||
    !paneGeneric ||
    !genericTitle ||
    !genericSubtitle ||
    !genericFields ||
    !genericSaveBtn ||
    !genericResetBtn ||
    navButtons.length === 0 ||
    engineInputs.length === 0 ||
    !cursorPanel ||
    !connPill ||
    !engineRuntimeIcon ||
    !engineSettingsTitle ||
    !enginePathLabel ||
    !enginePathHint ||
    !pathInput ||
    !outSel ||
    !trustSel ||
    !forceChk ||
    !testBtn ||
    !saveBtn
  ) {
    return;
  }

  let statusTimerId = null;
  let genericStatusTimerId = null;
  let templatesStatusTimerId = null;
  let repoStatusTimerId = null;
  let branchStatusTimerId = null;
  let cicdStatusTimerId = null;
  let testRunnerStatusTimerId = null;
  let rcaStatusTimerId = null;
  let qualityStatusTimerId = null;
  let notificationsStatusTimerId = null;
  let pathsStatusTimerId = null;
  let advancedStatusTimerId = null;
  let activeGenericPaneKey = "";
  let activeGenericFieldIds = [];
  const genericFieldRefs = {};

  const GENERIC_PANE_CONFIG = {
    "prompt-templates": {
      subtitle: "Configure prompt behavior for intent decomposition and code generation.",
      fields: [
        {
          id: "defaultTemplate",
          label: "Default Template",
          type: "select",
          options: [
            { value: "5g-ltm", label: "5G LTM Standard" },
            { value: "feature-first", label: "Feature First" },
            { value: "safe-refactor", label: "Safe Refactor" }
          ],
          defaultValue: "5g-ltm"
        },
        { id: "maxContextChunks", label: "Max Context Chunks", type: "number", defaultValue: 14, min: 1, max: 100 },
        { id: "includeAcceptanceCriteria", label: "Include Acceptance Criteria", type: "toggle", defaultValue: true }
      ]
    },
    "repository-config": {
      subtitle: "Source-control defaults used during generated commit/push operations.",
      fields: [
        { id: "remoteName", label: "Remote Name", type: "text", defaultValue: "origin", placeholder: "origin" },
        { id: "defaultBranch", label: "Default Branch", type: "text", defaultValue: "main", placeholder: "main" },
        { id: "autoFetchBeforeRuns", label: "Auto Fetch Before Runs", type: "toggle", defaultValue: true }
      ]
    },
    "branch-strategy": {
      subtitle: "Decide branch naming and merge behavior for generated changes.",
      fields: [
        { id: "branchPrefix", label: "Branch Prefix", type: "text", defaultValue: "feature/", placeholder: "feature/" },
        {
          id: "mergeMode",
          label: "Merge Mode",
          type: "select",
          options: [
            { value: "rebase", label: "Rebase" },
            { value: "squash", label: "Squash" },
            { value: "merge-commit", label: "Merge Commit" }
          ],
          defaultValue: "rebase"
        },
        { id: "deleteMergedBranches", label: "Delete Merged Branches", type: "toggle", defaultValue: true }
      ]
    },
    "ci-cd-pipeline": {
      subtitle: "Configure CI pipeline trigger and timeout policy.",
      fields: [
        { id: "provider", label: "Pipeline Provider", type: "select", options: [{ value: "jenkins", label: "Jenkins" }, { value: "github-actions", label: "GitHub Actions" }], defaultValue: "jenkins" },
        { id: "jobName", label: "Job Name", type: "text", defaultValue: "unified-platform-build", placeholder: "pipeline job name" },
        { id: "timeoutMinutes", label: "Timeout (minutes)", type: "number", defaultValue: 40, min: 1, max: 240 }
      ]
    },
    "test-runner": {
      subtitle: "Set test framework and retry behavior for generated validation runs.",
      fields: [
        { id: "framework", label: "Framework", type: "select", options: [{ value: "gtest", label: "GTest" }, { value: "pytest", label: "Pytest" }, { value: "jest", label: "Jest" }], defaultValue: "gtest" },
        { id: "parallelWorkers", label: "Parallel Workers", type: "number", defaultValue: 4, min: 1, max: 32 },
        { id: "rerunFailedTests", label: "Re-run Failed Tests", type: "toggle", defaultValue: true }
      ]
    },
    "rca-engine": {
      subtitle: "RCA engine controls for bug triage and fix-loop heuristics.",
      fields: [
        { id: "mode", label: "RCA Mode", type: "select", options: [{ value: "auto", label: "Auto" }, { value: "guided", label: "Guided" }, { value: "manual", label: "Manual" }], defaultValue: "auto" },
        { id: "maxFixAttempts", label: "Max Fix Attempts", type: "number", defaultValue: 3, min: 1, max: 10 },
        { id: "captureCrashDumps", label: "Capture Crash Dumps", type: "toggle", defaultValue: true }
      ]
    },
    "quality-gates": {
      subtitle: "Define minimum thresholds required before push approval.",
      fields: [
        { id: "minCoverage", label: "Min Coverage (%)", type: "number", defaultValue: 80, min: 1, max: 100 },
        { id: "maxCriticalIssues", label: "Max Critical Issues", type: "number", defaultValue: 0, min: 0, max: 20 },
        { id: "blockOnLintErrors", label: "Block On Lint Errors", type: "toggle", defaultValue: true }
      ]
    },
    notifications: {
      subtitle: "Notification channels for run status, failures, and approvals.",
      fields: [
        { id: "emailEnabled", label: "Email Notifications", type: "toggle", defaultValue: true },
        { id: "slackWebhook", label: "Slack Webhook URL", type: "text", defaultValue: "", placeholder: "https://hooks.slack.com/services/..." },
        { id: "notifyOn", label: "Notify On", type: "select", options: [{ value: "all", label: "All events" }, { value: "failures", label: "Failures only" }, { value: "none", label: "None" }], defaultValue: "failures" }
      ]
    },
    "workspace-paths": {
      subtitle: "Workspace and output directory paths for codegen and reports.",
      fields: [
        { id: "workspaceRoot", label: "Workspace Root", type: "text", defaultValue: "/home/tcs/Downloads/Unified_Platform_UI", placeholder: "/path/to/workspace" },
        { id: "artifactsPath", label: "Artifacts Path", type: "text", defaultValue: "backend/services/codegen/outputs", placeholder: "relative/path" },
        { id: "tempPath", label: "Temp Path", type: "text", defaultValue: "/tmp/unified-platform", placeholder: "/tmp/unified-platform" }
      ]
    },
    advanced: {
      subtitle: "Advanced behavior for logging and debug flow control.",
      fields: [
        { id: "debugLogs", label: "Enable Debug Logs", type: "toggle", defaultValue: false },
        { id: "safeMode", label: "Safe Mode", type: "toggle", defaultValue: true },
        { id: "maxConcurrentRuns", label: "Max Concurrent Runs", type: "number", defaultValue: 1, min: 1, max: 8 }
      ]
    }
  };

  function flashCursorStatus(message, isWarning = false) {
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.hidden = !message;
    statusEl.classList.toggle("is-warning", isWarning);
    if (statusTimerId) window.clearTimeout(statusTimerId);
    statusTimerId = window.setTimeout(() => {
      statusEl.hidden = true;
      statusEl.textContent = "";
    }, 4500);
  }

  function flashGenericStatus(message, isWarning = false) {
    if (!genericStatus) return;
    genericStatus.textContent = message;
    genericStatus.hidden = !message;
    genericStatus.classList.toggle("is-warning", isWarning);
    if (genericStatusTimerId) window.clearTimeout(genericStatusTimerId);
    genericStatusTimerId = window.setTimeout(() => {
      genericStatus.hidden = true;
      genericStatus.textContent = "";
    }, 4500);
  }

  function showAiPane() {
    paneAi.removeAttribute("hidden");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function flashTemplatesStatus(message, isWarning = false) {
    if (!templatesStatus) return;
    templatesStatus.textContent = message;
    templatesStatus.hidden = !message;
    templatesStatus.classList.toggle("is-warning", isWarning);
    if (templatesStatusTimerId) window.clearTimeout(templatesStatusTimerId);
    templatesStatusTimerId = window.setTimeout(() => {
      templatesStatus.hidden = true;
      templatesStatus.textContent = "";
    }, 4500);
  }

  function showPromptTemplatesPane() {
    panePromptTemplates.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showRepositoryConfigPane() {
    paneRepositoryConfig.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showBranchStrategyPane() {
    paneBranchStrategy.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showCicdPipelinePane() {
    paneCicdPipeline.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showTestRunnerPane() {
    paneTestRunner.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showRcaEnginePane() {
    paneRcaEngine.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showQualityGatesPane() {
    paneQualityGates.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showNotificationsPane() {
    paneNotifications.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showWorkspacePathsPane() {
    paneWorkspacePaths.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function showAdvancedPane() {
    paneAdvanced.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneGeneric.setAttribute("hidden", "");
  }

  function readGenericInputs() {
    const values = {};
    activeGenericFieldIds.forEach((fieldId) => {
      const field = genericFieldRefs[fieldId];
      if (!field) return;
      if (field.type === "checkbox") {
        values[fieldId] = Boolean(field.checked);
      } else if (field.type === "number") {
        const parsed = Number(field.value);
        values[fieldId] = Number.isFinite(parsed) ? parsed : 0;
      } else {
        values[fieldId] = String(field.value ?? "").trim();
      }
    });
    return values;
  }

  function renderGenericFields(paneKey, config, storedValues = {}) {
    genericFields.innerHTML = "";
    activeGenericFieldIds = [];
    Object.keys(genericFieldRefs).forEach((k) => delete genericFieldRefs[k]);
    (config.fields || []).forEach((fieldDef) => {
      const fieldId = `settings-generic-${paneKey}-${fieldDef.id}`;
      const wrap = document.createElement("label");
      wrap.className = "settings-field";
      const label = document.createElement("span");
      label.className = "settings-field-label";
      label.textContent = fieldDef.label;
      wrap.appendChild(label);
      let input;
      const storedValue = Object.prototype.hasOwnProperty.call(storedValues, fieldDef.id)
        ? storedValues[fieldDef.id]
        : fieldDef.defaultValue;
      if (fieldDef.type === "select") {
        input = document.createElement("select");
        input.className = "settings-select";
        (fieldDef.options || []).forEach((opt) => {
          const option = document.createElement("option");
          option.value = opt.value;
          option.textContent = opt.label;
          input.appendChild(option);
        });
        input.value = String(storedValue ?? fieldDef.defaultValue ?? "");
      } else if (fieldDef.type === "toggle") {
        wrap.classList.add("settings-field-inline");
        const toggleWrap = document.createElement("div");
        toggleWrap.className = "settings-toggle-wrap";
        const toggle = document.createElement("label");
        toggle.className = "settings-toggle";
        input = document.createElement("input");
        input.type = "checkbox";
        input.role = "switch";
        input.checked = Boolean(storedValue);
        const slider = document.createElement("span");
        slider.className = "settings-toggle-slider";
        slider.setAttribute("aria-hidden", "true");
        toggle.appendChild(input);
        toggle.appendChild(slider);
        const copy = document.createElement("div");
        copy.className = "settings-toggle-copy";
        const copyTitle = document.createElement("span");
        copyTitle.className = "settings-field-label";
        copyTitle.textContent = fieldDef.label;
        copy.appendChild(copyTitle);
        if (fieldDef.hint) {
          const hint = document.createElement("span");
          hint.className = "settings-field-hint settings-field-hint-inline";
          hint.textContent = fieldDef.hint;
          copy.appendChild(hint);
        }
        toggleWrap.appendChild(toggle);
        toggleWrap.appendChild(copy);
        wrap.innerHTML = "";
        wrap.appendChild(toggleWrap);
      } else {
        input = document.createElement("input");
        input.className = "settings-input";
        input.type = fieldDef.type === "number" ? "number" : "text";
        if (fieldDef.type === "number") {
          if (typeof fieldDef.min === "number") input.min = String(fieldDef.min);
          if (typeof fieldDef.max === "number") input.max = String(fieldDef.max);
          input.value = String(storedValue ?? fieldDef.defaultValue ?? 0);
        } else {
          input.value = String(storedValue ?? fieldDef.defaultValue ?? "");
          if (fieldDef.placeholder) input.placeholder = fieldDef.placeholder;
        }
      }
      input.id = fieldId;
      wrap.appendChild(input);
      if (fieldDef.hint && fieldDef.type !== "toggle") {
        const hint = document.createElement("span");
        hint.className = "settings-field-hint";
        hint.textContent = fieldDef.hint;
        wrap.appendChild(hint);
      }
      genericFields.appendChild(wrap);
      genericFieldRefs[fieldDef.id] = input;
      activeGenericFieldIds.push(fieldDef.id);
    });
  }

  function showGenericPane(title, paneKey) {
    const config = GENERIC_PANE_CONFIG[paneKey];
    if (!config) return;
    const stored = readPlatformSettings();
    const storedValues = stored[paneKey] && typeof stored[paneKey] === "object" ? stored[paneKey] : {};
    genericTitle.textContent = title;
    genericSubtitle.textContent = config.subtitle;
    renderGenericFields(paneKey, config, storedValues);
    activeGenericPaneKey = paneKey;
    if (genericStatus) {
      genericStatus.hidden = true;
      genericStatus.textContent = "";
    }
    paneGeneric.removeAttribute("hidden");
    paneAi.setAttribute("hidden", "");
    panePromptTemplates.setAttribute("hidden", "");
    paneRepositoryConfig.setAttribute("hidden", "");
    paneBranchStrategy.setAttribute("hidden", "");
    paneCicdPipeline.setAttribute("hidden", "");
    paneTestRunner.setAttribute("hidden", "");
    paneRcaEngine.setAttribute("hidden", "");
    paneQualityGates.setAttribute("hidden", "");
    paneNotifications.setAttribute("hidden", "");
    paneWorkspacePaths.setAttribute("hidden", "");
    paneAdvanced.setAttribute("hidden", "");
  }

  function syncNavActive(activeBtn) {
    navButtons.forEach((b) => {
      if (activeBtn instanceof HTMLElement && b instanceof HTMLElement) {
        b.classList.toggle("active", b === activeBtn);
      }
    });
  }

  function updateEngineRowFlags(selectedValue) {
    engineInputs.forEach((input) => {
      const flag = input.closest(".settings-engine-option")?.querySelector(".settings-engine-flag");
      if (!flag || !(flag instanceof HTMLElement)) return;
      flag.classList.remove("flag-active", "flag-available", "flag-configure");
      if (input.value === selectedValue) {
        flag.classList.add("flag-active");
        flag.textContent = "Active";
      } else if (input.value === "custom-llm-endpoint") {
        flag.classList.add("flag-configure");
        flag.textContent = "Configure";
      } else {
        flag.classList.add("flag-available");
        flag.textContent = "Available";
      }
    });
  }

  const ENGINE_META = {
    "cursor-cli": {
      title: "Cursor CLI Settings",
      pathLabel: "Cursor Executable Path",
      pathHint: "Used to invoke Cursor CLI headlessly per intent run.",
      pathPlaceholder: "path\\to\\cursor-agent",
      status: "Connected",
      icon: { kind: "img", alt: "Cursor", src: "./assets/engine-icons/cursor.svg", className: "engine-cursor" }
    },
    "github-copilot-agent": {
      title: "GitHub Copilot Settings",
      pathLabel: "Copilot Agent Path",
      pathHint: "Path or command used to invoke Copilot agent mode.",
      pathPlaceholder: "path\\to\\copilot-agent",
      status: "Available",
      icon: { kind: "img", alt: "GitHub Copilot", src: "./assets/engine-icons/githubcopilot.svg", className: "engine-github" }
    },
    "amazon-q-developer": {
      title: "Amazon Q Settings",
      pathLabel: "Q Developer CLI Path",
      pathHint: "Path used to invoke Amazon Q Developer CLI.",
      pathPlaceholder: "path\\to\\q",
      status: "Available",
      icon: { kind: "img", alt: "AWS", src: "./assets/engine-icons/aws.svg", className: "engine-awsq" }
    },
    "claude-code": {
      title: "Claude Code Settings",
      pathLabel: "Claude CLI Path",
      pathHint: "Path or command used to invoke Claude Code CLI.",
      pathPlaceholder: "path\\to\\claude",
      status: "Available",
      icon: { kind: "img", alt: "Anthropic", src: "./assets/engine-icons/anthropic.svg", className: "engine-claude" }
    },
    "custom-llm-endpoint": {
      title: "Custom Endpoint Settings",
      pathLabel: "Endpoint / Runner Path",
      pathHint: "Use an executable or endpoint runner for custom model invocation.",
      pathPlaceholder: "https://api.example.com/v1",
      status: "Configure",
      icon: { kind: "text", value: "API", className: "engine-custom" }
    }
  };

  function getCurrentEngine() {
    const checked = document.querySelector('input[name="active-coding-engine"]:checked');
    return checked && "value" in checked ? checked.value : "cursor-cli";
  }

  function applyEngineConfigToForm(engine, config) {
    const merged = {
      path: "",
      outputFormat: "text",
      trustMode: "trust",
      forceOverwrite: true,
      ...(config || {})
    };
    pathInput.value = String(merged.path ?? "");
    if (["text", "json", "patch"].includes(String(merged.outputFormat))) {
      outSel.value = String(merged.outputFormat);
    } else {
      outSel.value = "text";
    }
    if (["trust", "review"].includes(String(merged.trustMode))) {
      trustSel.value = String(merged.trustMode);
    } else {
      trustSel.value = "trust";
    }
    forceChk.checked = Boolean(merged.forceOverwrite);
    const meta = ENGINE_META[engine] || ENGINE_META["cursor-cli"];
    engineSettingsTitle.textContent = meta.title;
    enginePathLabel.textContent = meta.pathLabel;
    enginePathHint.textContent = meta.pathHint;
    pathInput.placeholder = meta.pathPlaceholder;

    engineRuntimeIcon.classList.remove(
      "settings-panel-icon-gear",
      "engine-cursor",
      "engine-github",
      "engine-awsq",
      "engine-claude",
      "engine-custom"
    );
    const icon = meta.icon || { kind: "text", value: "?", className: "engine-custom" };
    engineRuntimeIcon.classList.add(icon.className);
    if (icon.kind === "img") {
      engineRuntimeIcon.innerHTML = `<img class="settings-panel-icon-img" src="${icon.src}" alt="${icon.alt}" />`;
    } else if (icon.kind === "text") {
      engineRuntimeIcon.innerHTML = `<span class="settings-panel-icon-text">${icon.value}</span>`;
    }
  }

  function getSavedEngineConfig(stored, engine) {
    const engineConfigs =
      stored.engineConfigs && typeof stored.engineConfigs === "object" ? stored.engineConfigs : {};
    if (engineConfigs[engine] && typeof engineConfigs[engine] === "object") {
      return engineConfigs[engine];
    }
    if (engine === "cursor-cli" && stored.cursorCli && typeof stored.cursorCli === "object") {
      return stored.cursorCli;
    }
    return null;
  }

  function syncCursorSection() {
    const value = getCurrentEngine();
    cursorPanel.removeAttribute("hidden");
    if (connLabel) {
      const meta = ENGINE_META[value] || ENGINE_META["cursor-cli"];
      connLabel.textContent = meta.status;
      if (meta.status === "Connected") {
        connPill.classList.remove("is-disconnected");
      } else {
        connPill.classList.add("is-disconnected");
      }
    }
    const stored = readPlatformSettings();
    const cfg = getSavedEngineConfig(stored, value);
    if (cfg) {
      const normalized = { ...cfg };
      if (normalized.outputFormat === "unified") normalized.outputFormat = "patch";
      if (normalized.trustMode === "no-trust") normalized.trustMode = "review";
      applyEngineConfigToForm(value, normalized);
    } else {
      applyEngineConfigToForm(value, {});
    }
    updateEngineRowFlags(value);
  }

  const stored = readPlatformSettings();
  const allowedEngines = new Set([
    "cursor-cli",
    "github-copilot-agent",
    "amazon-q-developer",
    "claude-code",
    "custom-llm-endpoint"
  ]);
  if (typeof stored.activeEngine === "string" && allowedEngines.has(stored.activeEngine)) {
    const match = document.querySelector(`input[name="active-coding-engine"][value="${stored.activeEngine}"]`);
    if (match && "checked" in match) {
      match.checked = true;
    }
  }
  if (stored.cursorCli && typeof stored.cursorCli === "object") {
    const c = stored.cursorCli;
    if (typeof c.path === "string") pathInput.value = c.path;
    if (typeof c.outputFormat === "string") {
      const normalizedOutput = c.outputFormat === "unified" ? "patch" : c.outputFormat;
      if (["text", "json", "patch"].includes(normalizedOutput)) {
        outSel.value = normalizedOutput;
      }
    }
    if (typeof c.trustMode === "string") {
      const normalizedTrust = c.trustMode === "no-trust" ? "review" : c.trustMode;
      if (["trust", "review"].includes(normalizedTrust)) {
        trustSel.value = normalizedTrust;
      }
    }
    if (typeof c.forceOverwrite === "boolean") forceChk.checked = c.forceOverwrite;
  }

  syncCursorSection();

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!(btn instanceof HTMLElement)) return;
      const pane = btn.dataset.settingsPane;
      const title = btn.dataset.paneTitle || "Settings";
      syncNavActive(btn);
      if (pane === "ai-engine") {
        showAiPane();
      } else if (pane === "prompt-templates") {
        showPromptTemplatesPane();
      } else if (pane === "repository-config") {
        showRepositoryConfigPane();
      } else if (pane === "branch-strategy") {
        showBranchStrategyPane();
      } else if (pane === "ci-cd-pipeline") {
        showCicdPipelinePane();
      } else if (pane === "test-runner") {
        showTestRunnerPane();
      } else if (pane === "rca-engine") {
        showRcaEnginePane();
      } else if (pane === "quality-gates") {
        showQualityGatesPane();
      } else if (pane === "notifications") {
        showNotificationsPane();
      } else if (pane === "workspace-paths") {
        showWorkspacePathsPane();
      } else if (pane === "advanced") {
        showAdvancedPane();
      } else {
        showGenericPane(title, pane);
      }
    });
  });

  engineInputs.forEach((input) => {
    input.addEventListener("change", () => {
      syncCursorSection();
    });
  });

  saveBtn.addEventListener("click", () => {
    const activeEngine = getCurrentEngine();
    const next = readPlatformSettings();
    next.activeEngine = activeEngine;
    const cfg = {
      path: pathInput.value.trim(),
      outputFormat: outSel.value,
      trustMode: trustSel.value,
      forceOverwrite: forceChk.checked
    };
    next.engineConfigs =
      next.engineConfigs && typeof next.engineConfigs === "object" ? next.engineConfigs : {};
    next.engineConfigs[activeEngine] = cfg;
    if (activeEngine === "cursor-cli") {
      next.cursorCli = cfg;
    }
    writePlatformSettings(next);
    flashCursorStatus(`${(ENGINE_META[activeEngine] || ENGINE_META["cursor-cli"]).title} saved locally.`);
  });

  testBtn.addEventListener("click", () => {
    const activeEngine = getCurrentEngine();
    const p = pathInput.value.trim();
    if (!p) {
      flashCursorStatus("Set executable path before testing.", true);
      return;
    }
    flashCursorStatus(
      `${(ENGINE_META[activeEngine] || ENGINE_META["cursor-cli"]).title}: path looks valid (no shell execution in this build). Save to persist.`
    );
  });

  function readPromptTemplatesSettings() {
    const stored = readPlatformSettings();
    const pt = stored.promptTemplates && typeof stored.promptTemplates === "object" ? stored.promptTemplates : {};
    const routes = Array.isArray(pt.routes) ? pt.routes : null;
    return {
      defaultTemplate: typeof pt.defaultTemplate === "string" ? pt.defaultTemplate : "code_prompt_generic_3gpp.txt",
      directory: typeof pt.directory === "string" ? pt.directory : "C:\\Users\\Chandu\\Desktop\\prompts\\",
      routes:
        routes ||
        [
          { tag: "F1AP", file: "code_prompt_f1ap.txt" },
          { tag: "RRC", file: "code_prompt_rrc.txt" },
          { tag: "default", file: "code_prompt_generic_3gpp.txt" }
        ]
    };
  }

  function writePromptTemplatesSettings(nextSettings) {
    const next = readPlatformSettings();
    next.promptTemplates = nextSettings;
    writePlatformSettings(next);
  }

  function renderPromptTemplateRoutes(routes) {
    if (!templatesRoutes) return;
    templatesRoutes.innerHTML = "";
    routes.forEach((route, idx) => {
      const row = document.createElement("div");
      row.className = "settings-template-route";
      const tag = document.createElement("span");
      tag.className = "settings-template-tag";
      tag.textContent = String(route.tag || "").trim() || "TAG";
      const arrow = document.createElement("span");
      arrow.className = "settings-template-arrow";
      arrow.textContent = "→";
      const file = document.createElement("div");
      file.className = "settings-template-file";
      file.textContent = String(route.file || "").trim() || "(select template)";
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "settings-template-remove";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        const cur = readPromptTemplatesSettings();
        cur.routes.splice(idx, 1);
        writePromptTemplatesSettings(cur);
        renderPromptTemplateRoutes(cur.routes);
        flashTemplatesStatus("Route removed. Click Save Changes to persist UI state.");
      });
      row.append(tag, arrow, file, remove);
      templatesRoutes.appendChild(row);
    });
  }

  function loadPromptTemplatesUi() {
    if (!templatesDefaultSel || !templatesDirInput || !templatesRoutes) return;
    const pt = readPromptTemplatesSettings();
    templatesDefaultSel.value = pt.defaultTemplate;
    templatesDirInput.value = pt.directory;
    renderPromptTemplateRoutes(pt.routes);
  }

  if (templatesAddBtn) {
    templatesAddBtn.addEventListener("click", () => {
      const pt = readPromptTemplatesSettings();
      pt.routes.push({ tag: "NEW", file: pt.defaultTemplate });
      renderPromptTemplateRoutes(pt.routes);
      writePromptTemplatesSettings(pt);
      flashTemplatesStatus("Added a route (NEW). Update it in storage in next iteration.");
    });
  }

  if (templatesSaveBtn) {
    templatesSaveBtn.addEventListener("click", () => {
      if (!templatesDefaultSel || !templatesDirInput) return;
      const pt = readPromptTemplatesSettings();
      pt.defaultTemplate = templatesDefaultSel.value;
      pt.directory = templatesDirInput.value.trim();
      writePromptTemplatesSettings(pt);
      flashTemplatesStatus("Prompt template settings saved locally.");
    });
  }

  if (templatesDefaultSel) {
    templatesDefaultSel.addEventListener("change", () => {
      const pt = readPromptTemplatesSettings();
      pt.defaultTemplate = templatesDefaultSel.value;
      writePromptTemplatesSettings(pt);
      flashTemplatesStatus("Default template updated (saved locally).");
    });
  }

  if (templatesDirInput) {
    templatesDirInput.addEventListener("change", () => {
      const pt = readPromptTemplatesSettings();
      pt.directory = templatesDirInput.value.trim();
      writePromptTemplatesSettings(pt);
      flashTemplatesStatus("Template directory updated (saved locally).");
    });
  }

  loadPromptTemplatesUi();

  function flashRepoStatus(message, isWarning = false) {
    if (!repoStatus) return;
    repoStatus.textContent = message;
    repoStatus.hidden = !message;
    repoStatus.classList.toggle("is-warning", isWarning);
    if (repoStatusTimerId) window.clearTimeout(repoStatusTimerId);
    repoStatusTimerId = window.setTimeout(() => {
      repoStatus.hidden = true;
      repoStatus.textContent = "";
    }, 4500);
  }

  function setRepoReachable(reachable, labelText) {
    if (!repoReachablePill || !repoReachableLabel) return;
    repoReachablePill.classList.toggle("is-disconnected", !reachable);
    repoReachableLabel.textContent = labelText;
  }

  function readRepoSettings() {
    const stored = readPlatformSettings();
    const rc = stored.repositoryConfig && typeof stored.repositoryConfig === "object" ? stored.repositoryConfig : {};
    return {
      url: typeof rc.url === "string" ? rc.url : (repoUrlInput?.value || ""),
      workspace: typeof rc.workspace === "string" ? rc.workspace : (repoWorkspaceInput?.value || ""),
      stack: typeof rc.stack === "string" ? rc.stack : "c-cpp-embedded",
      auth: typeof rc.auth === "string" ? rc.auth : "ssh",
      reachable: typeof rc.reachable === "boolean" ? rc.reachable : false
    };
  }

  function writeRepoSettings(nextSettings) {
    const next = readPlatformSettings();
    next.repositoryConfig = nextSettings;
    writePlatformSettings(next);
  }

  function loadRepoUi() {
    if (!repoUrlInput || !repoWorkspaceInput || !repoStackSel || !repoAuthSel) return;
    const rc = readRepoSettings();
    repoUrlInput.value = rc.url;
    repoWorkspaceInput.value = rc.workspace;
    repoStackSel.value = rc.stack;
    repoAuthSel.value = rc.auth;
    setRepoReachable(Boolean(rc.reachable), rc.reachable ? "Reachable" : "Unverified");
  }

  if (repoTestBtn) {
    repoTestBtn.addEventListener("click", () => {
      if (!repoUrlInput) return;
      const url = repoUrlInput.value.trim();
      if (!url) {
        setRepoReachable(false, "Unverified");
        flashRepoStatus("Repository URL is required to test connectivity.", true);
        return;
      }
      const looksLikeGit =
        url.startsWith("http://") ||
        url.startsWith("https://") ||
        url.startsWith("ssh://") ||
        url.includes("@") ||
        url.endsWith(".git");
      if (!looksLikeGit) {
        setRepoReachable(false, "Unverified");
        flashRepoStatus("URL does not look like a valid git clone URL.", true);
        return;
      }
      setRepoReachable(true, "Reachable");
      const rc = readRepoSettings();
      rc.url = url;
      rc.reachable = true;
      writeRepoSettings(rc);
      flashRepoStatus("Connectivity looks good (UI-only check). Save to persist.");
    });
  }

  if (repoSaveBtn) {
    repoSaveBtn.addEventListener("click", () => {
      if (!repoUrlInput || !repoWorkspaceInput || !repoStackSel || !repoAuthSel) return;
      const rc = readRepoSettings();
      rc.url = repoUrlInput.value.trim();
      rc.workspace = repoWorkspaceInput.value.trim();
      rc.stack = repoStackSel.value;
      rc.auth = repoAuthSel.value;
      writeRepoSettings(rc);
      flashRepoStatus("Repository configuration saved locally.");
    });
  }

  loadRepoUi();

  function flashBranchStatus(message, isWarning = false) {
    if (!branchStatus) return;
    branchStatus.textContent = message;
    branchStatus.hidden = !message;
    branchStatus.classList.toggle("is-warning", isWarning);
    if (branchStatusTimerId) window.clearTimeout(branchStatusTimerId);
    branchStatusTimerId = window.setTimeout(() => {
      branchStatus.hidden = true;
      branchStatus.textContent = "";
    }, 4500);
  }

  function readBranchSettings() {
    const stored = readPlatformSettings();
    const bs = stored.branchStrategy && typeof stored.branchStrategy === "object" ? stored.branchStrategy : {};
    return {
      baseBranch: typeof bs.baseBranch === "string" ? bs.baseBranch : "develop",
      branchPattern: typeof bs.branchPattern === "string" ? bs.branchPattern : "feature/{intent_id}_{date}",
      autoPush: typeof bs.autoPush === "boolean" ? bs.autoPush : true,
      commitTemplate: typeof bs.commitTemplate === "string" ? bs.commitTemplate : "feat: {intent_summary}"
    };
  }

  function writeBranchSettings(nextSettings) {
    const next = readPlatformSettings();
    next.branchStrategy = nextSettings;
    writePlatformSettings(next);
  }

  function loadBranchUi() {
    if (!branchBaseInput || !branchPatternInput || !branchAutoPush || !branchCommitTemplate) return;
    const bs = readBranchSettings();
    branchBaseInput.value = bs.baseBranch;
    branchPatternInput.value = bs.branchPattern;
    branchAutoPush.checked = Boolean(bs.autoPush);
    branchCommitTemplate.value = bs.commitTemplate;
  }

  if (branchSaveBtn) {
    branchSaveBtn.addEventListener("click", () => {
      if (!branchBaseInput || !branchPatternInput || !branchAutoPush || !branchCommitTemplate) return;
      const base = branchBaseInput.value.trim();
      const pattern = branchPatternInput.value.trim();
      if (!base) {
        flashBranchStatus("Base Branch is required.", true);
        return;
      }
      if (!pattern || !pattern.includes("{intent_id}")) {
        flashBranchStatus("Feature Branch Pattern must include {intent_id}.", true);
        return;
      }
      writeBranchSettings({
        baseBranch: base,
        branchPattern: pattern,
        autoPush: branchAutoPush.checked,
        commitTemplate: branchCommitTemplate.value.trim()
      });
      flashBranchStatus("Branch strategy saved locally.");
    });
  }

  loadBranchUi();

  function flashCicdStatus(message, isWarning = false) {
    if (!cicdStatus) return;
    cicdStatus.textContent = message;
    cicdStatus.hidden = !message;
    cicdStatus.classList.toggle("is-warning", isWarning);
    if (cicdStatusTimerId) window.clearTimeout(cicdStatusTimerId);
    cicdStatusTimerId = window.setTimeout(() => {
      cicdStatus.hidden = true;
      cicdStatus.textContent = "";
    }, 4500);
  }

  function readCicdSettings() {
    const stored = readPlatformSettings();
    const c = stored.cicdPipeline && typeof stored.cicdPipeline === "object" ? stored.cicdPipeline : {};
    return {
      system: typeof c.system === "string" ? c.system : "jenkins",
      url: typeof c.url === "string" ? c.url : "http://jenkins.internal.tcs.com:8080",
      job: typeof c.job === "string" ? c.job : "oai-cu-ltm-build",
      token: typeof c.token === "string" ? c.token : "****************",
      triggerOnPush: typeof c.triggerOnPush === "boolean" ? c.triggerOnPush : true,
      pollSeconds: typeof c.pollSeconds === "number" ? c.pollSeconds : 15
    };
  }

  function writeCicdSettings(nextSettings) {
    const next = readPlatformSettings();
    next.cicdPipeline = nextSettings;
    writePlatformSettings(next);
  }

  function syncCicdPill(system) {
    if (!cicdPill || !cicdPillLabel) return;
    if (system === "jenkins") {
      cicdPill.classList.remove("is-disconnected");
      cicdPillLabel.textContent = "Jenkins Connected";
    } else {
      cicdPill.classList.add("is-disconnected");
      cicdPillLabel.textContent = "Available";
    }
  }

  function loadCicdUi() {
    const c = readCicdSettings();
    cicdInputs.forEach((i) => {
      if (i.value === c.system) i.checked = true;
    });
    if (cicdUrl) cicdUrl.value = c.url;
    if (cicdJob) cicdJob.value = c.job;
    if (cicdToken) cicdToken.value = c.token;
    if (cicdTrigger) cicdTrigger.checked = c.triggerOnPush;
    if (cicdPoll) cicdPoll.value = String(c.pollSeconds);
    syncCicdPill(c.system);
  }

  cicdInputs.forEach((input) => {
    input.addEventListener("change", () => {
      const checked = Array.from(cicdInputs).find((i) => i.checked);
      syncCicdPill(checked ? checked.value : "jenkins");
    });
  });

  if (cicdTestBtn) {
    cicdTestBtn.addEventListener("click", () => {
      const url = (cicdUrl?.value || "").trim();
      const job = (cicdJob?.value || "").trim();
      if (!url || !job) {
        flashCicdStatus("Jenkins URL and Job Name are required for connectivity test.", true);
        return;
      }
      flashCicdStatus("Connection looks valid (UI-only check).");
    });
  }

  if (cicdSaveBtn) {
    cicdSaveBtn.addEventListener("click", () => {
      const checked = Array.from(cicdInputs).find((i) => i.checked);
      const pollVal = Number(cicdPoll?.value || "15");
      writeCicdSettings({
        system: checked ? checked.value : "jenkins",
        url: (cicdUrl?.value || "").trim(),
        job: (cicdJob?.value || "").trim(),
        token: String(cicdToken?.value || ""),
        triggerOnPush: Boolean(cicdTrigger?.checked),
        pollSeconds: Number.isFinite(pollVal) && pollVal > 0 ? pollVal : 15
      });
      flashCicdStatus("CI/CD pipeline settings saved locally.");
    });
  }

  loadCicdUi();

  function flashTestRunnerStatus(message, isWarning = false) {
    if (!testRunnerStatus) return;
    testRunnerStatus.textContent = message;
    testRunnerStatus.hidden = !message;
    testRunnerStatus.classList.toggle("is-warning", isWarning);
    if (testRunnerStatusTimerId) window.clearTimeout(testRunnerStatusTimerId);
    testRunnerStatusTimerId = window.setTimeout(() => {
      testRunnerStatus.hidden = true;
      testRunnerStatus.textContent = "";
    }, 4500);
  }

  function syncTestRunnerFlags(activeValue) {
    testRunnerInputs.forEach((input) => {
      const flag = input.closest(".settings-engine-option")?.querySelector(".settings-engine-flag");
      if (!flag || !(flag instanceof HTMLElement)) return;
      flag.classList.remove("flag-active", "flag-available");
      if (input.value === activeValue) {
        flag.classList.add("flag-active");
        flag.textContent = "Active";
      } else {
        flag.classList.add("flag-available");
        flag.textContent = "Available";
      }
    });
  }

  function readTestRunnerSettings() {
    const stored = readPlatformSettings();
    const t = stored.testRunner && typeof stored.testRunner === "object" ? stored.testRunner : {};
    return {
      framework: typeof t.framework === "string" ? t.framework : "ctest",
      command: typeof t.command === "string" ? t.command : "ctest --output-on-failure -R TC_F1AP",
      resultFormat: typeof t.resultFormat === "string" ? t.resultFormat : "junit-xml",
      timeoutSeconds: typeof t.timeoutSeconds === "number" ? t.timeoutSeconds : 300
    };
  }

  function writeTestRunnerSettings(nextSettings) {
    const next = readPlatformSettings();
    next.testRunner = nextSettings;
    writePlatformSettings(next);
  }

  function loadTestRunnerUi() {
    const t = readTestRunnerSettings();
    testRunnerInputs.forEach((i) => {
      if (i.value === t.framework) i.checked = true;
    });
    if (testRunnerCommand) testRunnerCommand.value = t.command;
    if (testRunnerResultFormat) testRunnerResultFormat.value = t.resultFormat;
    if (testRunnerTimeout) testRunnerTimeout.value = String(t.timeoutSeconds);
    syncTestRunnerFlags(t.framework);
  }

  testRunnerInputs.forEach((input) => {
    input.addEventListener("change", () => {
      const checked = Array.from(testRunnerInputs).find((i) => i.checked);
      syncTestRunnerFlags(checked ? checked.value : "ctest");
    });
  });

  if (testRunnerSaveBtn) {
    testRunnerSaveBtn.addEventListener("click", () => {
      const checked = Array.from(testRunnerInputs).find((i) => i.checked);
      const timeoutVal = Number(testRunnerTimeout?.value || "300");
      const command = String(testRunnerCommand?.value || "").trim();
      if (!command) {
        flashTestRunnerStatus("Test Command is required.", true);
        return;
      }
      writeTestRunnerSettings({
        framework: checked ? checked.value : "ctest",
        command,
        resultFormat: String(testRunnerResultFormat?.value || "junit-xml"),
        timeoutSeconds: Number.isFinite(timeoutVal) && timeoutVal >= 10 ? timeoutVal : 300
      });
      flashTestRunnerStatus("Test runner settings saved locally.");
    });
  }

  loadTestRunnerUi();

  function flashRcaStatus(message, isWarning = false) {
    if (!rcaStatus) return;
    rcaStatus.textContent = message;
    rcaStatus.hidden = !message;
    rcaStatus.classList.toggle("is-warning", isWarning);
    if (rcaStatusTimerId) window.clearTimeout(rcaStatusTimerId);
    rcaStatusTimerId = window.setTimeout(() => {
      rcaStatus.hidden = true;
      rcaStatus.textContent = "";
    }, 4500);
  }

  function readRcaSettings() {
    const stored = readPlatformSettings();
    const r = stored.rcaEngine && typeof stored.rcaEngine === "object" ? stored.rcaEngine : {};
    return {
      autoTrigger: typeof r.autoTrigger === "boolean" ? r.autoTrigger : true,
      autoApplyHighConfidence: typeof r.autoApplyHighConfidence === "boolean" ? r.autoApplyHighConfidence : true,
      reviewBelowThreshold: typeof r.reviewBelowThreshold === "boolean" ? r.reviewBelowThreshold : true,
      searchGitHistory: typeof r.searchGitHistory === "boolean" ? r.searchGitHistory : true,
      autoApplyThreshold: typeof r.autoApplyThreshold === "number" ? r.autoApplyThreshold : 80,
      maxIterations: typeof r.maxIterations === "number" ? r.maxIterations : 3,
      onMaxIterations: typeof r.onMaxIterations === "string" ? r.onMaxIterations : "pause-notify"
    };
  }

  function writeRcaSettings(nextSettings) {
    const next = readPlatformSettings();
    next.rcaEngine = nextSettings;
    writePlatformSettings(next);
  }

  function loadRcaUi() {
    if (
      !rcaAutoTrigger ||
      !rcaAutoApply ||
      !rcaReviewBelowThreshold ||
      !rcaSearchHistory ||
      !rcaAutoApplyThreshold ||
      !rcaMaxIterations ||
      !rcaMaxIterationAction
    ) {
      return;
    }
    const r = readRcaSettings();
    rcaAutoTrigger.checked = r.autoTrigger;
    rcaAutoApply.checked = r.autoApplyHighConfidence;
    rcaReviewBelowThreshold.checked = r.reviewBelowThreshold;
    rcaSearchHistory.checked = r.searchGitHistory;
    rcaAutoApplyThreshold.value = String(r.autoApplyThreshold);
    rcaMaxIterations.value = String(r.maxIterations);
    rcaMaxIterationAction.value = r.onMaxIterations;
  }

  if (rcaSaveBtn) {
    rcaSaveBtn.addEventListener("click", () => {
      if (!rcaAutoApplyThreshold || !rcaMaxIterations) return;
      const threshold = Number(rcaAutoApplyThreshold.value);
      const maxIterations = Number(rcaMaxIterations.value);
      if (!Number.isFinite(threshold) || threshold < 0 || threshold > 100) {
        flashRcaStatus("Auto-apply threshold must be between 0 and 100.", true);
        return;
      }
      if (!Number.isFinite(maxIterations) || maxIterations < 1 || maxIterations > 20) {
        flashRcaStatus("Max RCA iterations must be between 1 and 20.", true);
        return;
      }
      writeRcaSettings({
        autoTrigger: Boolean(rcaAutoTrigger?.checked),
        autoApplyHighConfidence: Boolean(rcaAutoApply?.checked),
        reviewBelowThreshold: Boolean(rcaReviewBelowThreshold?.checked),
        searchGitHistory: Boolean(rcaSearchHistory?.checked),
        autoApplyThreshold: Math.round(threshold),
        maxIterations: Math.round(maxIterations),
        onMaxIterations: String(rcaMaxIterationAction?.value || "pause-notify")
      });
      flashRcaStatus("RCA engine settings saved locally.");
    });
  }

  loadRcaUi();

  function flashQualityStatus(message, isWarning = false) {
    if (!qualityStatus) return;
    qualityStatus.textContent = message;
    qualityStatus.hidden = !message;
    qualityStatus.classList.toggle("is-warning", isWarning);
    if (qualityStatusTimerId) window.clearTimeout(qualityStatusTimerId);
    qualityStatusTimerId = window.setTimeout(() => {
      qualityStatus.hidden = true;
      qualityStatus.textContent = "";
    }, 4500);
  }

  function readQualitySettings() {
    const stored = readPlatformSettings();
    const q = stored.qualityGates && typeof stored.qualityGates === "object" ? stored.qualityGates : {};
    return {
      minPassRate: typeof q.minPassRate === "number" ? q.minPassRate : 100,
      minCoverage: typeof q.minCoverage === "number" ? q.minCoverage : 80,
      maxWarnings: typeof q.maxWarnings === "number" ? q.maxWarnings : 5,
      blockOnFail: typeof q.blockOnFail === "boolean" ? q.blockOnFail : true
    };
  }

  function writeQualitySettings(nextSettings) {
    const next = readPlatformSettings();
    next.qualityGates = nextSettings;
    writePlatformSettings(next);
  }

  function loadQualityUi() {
    if (!qualityMinPassRate || !qualityMinCoverage || !qualityMaxWarnings || !qualityBlockOnFail) return;
    const q = readQualitySettings();
    qualityMinPassRate.value = String(q.minPassRate);
    qualityMinCoverage.value = String(q.minCoverage);
    qualityMaxWarnings.value = String(q.maxWarnings);
    qualityBlockOnFail.checked = q.blockOnFail;
  }

  if (qualitySaveBtn) {
    qualitySaveBtn.addEventListener("click", () => {
      if (!qualityMinPassRate || !qualityMinCoverage || !qualityMaxWarnings || !qualityBlockOnFail) return;
      const minPassRate = Number(qualityMinPassRate.value);
      const minCoverage = Number(qualityMinCoverage.value);
      const maxWarnings = Number(qualityMaxWarnings.value);
      if (!Number.isFinite(minPassRate) || minPassRate < 0 || minPassRate > 100) {
        flashQualityStatus("Min test pass rate must be between 0 and 100.", true);
        return;
      }
      if (!Number.isFinite(minCoverage) || minCoverage < 0 || minCoverage > 100) {
        flashQualityStatus("Min code coverage must be between 0 and 100.", true);
        return;
      }
      if (!Number.isFinite(maxWarnings) || maxWarnings < 0 || maxWarnings > 1000) {
        flashQualityStatus("Max build warnings must be between 0 and 1000.", true);
        return;
      }
      writeQualitySettings({
        minPassRate: Math.round(minPassRate),
        minCoverage: Math.round(minCoverage),
        maxWarnings: Math.round(maxWarnings),
        blockOnFail: Boolean(qualityBlockOnFail.checked)
      });
      flashQualityStatus("Quality gate settings saved locally.");
    });
  }

  loadQualityUi();

  function flashNotificationsStatus(message, isWarning = false) {
    if (!notificationsStatus) return;
    notificationsStatus.textContent = message;
    notificationsStatus.hidden = !message;
    notificationsStatus.classList.toggle("is-warning", isWarning);
    if (notificationsStatusTimerId) window.clearTimeout(notificationsStatusTimerId);
    notificationsStatusTimerId = window.setTimeout(() => {
      notificationsStatus.hidden = true;
      notificationsStatus.textContent = "";
    }, 4500);
  }

  function readNotificationsSettings() {
    const stored = readPlatformSettings();
    const n = stored.notifications && typeof stored.notifications === "object" ? stored.notifications : {};
    return {
      emailOnCompletion: typeof n.emailOnCompletion === "boolean" ? n.emailOnCompletion : true,
      emailOnFailure: typeof n.emailOnFailure === "boolean" ? n.emailOnFailure : true,
      teamsEnabled: typeof n.teamsEnabled === "boolean" ? n.teamsEnabled : false,
      slackEnabled: typeof n.slackEnabled === "boolean" ? n.slackEnabled : false,
      emailAddress: typeof n.emailAddress === "string" ? n.emailAddress : "chandu.vangal@tcs.com"
    };
  }

  function writeNotificationsSettings(nextSettings) {
    const next = readPlatformSettings();
    next.notifications = nextSettings;
    writePlatformSettings(next);
  }

  function loadNotificationsUi() {
    if (
      !notifyEmailCompletion ||
      !notifyEmailFailure ||
      !notifyTeamsEnabled ||
      !notifySlackEnabled ||
      !notifyEmailAddress
    ) {
      return;
    }
    const n = readNotificationsSettings();
    notifyEmailCompletion.checked = n.emailOnCompletion;
    notifyEmailFailure.checked = n.emailOnFailure;
    notifyTeamsEnabled.checked = n.teamsEnabled;
    notifySlackEnabled.checked = n.slackEnabled;
    notifyEmailAddress.value = n.emailAddress;
  }

  if (notificationsSaveBtn) {
    notificationsSaveBtn.addEventListener("click", () => {
      if (
        !notifyEmailCompletion ||
        !notifyEmailFailure ||
        !notifyTeamsEnabled ||
        !notifySlackEnabled ||
        !notifyEmailAddress
      ) {
        return;
      }
      const email = String(notifyEmailAddress.value || "").trim();
      const emailLooksValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
      if ((notifyEmailCompletion.checked || notifyEmailFailure.checked) && !emailLooksValid) {
        flashNotificationsStatus("Enter a valid email address when email notifications are enabled.", true);
        return;
      }
      writeNotificationsSettings({
        emailOnCompletion: Boolean(notifyEmailCompletion.checked),
        emailOnFailure: Boolean(notifyEmailFailure.checked),
        teamsEnabled: Boolean(notifyTeamsEnabled.checked),
        slackEnabled: Boolean(notifySlackEnabled.checked),
        emailAddress: email
      });
      flashNotificationsStatus("Notification settings saved locally.");
    });
  }

  loadNotificationsUi();

  function flashPathsStatus(message, isWarning = false) {
    if (!pathsStatus) return;
    pathsStatus.textContent = message;
    pathsStatus.hidden = !message;
    pathsStatus.classList.toggle("is-warning", isWarning);
    if (pathsStatusTimerId) window.clearTimeout(pathsStatusTimerId);
    pathsStatusTimerId = window.setTimeout(() => {
      pathsStatus.hidden = true;
      pathsStatus.textContent = "";
    }, 4500);
  }

  function readPathsSettings() {
    const stored = readPlatformSettings();
    const w = stored.workspacePaths && typeof stored.workspacePaths === "object" ? stored.workspacePaths : {};
    return {
      platformRoot:
        typeof w.platformRoot === "string" ? w.platformRoot : "/home/tcs/Downloads/Unified_Platform_UI",
      promptOutputDir: typeof w.promptOutputDir === "string" ? w.promptOutputDir : "./prompts/generated/",
      logOutputDir: typeof w.logOutputDir === "string" ? w.logOutputDir : "./logs/",
      testResultsDir: typeof w.testResultsDir === "string" ? w.testResultsDir : "./test_results/"
    };
  }

  function writePathsSettings(nextSettings) {
    const next = readPlatformSettings();
    next.workspacePaths = nextSettings;
    writePlatformSettings(next);
  }

  function loadPathsUi() {
    if (!pathPlatformRoot || !pathPromptOutput || !pathLogOutput || !pathTestResults) return;
    const w = readPathsSettings();
    pathPlatformRoot.value = w.platformRoot;
    pathPromptOutput.value = w.promptOutputDir;
    pathLogOutput.value = w.logOutputDir;
    pathTestResults.value = w.testResultsDir;
  }

  if (pathsSaveBtn) {
    pathsSaveBtn.addEventListener("click", () => {
      if (!pathPlatformRoot || !pathPromptOutput || !pathLogOutput || !pathTestResults) return;
      const platformRoot = String(pathPlatformRoot.value || "").trim();
      if (!platformRoot) {
        flashPathsStatus("Platform root is required.", true);
        return;
      }
      writePathsSettings({
        platformRoot,
        promptOutputDir: String(pathPromptOutput.value || "").trim(),
        logOutputDir: String(pathLogOutput.value || "").trim(),
        testResultsDir: String(pathTestResults.value || "").trim()
      });
      flashPathsStatus("Workspace path settings saved locally.");
    });
  }

  loadPathsUi();

  const ADVANCED_DEFAULTS = {
    dryRunMode: false,
    verboseLogging: true,
    persistSessionState: true,
    version: "UIP v1.0.0-beta - Build 20260508"
  };

  function flashAdvancedStatus(message, isWarning = false) {
    if (!advancedStatus) return;
    advancedStatus.textContent = message;
    advancedStatus.hidden = !message;
    advancedStatus.classList.toggle("is-warning", isWarning);
    if (advancedStatusTimerId) window.clearTimeout(advancedStatusTimerId);
    advancedStatusTimerId = window.setTimeout(() => {
      advancedStatus.hidden = true;
      advancedStatus.textContent = "";
    }, 4500);
  }

  function readAdvancedSettings() {
    const stored = readPlatformSettings();
    const a = stored.advanced && typeof stored.advanced === "object" ? stored.advanced : {};
    return {
      dryRunMode:
        typeof a.dryRunMode === "boolean" ? a.dryRunMode : ADVANCED_DEFAULTS.dryRunMode,
      verboseLogging:
        typeof a.verboseLogging === "boolean"
          ? a.verboseLogging
          : ADVANCED_DEFAULTS.verboseLogging,
      persistSessionState:
        typeof a.persistSessionState === "boolean"
          ? a.persistSessionState
          : ADVANCED_DEFAULTS.persistSessionState,
      version:
        typeof a.version === "string" && a.version.trim()
          ? a.version
          : ADVANCED_DEFAULTS.version
    };
  }

  function writeAdvancedSettings(nextSettings) {
    const next = readPlatformSettings();
    next.advanced = nextSettings;
    writePlatformSettings(next);
  }

  function applyAdvancedValues(values) {
    if (advancedDryRun) advancedDryRun.checked = Boolean(values.dryRunMode);
    if (advancedVerboseLogging) advancedVerboseLogging.checked = Boolean(values.verboseLogging);
    if (advancedPersistSession) advancedPersistSession.checked = Boolean(values.persistSessionState);
    if (advancedVersion) advancedVersion.textContent = String(values.version || ADVANCED_DEFAULTS.version);
  }

  function loadAdvancedUi() {
    if (!advancedDryRun || !advancedVerboseLogging || !advancedPersistSession || !advancedVersion) return;
    applyAdvancedValues(readAdvancedSettings());
  }

  if (advancedSaveBtn) {
    advancedSaveBtn.addEventListener("click", () => {
      if (!advancedDryRun || !advancedVerboseLogging || !advancedPersistSession || !advancedVersion) return;
      writeAdvancedSettings({
        dryRunMode: Boolean(advancedDryRun.checked),
        verboseLogging: Boolean(advancedVerboseLogging.checked),
        persistSessionState: Boolean(advancedPersistSession.checked),
        version: String(advancedVersion.textContent || ADVANCED_DEFAULTS.version)
      });
      flashAdvancedStatus("Advanced settings saved locally.");
    });
  }

  if (advancedResetBtn) {
    advancedResetBtn.addEventListener("click", () => {
      applyAdvancedValues(ADVANCED_DEFAULTS);
      writeAdvancedSettings(ADVANCED_DEFAULTS);
      flashAdvancedStatus("Advanced settings reset to defaults.");
    });
  }

  loadAdvancedUi();

  genericSaveBtn.addEventListener("click", () => {
    if (!activeGenericPaneKey) return;
    const next = readPlatformSettings();
    next[activeGenericPaneKey] = readGenericInputs();
    writePlatformSettings(next);
    flashGenericStatus("Settings saved locally in this browser.");
  });

  genericResetBtn.addEventListener("click", () => {
    if (!activeGenericPaneKey) return;
    const config = GENERIC_PANE_CONFIG[activeGenericPaneKey];
    const defaults = {};
    (config.fields || []).forEach((field) => {
      defaults[field.id] = field.defaultValue;
    });
    renderGenericFields(activeGenericPaneKey, config, defaults);
    flashGenericStatus("Default values restored. Click Save Changes to persist.");
  });
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

initSettingsUi();

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

centerTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const paneId = tab.dataset.centerPane || "logs-outputs";
    setActiveCenterPane(paneId);
  });
});

setActiveCenterPane("logs-outputs");
setReviewActionsVisible(false);
setCommitMessageVisible(false);
renderMilestones(getInitialMilestones());
setPipelineVisibility(false);
renderInitialLogView();
refreshActionButtons();
