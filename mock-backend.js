const express = require("express");
const cors = require("cors");

const sessions = new Map();

function uid() {
  return `sess_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
}

function createMilestones() {
  return [
    { id: "feature_validation", label: "Feature Validation", status: "not_completed" },
    { id: "knowledge_retrieval", label: "Knowledge Retrieval", status: "not_completed" },
    { id: "template_orchestrator", label: "Template Orchestrator", status: "not_completed" },
    { id: "self_learning_agent", label: "Self Learning Agent", status: "not_completed" },
    { id: "prompt_generation", label: "Prompt Generation", status: "not_completed" },
    { id: "code_generation", label: "Code Generation", status: "not_completed" },
    { id: "commit_push", label: "Commit & Push", status: "not_completed" },
    { id: "jenkins_checkout", label: "Checkout Branch", status: "not_completed" },
    { id: "test_script_generation", label: "Test Script Generation", status: "not_completed" },
    { id: "build_compile", label: "Build / Compile", status: "not_completed" },
    { id: "rca_build_fix", label: "RCA Build Fix Loop", status: "not_completed" },
    { id: "runtime_execute", label: "Run / Execute", status: "not_completed" },
    { id: "rca_runtime_fix", label: "RCA Runtime Fix Loop", status: "not_completed" },
    { id: "test_scoring", label: "Test Execution & Scoring", status: "not_completed" }
  ];
}

function getProgressSummary(milestones = [], state = "not_started") {
  const total = milestones.length;
  const completed = milestones.filter((m) => m.status === "completed").length;
  const inProgress = milestones.filter((m) => m.status === "in_progress").length;
  const failed = milestones.filter((m) => m.status === "failed").length;
  const pending = milestones.filter((m) => m.status === "not_completed").length;
  const completionPercent = total ? Math.round((completed / total) * 100) : 0;

  return {
    overall_state: state,
    completion_percent: completionPercent,
    counts: {
      total,
      completed,
      in_progress: inProgress,
      failed,
      not_completed: pending
    },
    stages: milestones.map((m) => ({
      stage_id: m.id,
      stage_name: m.label,
      status: m.status
    }))
  };
}

function pushLog(session, stage, message, type = "info") {
  session.logs.push({
    timestamp: new Date().toISOString(),
    stage,
    type,
    message
  });
}

function setStatus(session, milestoneId, status) {
  const item = session.milestones.find((m) => m.id === milestoneId);
  if (item) item.status = status;
}

function markFailed(session, milestoneId, message) {
  setStatus(session, milestoneId, "failed");
  session.state = "failed";
  session.failed = true;
  pushLog(session, milestoneId, message, "error");
}

function buildPrompt(intent) {
  return [
    "You are an OAI 5G code generation agent.",
    `Implement the requested feature intent: "${intent}".`,
    "Use existing OAI patterns and preserve interface compatibility.",
    "Touch only relevant protocol modules and include unit/integration hooks.",
    "Return clean, compilable code and concise rationale."
  ].join("\n");
}

function createCodePreview(intent) {
  return [
    "diff --git a/src/rrc/rrcHandler.js b/src/rrc/rrcHandler.js",
    "index 2a4d1ef..8f71c34 100644",
    "--- a/src/rrc/rrcHandler.js",
    "+++ b/src/rrc/rrcHandler.js",
    "@@",
    `-// TODO: implement feature intent handling`,
    `+// Feature intent: ${intent}`,
    "+function handleFeatureIntent(ctx) {",
    "+  if (!ctx || !ctx.ueId) return { accepted: false, reason: \"missing_ue\" };",
    "+  return { accepted: true, action: \"trigger_rrc_update\" };",
    "+}",
    "-function validateIntent(intent) {",
    "-  return !!intent;",
    "-}",
    "+function validateIntent(intent) {",
    "+  return typeof intent === \"string\" && intent.trim().length > 3;",
    "+}",
    "",
    "diff --git a/src/ngap/sessionManager.js b/src/ngap/sessionManager.js",
    "index 0d01f19..5be22aa 100644",
    "--- a/src/ngap/sessionManager.js",
    "+++ b/src/ngap/sessionManager.js",
    "@@",
    "-export function createSession(payload) {",
    "-  return { id: payload.id, state: \"NEW\" };",
    "-}",
    "+export function createSession(payload) {",
    "+  const base = { id: payload.id, state: \"NEW\" };",
    "+  if (payload.intentTag) base.intentTag = payload.intentTag;",
    "+  return base;",
    "+}",
    "-export function closeSession(session) {",
    "-  return { ...session, state: \"CLOSED\" };",
    "-}",
    "+export function closeSession(session) {",
    "+  return { ...session, state: \"TERMINATED\", closedAt: Date.now() };",
    "+}"
  ].join("\n");
}

const STEP_DELAY_MS = 2000;

function scheduleStep(stepIndex, fn) {
  setTimeout(fn, stepIndex * STEP_DELAY_MS);
}

function startPromptFlow(session) {
  session.state = "prompt_generation_in_progress";
  let step = 0;

  scheduleStep(step++, () => {
    setStatus(session, "feature_validation", "in_progress");
    pushLog(session, "feature_validation", "Intent accepted and protocol discovery started.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "feature_validation", "completed");
    setStatus(session, "knowledge_retrieval", "in_progress");
    pushLog(session, "knowledge_retrieval", "ETSI specs identified and relevant sections fetched.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "knowledge_retrieval", "completed");
    setStatus(session, "template_orchestrator", "in_progress");
    pushLog(session, "template_orchestrator", "Template selected from protocol template pool.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "template_orchestrator", "completed");
    setStatus(session, "self_learning_agent", "in_progress");
    pushLog(session, "self_learning_agent", "Running ambiguity checks with validation rules.");
  });

  scheduleStep(step++, () => {
    if (/fail_prompt|fail_generation/i.test(session.intent)) {
      markFailed(
        session,
        "self_learning_agent",
        "Validation failure encountered. Prompt generation halted."
      );
      return;
    }

    if (session.ambiguityQuestion) {
      session.state = "ambiguity_required";
      pushLog(session, "self_learning_agent", session.ambiguityQuestion, "warning");
      return;
    }

    setStatus(session, "self_learning_agent", "completed");
    setStatus(session, "prompt_generation", "in_progress");
    pushLog(session, "prompt_generation", "Composing final prompt from context-filled template.");
  });

  scheduleStep(step++, () => {
    if (session.state !== "prompt_generation_in_progress") return;
    session.prompt = buildPrompt(session.intent);
    setStatus(session, "prompt_generation", "completed");
    session.state = "prompt_ready";
    pushLog(session, "prompt_generation", "Prompt generated successfully.");
  });
}

function startResolveFlow(session, resolution) {
  session.state = "prompt_generation_in_progress";
  let step = 0;

  scheduleStep(step++, () => {
    setStatus(session, "self_learning_agent", "in_progress");
    pushLog(session, "self_learning_agent", `User provided ambiguity resolution: ${resolution}`);
  });

  scheduleStep(step++, () => {
    if (/fail|invalid/i.test(resolution)) {
      markFailed(session, "self_learning_agent", "Provided resolution did not satisfy ambiguity rules.");
      return;
    }
    setStatus(session, "self_learning_agent", "completed");
    setStatus(session, "prompt_generation", "in_progress");
    pushLog(session, "prompt_generation", "Rebuilding prompt after ambiguity resolution.");
  });

  scheduleStep(step++, () => {
    if (session.state !== "prompt_generation_in_progress") return;
    session.ambiguityQuestion = null;
    session.ambiguity_items = [];
    session.prompt = buildPrompt(`${session.intent}. Resolved scope: ${resolution}`);
    setStatus(session, "prompt_generation", "completed");
    session.state = "prompt_ready";
    pushLog(session, "prompt_generation", "Prompt generated successfully after ambiguity resolution.");
  });
}

function startCodeGenerationFlow(session) {
  session.state = "code_generation_in_progress";
  let step = 0;

  scheduleStep(step++, () => {
    setStatus(session, "code_generation", "in_progress");
    pushLog(session, "code_generation", "Sending prompt to Cursor CLI for code generation.");
  });

  scheduleStep(step++, () => {
    if (/fail_code|compile_fail/i.test(session.intent)) {
      markFailed(session, "code_generation", "Cursor code generation/build validation failed.");
      return;
    }
    session.codePreview = createCodePreview(session.intent);
    pushLog(session, "code_generation", "Cursor output received.");
  });

  scheduleStep(step++, () => {
    if (session.state !== "code_generation_in_progress") return;
    setStatus(session, "code_generation", "completed");
    session.state = "code_ready_for_review";
  });
}

function startCommitPushFlow(session) {
  session.state = "commit_push_in_progress";
  let step = 0;

  scheduleStep(step++, () => {
    setStatus(session, "commit_push", "in_progress");
    pushLog(session, "commit_push", "Committing generated changes.");
  });

  scheduleStep(step++, () => {
    if (/fail_commit|git_fail/i.test(session.intent)) {
      markFailed(session, "commit_push", "Git commit/push failed due to repository validation.");
      return;
    }
    pushLog(session, "commit_push", "Pushing branch to remote: new_feature.");
  });

  scheduleStep(step++, () => {
    if (session.state !== "commit_push_in_progress") return;
    setStatus(session, "commit_push", "completed");
    session.state = "jenkins_pipeline_in_progress";
    pushLog(session, "commit_push", "Commit & push complete. Triggering Jenkins pipeline.");
    startJenkinsFlow(session);
  });
}

function startJenkinsFlow(session) {
  let step = 0;
  session.jenkins = {
    buildFixAttempt: 0,
    runtimeFixAttempt: 0
  };

  scheduleStep(step++, () => {
    setStatus(session, "jenkins_checkout", "in_progress");
    pushLog(session, "jenkins_checkout", "Jenkins: pulling latest new_feature branch.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "jenkins_checkout", "completed");
    setStatus(session, "test_script_generation", "in_progress");
    pushLog(
      session,
      "test_script_generation",
      "Invoking Test Automation entry script with CodeGen knowledge retrieval context."
    );
  });

  scheduleStep(step++, () => {
    setStatus(session, "test_script_generation", "completed");
    setStatus(session, "build_compile", "in_progress");
    pushLog(session, "build_compile", "Jenkins: starting OAI build/compile.");
  });

  // Build loop: fail once, fix, then pass.
  scheduleStep(step++, () => {
    setStatus(session, "build_compile", "failed");
    setStatus(session, "rca_build_fix", "in_progress");
    session.jenkins.buildFixAttempt += 1;
    pushLog(
      session,
      "rca_build_fix",
      `Build failed. Sending logs to RCA (attempt ${session.jenkins.buildFixAttempt}).`
    );
  });

  scheduleStep(step++, () => {
    setStatus(session, "rca_build_fix", "completed");
    pushLog(session, "rca_build_fix", "RCA suggested patch applied and committed to new_feature.");
    setStatus(session, "build_compile", "in_progress");
    pushLog(session, "build_compile", "Jenkins: re-pulling branch and rebuilding after RCA fix.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "build_compile", "completed");
    setStatus(session, "runtime_execute", "in_progress");
    pushLog(session, "runtime_execute", "Jenkins: executing OAI runtime flow.");
  });

  // Runtime loop: fail once, fix, then pass.
  scheduleStep(step++, () => {
    setStatus(session, "runtime_execute", "failed");
    setStatus(session, "rca_runtime_fix", "in_progress");
    session.jenkins.runtimeFixAttempt += 1;
    pushLog(
      session,
      "rca_runtime_fix",
      `Runtime errors detected. Sending runtime logs to RCA (attempt ${session.jenkins.runtimeFixAttempt}).`
    );
  });

  scheduleStep(step++, () => {
    setStatus(session, "rca_runtime_fix", "completed");
    pushLog(session, "rca_runtime_fix", "RCA runtime patch applied and committed.");
    setStatus(session, "runtime_execute", "in_progress");
    pushLog(session, "runtime_execute", "Jenkins: rerunning runtime after RCA fix.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "runtime_execute", "completed");
    setStatus(session, "test_scoring", "in_progress");
    pushLog(session, "test_scoring", "Executing generated test scripts against runtime logs.");
  });

  scheduleStep(step++, () => {
    setStatus(session, "test_scoring", "completed");
    session.state = "pipeline_done";
    pushLog(session, "test_scoring", "Test scoring complete. Overall score: 91.6");
    pushLog(session, "jenkins_pipeline", "Jenkins pipeline finished successfully.");
  });
}

function startMockBackend(port = 4100) {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/api/common/health", (_req, res) => {
    res.json({ status: "ok", service: "mock-codegen-backend" });
  });

  app.post("/api/codegen/generate", (req, res) => {
    const intent = req.body?.intent?.trim();
    if (!intent) {
      return res.status(400).json({ error: "intent is required" });
    }

    const sessionId = uid();
    const requiresAmbiguity = /handover|mobility|rrc|ambiguous|multi-interface/i.test(intent);
    const ambiguityItems = requiresAmbiguity
      ? [
          "Confirm target protocol scope: RRC (UE–gNB), NGAP, or F1AP.",
          "Deployment shape: SA only, or also NSA / EN-DC?",
          "Any mandatory timers or counters from a specific 3GPP release (e.g. Rel-17 vs Rel-18)?"
        ]
      : [];
    const session = {
      sessionId,
      intent,
      state: requiresAmbiguity ? "ambiguity_required" : "prompt_ready",
      failed: false,
      milestones: createMilestones(),
      logs: [],
      prompt: "",
      codePreview: "",
      ambiguity_items: ambiguityItems,
      ambiguityQuestion: requiresAmbiguity
        ? "Self-learning needs your input on the open points (see panel) before composing the final prompt."
        : null
    };

    sessions.set(sessionId, session);
    startPromptFlow(session);

    if (requiresAmbiguity) {
      return res.json({
        session_id: sessionId,
        state: "prompt_generation_in_progress",
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, "prompt_generation_in_progress")
      });
    }

    return res.json({
      session_id: sessionId,
      state: "prompt_generation_in_progress",
      milestones: session.milestones,
      logs: session.logs,
      progress: getProgressSummary(session.milestones, "prompt_generation_in_progress")
    });
  });

  app.post("/api/codegen/resolve-ambiguities", (req, res) => {
    const sessionId = req.body?.session_id;
    const resolution = req.body?.resolution?.trim();
    const session = sessions.get(sessionId);

    if (!session) return res.status(404).json({ error: "Session not found" });
    if (!resolution) return res.status(400).json({ error: "resolution is required" });
    if (session.failed) {
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }

    if (/fail|invalid/i.test(resolution)) {
      markFailed(session, "self_learning_agent", "Provided resolution did not satisfy ambiguity rules.");
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }

    startResolveFlow(session, resolution);

    res.json({
      session_id: session.sessionId,
      state: "prompt_generation_in_progress",
      milestones: session.milestones,
      logs: session.logs,
      progress: getProgressSummary(session.milestones, "prompt_generation_in_progress")
    });
  });

  app.post("/api/codegen/generate-code", (req, res) => {
    const session = sessions.get(req.body?.session_id);
    if (!session) return res.status(404).json({ error: "Session not found" });
    if (!session.prompt) return res.status(400).json({ error: "Prompt not ready" });
    if (session.failed) {
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }

    startCodeGenerationFlow(session);

    res.json({
      session_id: session.sessionId,
      state: "code_generation_in_progress",
      milestones: session.milestones,
      logs: session.logs,
      progress: getProgressSummary(session.milestones, "code_generation_in_progress")
    });
  });

  app.post("/api/codegen/update-prompt", (req, res) => {
    const session = sessions.get(req.body?.session_id);
    const prompt = req.body?.prompt?.trim();
    if (!session) return res.status(404).json({ error: "Session not found" });
    if (!prompt) return res.status(400).json({ error: "prompt is required" });
    if (!session.prompt) return res.status(400).json({ error: "Prompt not ready" });
    if (session.failed) {
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }

    session.prompt = prompt;
    pushLog(session, "prompt_generation", "Prompt updated by user.");

    res.json({
      session_id: session.sessionId,
      state: session.state,
      milestones: session.milestones,
      logs: session.logs,
      prompt: session.prompt,
      progress: getProgressSummary(session.milestones, session.state)
    });
  });

  app.post("/api/codegen/commit-push", (req, res) => {
    const session = sessions.get(req.body?.session_id);
    if (!session) return res.status(404).json({ error: "Session not found" });
    if (session.failed) {
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }

    startCommitPushFlow(session);

    res.json({
      session_id: session.sessionId,
      state: "commit_push_in_progress",
      milestones: session.milestones,
      logs: session.logs,
      progress: getProgressSummary(session.milestones, "commit_push_in_progress")
    });
  });

  app.post("/api/jenkins/start", (req, res) => {
    const session = sessions.get(req.body?.session_id);
    if (!session) return res.status(404).json({ error: "Session not found" });
    if (session.failed) {
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }
    if (session.state === "pipeline_done" || session.state === "jenkins_pipeline_in_progress") {
      return res.json({
        session_id: session.sessionId,
        state: session.state,
        milestones: session.milestones,
        logs: session.logs,
        progress: getProgressSummary(session.milestones, session.state)
      });
    }
    session.state = "jenkins_pipeline_in_progress";
    pushLog(session, "jenkins_pipeline", "Manual Jenkins pipeline trigger received.");
    startJenkinsFlow(session);
    return res.json({
      session_id: session.sessionId,
      state: session.state,
      milestones: session.milestones,
      logs: session.logs,
      progress: getProgressSummary(session.milestones, session.state)
    });
  });

  app.get("/api/codegen/progress/:sessionId", (req, res) => {
    const session = sessions.get(req.params.sessionId);
    if (!session) return res.status(404).json({ error: "Session not found" });
    res.json({
      session_id: session.sessionId,
      progress: getProgressSummary(session.milestones, session.state)
    });
  });

  app.get("/api/codegen/session/:sessionId", (req, res) => {
    const session = sessions.get(req.params.sessionId);
    if (!session) return res.status(404).json({ error: "Session not found" });
    res.json({
      ...session,
      progress: getProgressSummary(session.milestones, session.state)
    });
  });

  return new Promise((resolve) => {
    const server = app.listen(port, () => resolve(server));
  });
}

module.exports = { startMockBackend };
