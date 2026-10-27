const SESSION_KEY = "spam-detector-v1";

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const statusMeta = document.getElementById("status-meta");
const commentInput = document.getElementById("comment-text");
const analyzeBtn = document.getElementById("btn-analyze");
const errorEl = document.getElementById("error-text");
const resultCard = document.getElementById("result-card");
const resultClass = document.getElementById("result-class");
const barHam = document.getElementById("bar-ham");
const barSpam = document.getElementById("bar-spam");
const pctHam = document.getElementById("pct-ham");
const pctSpam = document.getElementById("pct-spam");

let modelReady = false;
let pollTimer = null;
/** @type {{ classification: string, ham: number, spam: number } | null} */
let lastPrediction = null;

function getSessionPayload() {
  return { comment: commentInput.value, lastPrediction };
}

function persistState() {
  AppSession.save(SESSION_KEY, getSessionPayload());
}

function applyPrediction(pred) {
  resultCard.hidden = false;
  resultClass.textContent = pred.classification.toUpperCase();
  resultClass.className = "result-class " + pred.classification;
  barHam.style.width = `${pred.ham}%`;
  barSpam.style.width = `${pred.spam}%`;
  pctHam.textContent = `${pred.ham}%`;
  pctSpam.textContent = `${pred.spam}%`;
}

function restoreState() {
  const saved = AppSession.load(SESSION_KEY);
  if (!saved) return;
  if (saved.comment != null) commentInput.value = saved.comment;
  if (saved.lastPrediction) {
    lastPrediction = saved.lastPrediction;
    applyPrediction(lastPrediction);
  }
}

AppSession.wireHelpSave(SESSION_KEY, getSessionPayload);
commentInput.addEventListener("input", () => persistState());

function setError(message) {
  if (!message) {
    errorEl.hidden = true;
    errorEl.textContent = "";
    return;
  }
  errorEl.hidden = false;
  errorEl.textContent = message;
}

function applyStatus(data) {
  statusDot.classList.remove("ready", "training", "error");
  if (data.error) {
    statusDot.classList.add("error");
    statusText.textContent = "Model failed to load.";
    statusMeta.hidden = false;
    statusMeta.textContent = data.error;
    modelReady = false;
    analyzeBtn.disabled = true;
    return;
  }
  if (data.ready) {
    statusDot.classList.add("ready");
    statusText.textContent = "Model ready";
    const acc = data.holdout_accuracy ?? data.cv_mean_accuracy;
    statusMeta.hidden = !acc;
    statusMeta.textContent = acc
      ? `Holdout accuracy on training data: ${acc}%`
      : "";
    modelReady = true;
    analyzeBtn.disabled = false;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    return;
  }
  statusDot.classList.add("training");
  statusText.textContent = data.training
    ? "Training model (first run may take a few minutes)…"
    : "Starting model load…";
  statusMeta.hidden = true;
  modelReady = false;
  analyzeBtn.disabled = true;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    applyStatus(data);
  } catch {
    statusDot.classList.add("error");
    statusText.textContent = "Could not reach the server.";
  }
}

async function analyze() {
  setError(null);
  const text = commentInput.value.trim();
  if (!text) {
    setError("Enter comment text first.");
    return;
  }
  analyzeBtn.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Classification failed.");
    lastPrediction = {
      classification: data.classification,
      ham: data.probabilities.ham,
      spam: data.probabilities.spam,
    };
    applyPrediction(lastPrediction);
    persistState();
  } catch (err) {
    setError(err.message || "Classification failed.");
  } finally {
    if (modelReady) analyzeBtn.disabled = false;
  }
}

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    commentInput.value = chip.dataset.text || "";
    commentInput.focus();
    persistState();
  });
});

analyzeBtn.addEventListener("click", () => analyze());
pollTimer = setInterval(pollStatus, 2000);
pollStatus();
restoreState();
