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
    resultCard.hidden = false;
    resultClass.textContent = data.classification.toUpperCase();
    resultClass.className = "result-class " + data.classification;
    const ham = data.probabilities.ham;
    const spam = data.probabilities.spam;
    barHam.style.width = `${ham}%`;
    barSpam.style.width = `${spam}%`;
    pctHam.textContent = `${ham}%`;
    pctSpam.textContent = `${spam}%`;
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
  });
});

analyzeBtn.addEventListener("click", () => analyze());
pollTimer = setInterval(pollStatus, 2000);
pollStatus();
