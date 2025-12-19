const statusText = document.getElementById("statusText");
const scenarioText = document.getElementById("scenarioText");
const optionsList = document.getElementById("optionsList");
const planInput = document.getElementById("planInput");
const planList = document.getElementById("planList");
const perception = document.getElementById("perception");
const constraints = document.getElementById("constraints");
const resources = document.getElementById("resources");
const personalityText = document.getElementById("personalityText");
const socialText = document.getElementById("socialText");
const logText = document.getElementById("logText");
const metricsText = document.getElementById("metricsText");
const temporalBar = document.getElementById("temporalBar");
const temporalValue = document.getElementById("temporalValue");
const rewardExtrinsic = document.getElementById("rewardExtrinsic");
const rewardIntrinsic = document.getElementById("rewardIntrinsic");
const rewardTemporal = document.getElementById("rewardTemporal");
const rewardCausal = document.getElementById("rewardCausal");
const rewardMeta = document.getElementById("rewardMeta");
const rewardEmergent = document.getElementById("rewardEmergent");

let selectedOption = null;
let planSteps = [];

function renderPlan() {
  planList.innerHTML = "";
  planSteps.forEach((step, idx) => {
    const item = document.createElement("div");
    item.textContent = step;
    item.className = "plan-item";
    item.dataset.index = String(idx);
    item.onclick = () => {
      document.querySelectorAll(".plan-item").forEach((node) => node.classList.remove("selected"));
      item.classList.add("selected");
      planList.dataset.selected = String(idx);
    };
    planList.appendChild(item);
  });
}

function buildPlanText() {
  const parts = [];
  if (planSteps.length) {
    parts.push(`Plan: ${planSteps.join(" -> ")}`);
  }
  if (perception.value.trim()) {
    parts.push(`Perception: ${perception.value.trim()}`);
  }
  if (constraints.value.trim()) {
    parts.push(`Constraints: ${constraints.value.trim()}`);
  }
  if (resources.value.trim()) {
    parts.push(`Resources: ${resources.value.trim()}`);
  }
  return parts.length ? parts.join(" | ") : "choose best option";
}

function logLine(text) {
  logText.textContent += `${text}\n`;
  logText.scrollTop = logText.scrollHeight;
}

function formatPersonality(vector) {
  if (!vector || !vector.length) {
    return "No personality state available.";
  }
  return vector.map((val, idx) => `trait_${idx}: ${val.toFixed(3)}`).join("\n");
}

function formatSocial(socialWorld) {
  if (!socialWorld) {
    return "No social world available.";
  }
  return Object.entries(socialWorld)
    .map(([group, stats]) => {
      const trust = (stats.trust ?? 0).toFixed(2);
      const respect = (stats.respect ?? 0).toFixed(2);
      const influence = (stats.influence ?? 0).toFixed(2);
      return `${group}: trust=${trust} respect=${respect} infl=${influence}`;
    })
    .join("\n");
}

function renderState(info) {
  scenarioText.textContent = info.scenario_text || "";
  statusText.textContent = `Step ${info.step || ""} | Age ${info.age || ""} | Stage ${info.stage || ""}`;
  optionsList.innerHTML = "";
  (info.available_options || []).forEach((opt) => {
    const btn = document.createElement("button");
    btn.textContent = opt;
    btn.onclick = () => {
      document.querySelectorAll(".options button").forEach((node) => node.classList.remove("selected"));
      btn.classList.add("selected");
      selectedOption = opt;
    };
    optionsList.appendChild(btn);
  });
  personalityText.textContent = formatPersonality(info.updated_personality || info.personality_state);
  socialText.textContent = formatSocial(info.social_world);
}

function sparkline(values, color) {
  if (!values || !values.length) {
    return "";
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * 100;
    const y = 100 - ((v - min) / range) * 100;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return `
    <svg viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline fill="none" stroke="${color}" stroke-width="2" points="${points.join(" ")}" />
    </svg>
  `;
}

function renderMetrics(payload) {
  const metrics = payload.metrics || {};
  metricsText.textContent = JSON.stringify(metrics, null, 2);
  const temporal = payload.temporal_consistency ?? 0;
  temporalBar.style.width = `${Math.max(0, Math.min(temporal * 100, 100))}%`;
  temporalValue.textContent = temporal.toFixed(3);

  const rewards = payload.reward_components || {};
  rewardExtrinsic.innerHTML = sparkline(rewards.extrinsic || [], "#6be7ff");
  rewardIntrinsic.innerHTML = sparkline(rewards.intrinsic || [], "#2f5bff");
  rewardTemporal.innerHTML = sparkline(rewards.temporal || [], "#7cffc2");
  rewardCausal.innerHTML = sparkline(rewards.causal || [], "#ffc76b");
  rewardMeta.innerHTML = sparkline(rewards.meta || [], "#ff7ab2");
  rewardEmergent.innerHTML = sparkline(rewards.emergent || [], "#b67bff");
}

async function resetEnv() {
  const res = await fetch("/api/reset", { method: "POST" });
  const info = await res.json();
  renderState(info);
  logLine("Reset environment.");
}

async function stepEnv(actionText) {
  const res = await fetch("/api/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_text: actionText }),
  });
  const info = await res.json();
  if (info.error) {
    logLine(`Error: ${info.error}`);
    return;
  }
  renderState(info);
  logLine(
    `[step] action='${actionText.slice(0, 80)}' match=${info.action_match_score} src=${info.action_source} reward=${info.reward}`
  );
}

document.getElementById("addPlanBtn").onclick = () => {
  const text = planInput.value.trim();
  if (!text) return;
  planSteps.push(text);
  planInput.value = "";
  renderPlan();
};

document.getElementById("removePlanBtn").onclick = () => {
  const idx = parseInt(planList.dataset.selected || "-1", 10);
  if (idx >= 0) {
    planSteps.splice(idx, 1);
    planList.dataset.selected = "";
    renderPlan();
  }
};

document.getElementById("clearPlanBtn").onclick = () => {
  planSteps = [];
  planList.dataset.selected = "";
  renderPlan();
};

document.getElementById("stepPlanBtn").onclick = () => {
  stepEnv(buildPlanText());
};

document.getElementById("stepOptionBtn").onclick = () => {
  if (!selectedOption) {
    logLine("Select an option first.");
    return;
  }
  stepEnv(selectedOption);
};

document.getElementById("resetBtn").onclick = () => {
  resetEnv();
};

async function fetchMetrics() {
  const res = await fetch("/api/metrics");
  const payload = await res.json();
  renderMetrics(payload);
}

resetEnv();
fetchMetrics();
setInterval(fetchMetrics, 2000);
