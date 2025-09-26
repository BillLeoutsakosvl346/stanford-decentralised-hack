const METRICS_URL = "http://localhost:8000/metrics";
const LEADERBOARD_URL = "http://localhost:8000/leaderboard";

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function updateProgress(metrics) {
  const processed = metrics.processed_packets || 0;
  const target = metrics.target_packets || 1;
  const ratio = Math.min(processed / target, 1);
  const progressFill = document.getElementById("progress-fill");
  const packetCount = document.getElementById("packet-count");
  const valAccuracy = document.getElementById("val-accuracy");

  progressFill.style.width = `${ratio * 100}%`;
  packetCount.textContent = `${processed} / ${target} packets`;
  const accuracy = metrics.val_accuracy ? (metrics.val_accuracy * 100).toFixed(2) : "0.00";
  valAccuracy.textContent = `Val acc: ${accuracy}%`;
}

function updateLeaderboard(snapshot) {
  const list = document.getElementById("leaderboard-list");
  list.innerHTML = "";
  const entries = snapshot.leaderboard || [];
  entries.slice(0, 10).forEach((entry) => {
    const li = document.createElement("li");
    li.textContent = `${entry.client}: ${entry.credit.toFixed(4)} credits`;
    list.appendChild(li);
  });
}

async function refresh() {
  try {
    const [metrics, leaderboard] = await Promise.all([
      fetchJson(METRICS_URL),
      fetchJson(LEADERBOARD_URL),
    ]);
    updateProgress(metrics);
    updateLeaderboard(leaderboard);
  } catch (err) {
    console.error("Dashboard refresh failed", err);
  }
}

setInterval(refresh, 2000);
refresh();
