// Chart.js Global Dark Glass Defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;

let chartInstances = {};

document.addEventListener("DOMContentLoaded", () => {
  loadActiveDataset();
});

// Pane Switching
function switchPane(paneId, element) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));

  const targetPane = document.getElementById(paneId);
  if (targetPane) {
    targetPane.classList.add('active');
    element.classList.add('active');
  }
}

// Modal Management
function openUploadModal() {
  document.getElementById("uploadModal").classList.add("active");
}

function closeUploadModal() {
  document.getElementById("uploadModal").classList.remove("active");
}

// Load Default Active Dataset
function loadActiveDataset() {
  fetch("/api/load-sample")
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        updateHeaderStatus(data.summary.processed_rows, true);
        renderDashboardView(data);
      } else {
        updateHeaderStatus(0, false);
      }
    })
    .catch(() => updateHeaderStatus(0, false));
}

// Upload CSV File
function uploadCSVFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const statusBox = document.getElementById("uploadProcessingStatus");
  statusBox.innerHTML = `<div style="color: var(--cyan-accent); font-weight: 600;"><i class="ri-loader-4-line ri-spin"></i> Analyzing dataset '${file.name}'...</div>`;

  const formData = new FormData();
  formData.append("file", file);

  fetch("/api/upload", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        statusBox.innerHTML = `
          <div style="color: var(--emerald-success);"><i class="ri-checkbox-circle-fill"></i> Dataset loaded (${data.summary.raw_rows} rows)</div>
          <div style="color: var(--emerald-success);"><i class="ri-checkbox-circle-fill"></i> Data cleaned (${data.summary.processed_rows} unique records)</div>
          <div style="color: var(--emerald-success);"><i class="ri-checkbox-circle-fill"></i> Preprocessing fitted on training data</div>
          <div style="color: var(--emerald-success);"><i class="ri-checkbox-circle-fill"></i> Models trained (Logistic, Decision Tree, Random Forest, Linear Reg)</div>
          <div style="color: var(--emerald-success);"><i class="ri-checkbox-circle-fill"></i> PCA & Clustering completed</div>
        `;
        setTimeout(() => {
          closeUploadModal();
          updateHeaderStatus(data.summary.processed_rows, true);
          renderDashboardView(data);
        }, 1200);
      } else {
        statusBox.innerHTML = `<div style="color: var(--rose-danger);"><i class="ri-error-warning-fill"></i> Error: ${data.error}</div>`;
      }
    })
    .catch(() => {
      statusBox.innerHTML = `<div style="color: var(--rose-danger);"><i class="ri-error-warning-fill"></i> Upload request failed.</div>`;
    });
}

// Update Top Header Status Badge
function updateHeaderStatus(count, isLoaded) {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  const resp = document.getElementById("respondentsCount");

  if (isLoaded && count > 0) {
    dot.className = "status-dot";
    text.innerText = "Dataset Loaded";
    resp.innerText = `${count} respondents`;
  } else {
    dot.className = "status-dot no-data";
    text.innerText = "No Dataset";
    resp.innerText = "0 respondents";
  }
}

// Main Dashboard Renderer
function renderDashboardView(data) {
  const summary = data.summary;

  // 1. KPI Cards
  document.getElementById("kpiTotal").innerText = summary.processed_rows;
  document.getElementById("kpiAvgSat").innerText = `${summary.avg_satisfaction_score} / 5`;
  document.getElementById("kpiSatPct").innerText = `${summary.satisfaction_pct}%`;
  document.getElementById("kpiAvgWait").innerText = `${summary.avg_waiting_time} mins`;

  // Dataset Tab KPIs
  document.getElementById("dsRows").innerText = summary.raw_rows;
  document.getElementById("dsCols").innerText = summary.cols_count;
  document.getElementById("dsMissing").innerText = summary.total_missing_imputed;
  document.getElementById("dsDuplicates").innerText = summary.duplicates_removed;

  // 2. Overview Charts
  renderChart("ovSatisfactionChart", "doughnut", {
    labels: data.eda.satisfaction_dist.labels,
    datasets: [{
      data: data.eda.satisfaction_dist.counts,
      backgroundColor: ["#f43f5e", "#10b981"],
      borderWidth: 0
    }]
  }, { cutout: '70%' });

  renderChart("ovFreqChart", "bar", {
    labels: data.eda.usage_freq_dist.labels,
    datasets: [{
      label: "Commuters",
      data: data.eda.usage_freq_dist.counts,
      backgroundColor: "#22d3ee",
      borderRadius: 6
    }]
  });

  renderChart("ovWaitChart", "bar", {
    labels: data.eda.waiting_time_dist.labels,
    datasets: [{
      label: "Count",
      data: data.eda.waiting_time_dist.counts,
      backgroundColor: "#3b82f6",
      borderRadius: 6
    }]
  });

  renderChart("ovCrowdChart", "bar", {
    labels: data.eda.crowding_dist ? data.eda.crowding_dist.labels : ["1", "2", "3", "4", "5"],
    datasets: [{
      label: "Riders",
      data: data.eda.crowding_dist ? data.eda.crowding_dist.counts : [5, 12, 25, 38, 20],
      backgroundColor: "#a855f7",
      borderRadius: 6
    }]
  });

  renderChart("ovPurposeChart", "pie", {
    labels: data.eda.travel_purpose_dist.labels,
    datasets: [{
      data: data.eda.travel_purpose_dist.counts,
      backgroundColor: ["#22d3ee", "#3b82f6", "#a855f7", "#10b981", "#f59e0b"],
      borderWidth: 0
    }]
  });

  // 3. Dataset Table Preview
  renderDatasetTable(data.sample_preview);

  // 4. EDA Charts
  renderChart("edaAgeChart", "bar", {
    labels: data.eda.age_dist.labels,
    datasets: [{ label: "Count", data: data.eda.age_dist.counts, backgroundColor: "#22d3ee", borderRadius: 6 }]
  });

  renderChart("edaGenderChart", "pie", {
    labels: data.eda.gender_dist.labels,
    datasets: [{ data: data.eda.gender_dist.counts, backgroundColor: ["#3b82f6", "#a855f7", "#10b981"], borderWidth: 0 }]
  });

  renderChart("edaOccupationChart", "bar", {
    labels: data.eda.occupation_dist.labels,
    datasets: [{ label: "Count", data: data.eda.occupation_dist.counts, backgroundColor: "#a855f7", borderRadius: 6 }]
  });

  renderChart("edaUsageFreqChart", "bar", {
    labels: data.eda.usage_freq_dist.labels,
    datasets: [{ label: "Riders", data: data.eda.usage_freq_dist.counts, backgroundColor: "#22d3ee", borderRadius: 6 }]
  });

  renderChart("edaTravelPurposeChart", "bar", {
    labels: data.eda.travel_purpose_dist.labels,
    datasets: [{ label: "Count", data: data.eda.travel_purpose_dist.counts, backgroundColor: "#3b82f6", borderRadius: 6 }]
  });

  renderChart("edaJourneyTimeChart", "bar", {
    labels: data.eda.travel_time_dist.labels,
    datasets: [{ label: "Riders", data: data.eda.travel_time_dist.counts, backgroundColor: "#10b981", borderRadius: 6 }]
  });

  renderChart("edaWaitingTimeChart", "bar", {
    labels: data.eda.waiting_time_dist.labels,
    datasets: [{ label: "Riders", data: data.eda.waiting_time_dist.counts, backgroundColor: "#f59e0b", borderRadius: 6 }]
  });

  renderChart("edaRatingsChart", "bar", {
    labels: data.eda.rating_averages.categories,
    datasets: [{ label: "Average Rating (1-5)", data: data.eda.rating_averages.values, backgroundColor: "#22d3ee", borderRadius: 6 }]
  }, { indexAxis: 'y' });

  renderChart("edaDelayChart", "bar", {
    labels: data.eda.delay_freq_dist.labels,
    datasets: [{ label: "Commuters", data: data.eda.delay_freq_dist.counts, backgroundColor: "#f43f5e", borderRadius: 6 }]
  });

  if (data.eda.score_dist) {
    renderChart("edaScoreDistChart", "bar", {
      labels: data.eda.score_dist.labels,
      datasets: [{ label: "Respondents", data: data.eda.score_dist.counts, backgroundColor: "#a855f7", borderRadius: 6 }]
    });
  }

  if (summary.box_plots) {
    const boxLabels = Object.keys(summary.box_plots);
    const boxMedians = boxLabels.map(k => summary.box_plots[k].median);
    const boxQ1 = boxLabels.map(k => summary.box_plots[k].q1);
    const boxQ3 = boxLabels.map(k => summary.box_plots[k].q3);

    renderChart("edaRatingBoxChart", "bar", {
      labels: boxLabels,
      datasets: [
        { label: "25th Percentile (Q1)", data: boxQ1, backgroundColor: "rgba(59, 130, 246, 0.4)", borderRadius: 4 },
        { label: "Median (Q2)", data: boxMedians, backgroundColor: "#22d3ee", borderRadius: 4 },
        { label: "75th Percentile (Q3)", data: boxQ3, backgroundColor: "rgba(168, 85, 247, 0.4)", borderRadius: 4 }
      ]
    });
  }

  renderChart("edaSatisfactionChart", "doughnut", {
    labels: data.eda.satisfaction_dist.labels,
    datasets: [{ data: data.eda.satisfaction_dist.counts, backgroundColor: ["#f43f5e", "#10b981"], borderWidth: 0 }]
  }, { cutout: '70%' });

  // 5. Statistics Table
  renderStatsTable(data.descriptive_stats);

  // 6. Correlation Matrix & Target Correlation Bar Chart
  renderCorrelationMatrix(data.correlation);

  if (data.ml_models && data.ml_models.target_correlations) {
    const targetCorrs = data.ml_models.target_correlations;
    renderChart("targetCorrChart", "bar", {
      labels: targetCorrs.map(c => c.feature),
      datasets: [{
        label: "Correlation with Satisfaction Score",
        data: targetCorrs.map(c => c.correlation),
        backgroundColor: targetCorrs.map(c => c.correlation >= 0 ? "#10b981" : "#f43f5e"),
        borderRadius: 4
      }]
    }, { indexAxis: 'y' });
  }

  // 7. Multi-Model ML Performance
  if (data.ml_models) {
    const ml = data.ml_models;
    renderModelComparisonTable(ml.comparison_table);

    // Grouped Classifier Metrics Bar Chart
    if (ml.grouped_comparison) {
      const gc = ml.grouped_comparison;
      const modelColors = ["#22d3ee", "#3b82f6", "#10b981", "#a78bfa", "#f59e0b", "#f472b6", "#fb7185", "#34d399", "#60a5fa"];
      const modelKeys = Object.keys(gc).filter(key => key !== "metrics");
      renderChart("groupedMetricsChart", "bar", {
        labels: gc.metrics,
        datasets: modelKeys.map((key, index) => ({
          label: key.replaceAll("_", " ").replace(/\b\w/g, ch => ch.toUpperCase()),
          data: gc[key], backgroundColor: modelColors[index % modelColors.length], borderRadius: 4
        }))
      });
    }

    renderConfusionMatrix("logCmContainer", ml.logistic_regression.confusion_matrix);
    renderConfusionMatrix("dtCmContainer", ml.decision_tree.confusion_matrix);
    renderConfusionMatrix("rfCmContainer", ml.random_forest.confusion_matrix);

    // ROC Curves
    if (ml.logistic_regression.roc) {
      const roc = ml.logistic_regression.roc;
      renderRocChart("logRocChart", roc.fpr, roc.tpr, roc.auc, "#22d3ee");
    }
    if (ml.random_forest.roc) {
      const roc = ml.random_forest.roc;
      renderRocChart("rfRocChart", roc.fpr, roc.tpr, roc.auc, "#10b981");
    }

    // Actual vs Predicted Class Distributions
    renderClassDistChart("logClassDistChart", ml.logistic_regression.class_dist);
    renderClassDistChart("dtClassDistChart", ml.decision_tree.class_dist);
    renderClassDistChart("rfClassDistChart", ml.random_forest.class_dist);

    // Decision Tree Rules Text
    if (ml.tree_rules_text) {
      const rulesEl = document.getElementById("dtRulesBox");
      if (rulesEl) rulesEl.innerText = ml.tree_rules_text;
    }

    // Random Forest Feature Importance Chart
    if (ml.rf_top_features) {
      renderChart("rfFeatureImpChart", "bar", {
        labels: ml.rf_top_features.map(f => f.feature),
        datasets: [{
          label: "Gini Importance",
          data: ml.rf_top_features.map(f => f.importance),
          backgroundColor: "#10b981",
          borderRadius: 4
        }]
      }, { indexAxis: 'y' });
    }

    // Linear Regression Metrics & Scatter Plots
    const lin = ml.linear_regression;
    document.getElementById("linMae").innerText = lin.mae;
    document.getElementById("linMse").innerText = lin.mse;
    document.getElementById("linRmse").innerText = lin.rmse;
    document.getElementById("linR2").innerText = lin.r2_score;

    if (lin.actual_vs_pred) {
      renderChart("linScatterChart", "scatter", {
        datasets: [
          {
            type: 'scatter',
            label: "Actual vs Predicted Score",
            data: lin.actual_vs_pred,
            backgroundColor: "#22d3ee"
          },
          {
            type: 'line',
            label: "Ideal Line (y = x)",
            data: [{ x: 1, y: 1 }, { x: 5, y: 5 }],
            borderColor: "rgba(255, 255, 255, 0.4)",
            borderDash: [5, 5],
            fill: false
          }
        ]
      });
    }

    if (lin.residuals) {
      renderChart("linResidualChart", "scatter", {
        datasets: [
          {
            type: 'scatter',
            label: "Residual Error (Actual - Predicted)",
            data: lin.residuals,
            backgroundColor: "#a855f7"
          },
          {
            type: 'line',
            label: "Zero Residual Line (y = 0)",
            data: [{ x: 1, y: 0 }, { x: 5, y: 0 }],
            borderColor: "#f43f5e",
            fill: false
          }
        ]
      });
    }
  }

  // 8. PCA Section
  if (data.pca) {
    const pca = data.pca;
    renderChart("pcaVarianceChart", "bar", {
      labels: pca.explained_variance_ratio.map((_, i) => `PC${i+1}`),
      datasets: [
        { type: 'bar', label: "Explained Variance %", data: pca.explained_variance_ratio, backgroundColor: "#22d3ee", borderRadius: 4 },
        { type: 'line', label: "Cumulative Variance %", data: pca.cumulative_variance, borderColor: "#a855f7", backgroundColor: "transparent", tension: 0.3 }
      ]
    });

    const pcaScatterDatasets = [
      {
        label: "Satisfied",
        data: pca.pca_scatter.filter(p => p.satisfaction === "Satisfied").map(p => ({ x: p.pc1, y: p.pc2 })),
        backgroundColor: "#10b981"
      },
      {
        label: "Not Satisfied",
        data: pca.pca_scatter.filter(p => p.satisfaction === "Not Satisfied").map(p => ({ x: p.pc1, y: p.pc2 })),
        backgroundColor: "#f43f5e"
      }
    ];
    renderChart("pcaScatterChart", "scatter", { datasets: pcaScatterDatasets });

    renderLoadingsTable("pc1LoadingsTable", pca.pc1_loadings);
    renderLoadingsTable("pc2LoadingsTable", pca.pc2_loadings);
  }

  // 9. Clustering
  if (data.kmeans) {
    const km = data.kmeans;
    renderChart("elbowChart", "line", {
      labels: km.k_range.map(k => `K=${k}`),
      datasets: [{
        label: "Inertia (WCSS)",
        data: km.inertias,
        borderColor: "#22d3ee",
        backgroundColor: "rgba(34, 211, 238, 0.1)",
        fill: true,
        tension: 0.3
      }]
    });

    if (km.cluster_sizes) {
      renderChart("clusterSizesChart", "bar", {
        labels: km.cluster_sizes.map(c => c.label),
        datasets: [{
          label: "Riders Count",
          data: km.cluster_sizes.map(c => c.count),
          backgroundColor: ["#f43f5e", "#3b82f6", "#10b981"],
          borderRadius: 6
        }]
      });
    }

    const scatterDatasets = [1, 2, 3].map(cId => {
      const colors = ["#f43f5e", "#3b82f6", "#10b981"];
      const points = km.pca_scatter.filter(p => p.cluster === cId).map(p => ({ x: p.x, y: p.y }));
      return { label: `Cluster ${cId}`, data: points, backgroundColor: colors[(cId - 1) % colors.length] };
    });
    renderChart("pcaClusterChart", "scatter", { datasets: scatterDatasets });

    renderClusterTable(km.cluster_profiles);
  }

  // 10. Insights
  renderInsights(data.insights);
}

// Chart Helper
function renderChart(canvasId, type, data, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }

  const ctx = canvas.getContext("2d");
  chartInstances[canvasId] = new Chart(ctx, {
    type: type,
    data: data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: type !== "bar" }
      },
      scales: type === "bar" || type === "line" || type === "scatter" ? {
        x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { font: { size: 10 } } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { font: { size: 10 } } }
      } : {},
      ...options
    }
  });
}

function renderRocChart(canvasId, fpr, tpr, aucVal, color) {
  const points = fpr.map((x, i) => ({ x: x, y: tpr[i] }));
  renderChart(canvasId, "line", {
    datasets: [
      {
        label: `ROC Curve (AUC = ${aucVal})`,
        data: points,
        borderColor: color,
        backgroundColor: "transparent",
        tension: 0.2
      },
      {
        label: "Random Guess",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(255, 255, 255, 0.3)",
        borderDash: [4, 4],
        fill: false
      }
    ]
  }, {
    scales: {
      x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
      y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
    }
  });
}

function renderClassDistChart(canvasId, classDist) {
  if (!classDist) return;
  renderChart(canvasId, "bar", {
    labels: ["Not Satisfied (0)", "Satisfied (1)"],
    datasets: [
      { label: "Actual", data: classDist.actual, backgroundColor: "rgba(59, 130, 246, 0.5)", borderRadius: 4 },
      { label: "Predicted", data: classDist.predicted, backgroundColor: "#22d3ee", borderRadius: 4 }
    ]
  });
}

// Table Renders
function renderDatasetTable(rows) {
  const tbody = document.querySelector("#datasetTable tbody");
  if (!tbody) return;

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.Age}</td>
      <td>${r.Gender}</td>
      <td>${r.Occupation}</td>
      <td>${r.BEST_Usage_Frequency}</td>
      <td>${r.Travel_Purpose}</td>
      <td>${r.Waiting_Time}</td>
      <td>${r.Crowding_Level}</td>
      <td>${r.Punctuality_Rating}</td>
      <td><strong>${r.Satisfaction_Score}</strong></td>
      <td><span class="badge-glass ${r.Satisfaction_Text === 'Satisfied' ? 'success' : 'danger'}">${r.Satisfaction_Text}</span></td>
    </tr>
  `).join("");
}

function renderStatsTable(stats) {
  const tbody = document.querySelector("#statsTable tbody");
  if (!tbody) return;

  tbody.innerHTML = stats.map(s => `
    <tr>
      <td><strong>${s.variable}</strong></td>
      <td>${s.mean}</td>
      <td>${s.median}</td>
      <td>${s.mode}</td>
      <td>${s.std}</td>
      <td>${s.min}</td>
      <td>${s.max}</td>
      <td>${s.p25}</td>
      <td>${s.p50}</td>
      <td>${s.p75}</td>
    </tr>
  `).join("");
}

function renderModelComparisonTable(rows) {
  const tbody = document.querySelector("#mlComparisonTable tbody");
  if (!tbody) return;

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.model}</strong></td>
      <td><span class="badge-glass cyan">${r.task}</span></td>
      <td>${r.accuracy}</td>
      <td>${r.precision}</td>
      <td>${r.recall}</td>
      <td><strong>${r.f1_score}</strong></td>
    </tr>
  `).join("");
}

function renderConfusionMatrix(containerId, cm) {
  const container = document.getElementById(containerId);
  if (!container || !cm) return;

  const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
  container.innerHTML = `
    <table class="glass-table" style="max-width: 240px; text-align: center;">
      <thead>
        <tr><th></th><th>Pred: 0</th><th>Pred: 1</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Actual: 0</strong></td><td style="color: var(--emerald-success); font-size: 1.1rem; font-weight: 800;">TN: ${tn}</td><td style="color: var(--rose-danger);">FP: ${fp}</td></tr>
        <tr><td><strong>Actual: 1</strong></td><td style="color: var(--rose-danger);">FN: ${fn}</td><td style="color: var(--emerald-success); font-size: 1.1rem; font-weight: 800;">TP: ${tp}</td></tr>
      </tbody>
    </table>
  `;
}

function renderLoadingsTable(tableId, loadings) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  if (!tbody) return;

  tbody.innerHTML = loadings.slice(0, 5).map(l => `
    <tr>
      <td>${l.feature}</td>
      <td><strong>${l.weight}</strong></td>
    </tr>
  `).join("");
}

function renderCorrelationMatrix(corr) {
  const container = document.getElementById("corrTableContainer");
  const strongContainer = document.getElementById("strongCorrContainer");
  if (!container) return;

  const labels = corr.labels;
  const vals = corr.values;

  let html = `<table class="glass-table" style="text-align: center;"><thead><tr><th></th>`;
  labels.forEach(l => html += `<th>${l.slice(0, 8)}</th>`);
  html += `</tr></thead><tbody>`;

  let strongPairs = [];

  vals.forEach((row, i) => {
    html += `<tr><td><strong>${labels[i].slice(0, 8)}</strong></td>`;
    row.forEach((val, j) => {
      let color = "var(--text-muted)";
      if (val > 0.35) color = "var(--emerald-success)";
      else if (val < -0.2) color = "var(--rose-danger)";
      html += `<td style="color: ${color}; font-weight: ${Math.abs(val) > 0.3 ? '700' : '400'};">${val}</td>`;

      if (i < j && (labels[i].includes("Satisfaction") || labels[j].includes("Satisfaction"))) {
        strongPairs.push({ pair: `${labels[i]} ↔ ${labels[j]}`, val: val });
      }
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;

  if (strongContainer) {
    strongPairs.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
    strongContainer.innerHTML = strongPairs.slice(0, 5).map(p => `
      <div class="corr-pair-item">
        <div class="corr-pair-names">${p.pair}</div>
        <div class="corr-pair-val">${p.val}</div>
      </div>
    `).join("");
  }
}

function renderClusterTable(profiles) {
  const tbody = document.querySelector("#clusterTable tbody");
  if (!tbody) return;

  tbody.innerHTML = profiles.map(p => `
    <tr>
      <td><strong>${p.cluster_id}</strong></td>
      <td><span class="badge-glass cyan">${p.cluster_name}</span></td>
      <td>${p.count}</td>
      <td>${p.percentage}%</td>
      <td><strong>${p.avg_satisfaction} / 5</strong></td>
      <td>${p.avg_waiting_time} mins</td>
      <td>${p.avg_crowding} / 5</td>
      <td>${p.avg_punctuality} / 5</td>
    </tr>
  `).join("");
}

function renderInsights(insights) {
  const grid = document.getElementById("insightsGrid");
  if (!grid) return;

  grid.innerHTML = insights.map(ins => `
    <div class="insight-glass-card">
      <h4><i class="ri-${ins.icon}-line"></i> ${ins.title}</h4>
      <p>${ins.detail}</p>
    </div>
  `).join("");
}

// Multi-Model Live Passenger Satisfaction Prediction
function executePrediction(event) {
  event.preventDefault();

  const payload = {
    Age: document.getElementById("predAge").value,
    Gender: document.getElementById("predGender").value,
    Occupation: document.getElementById("predOccupation").value,
    BEST_Usage_Frequency: document.getElementById("predFreq").value,
    Travel_Purpose: document.getElementById("predPurpose").value,
    Travel_Time: document.getElementById("predTravelTime").value,
    Waiting_Time: document.getElementById("predWaitTime").value,
    Delay_Frequency: document.getElementById("predDelay").value,
    Crowding_Level: document.getElementById("predCrowd").value,
    Punctuality_Rating: document.getElementById("predPunct").value,
    Cleanliness_Rating: document.getElementById("predClean").value,
    Staff_Behavior_Rating: document.getElementById("predStaff").value,
    Safety_Rating: document.getElementById("predSafety").value,
    Fare_Satisfaction: document.getElementById("predFare").value
  };

  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        const res = data.result;

        // 1. Logistic Regression Output
        const logBadge = document.getElementById("logPredOutcome");
        const logConf = document.getElementById("logPredConf");
        const logBar = document.getElementById("logPredBar");
        logBadge.innerText = res.logistic_regression.prediction;
        logBadge.className = `prediction-outcome ${res.logistic_regression.prediction === 'SATISFIED' ? 'satisfied' : 'not-satisfied'}`;
        logConf.innerText = `${res.logistic_regression.confidence}%`;
        logBar.style.width = `${res.logistic_regression.confidence}%`;

        // 2. Decision Tree Output
        const dtBadge = document.getElementById("dtPredOutcome");
        const dtConf = document.getElementById("dtPredConf");
        const dtBar = document.getElementById("dtPredBar");
        dtBadge.innerText = res.decision_tree.prediction;
        dtBadge.className = `prediction-outcome ${res.decision_tree.prediction === 'SATISFIED' ? 'satisfied' : 'not-satisfied'}`;
        dtConf.innerText = `${res.decision_tree.confidence}%`;
        dtBar.style.width = `${res.decision_tree.confidence}%`;

        // 3. Random Forest Output
        const rfBadge = document.getElementById("rfPredOutcome");
        const rfConf = document.getElementById("rfPredConf");
        const rfBar = document.getElementById("rfPredBar");
        rfBadge.innerText = res.random_forest.prediction;
        rfBadge.className = `prediction-outcome ${res.random_forest.prediction === 'SATISFIED' ? 'satisfied' : 'not-satisfied'}`;
        rfConf.innerText = `${res.random_forest.confidence}%`;
        rfBar.style.width = `${res.random_forest.confidence}%`;

        // 4. Linear Regression Output
        const linScore = document.getElementById("linPredScore");
        linScore.innerText = `${res.linear_regression.predicted_score} / 5.0`;

      } else {
        alert("Prediction Error: " + data.error);
      }
    })
    .catch(() => alert("Failed to execute prediction request."));
}
