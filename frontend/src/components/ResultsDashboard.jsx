const API = "http://localhost:8000";

const NON_PERCENTAGE_METRICS = new Set([
  "mae", "mse", "rmse", "mean_absolute_error", "mean_squared_error",
  "root_mean_squared_error", "r2_score", "silhouette_score",
]);
const METADATA_KEYS = new Set([
  "algorithm", "iteration", "task_type", "model_path",
  "train_samples", "test_samples", "hyperparameters", "strategy",
]);

function MetricCard({ label, value, highlight }) {
  let formatted;
  if (typeof value === "number") {
    const isPercentage = value >= 0 && value <= 1 && !NON_PERCENTAGE_METRICS.has(label.toLowerCase());
    formatted = isPercentage ? (value * 100).toFixed(1) + "%" : value.toFixed(4);
  } else {
    formatted = String(value ?? "—");
  }

  return (
    <div className={`rounded-xl p-4 border ${highlight ? "border-brand-500 bg-brand-900/20" : "border-gray-700 bg-gray-800"}`}>
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${highlight ? "text-brand-400" : "text-white"}`}>
        {formatted}
      </div>
    </div>
  );
}

function VerdictBadge({ verdict }) {
  const isPass = verdict === "pass";
  return (
    <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${
      isPass ? "bg-green-900/50 text-green-400 border border-green-700" : "bg-red-900/50 text-red-400 border border-red-700"
    }`}>
      {isPass ? "✓ PASS" : "✗ RETRY"}
    </span>
  );
}

export default function ResultsDashboard({ evaluation, algorithmInfo, sessionId }) {
  if (!evaluation || Object.keys(evaluation).length === 0) return null;

  const best = evaluation.best_model || {};
  const rawMetrics = best.metrics || {};

  // Use the primary_metric from the pipeline's best_model (deterministic key from _primary_metric_key),
  // falling back to the evaluator's choice, then a generic "score".
  const primaryKey = best.primary_metric || evaluation.primary_metric || "score";
  // Read the actual value from script output metrics, not the evaluator's re-interpretation.
  const primaryValue = rawMetrics[primaryKey] ?? evaluation.score;

  const metricEntries = Object.entries(rawMetrics)
    .filter(([k]) => k !== primaryKey && !METADATA_KEYS.has(k))
    .filter(([, v]) => typeof v === "number");

  const bestAlgo = best.algorithm || algorithmInfo?.algorithm || "Model";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">
            {bestAlgo} Results
          </h2>
          <p className="text-gray-400 text-sm mt-0.5">{evaluation.summary}</p>
        </div>
        <VerdictBadge verdict={evaluation.verdict} />
      </div>

      {/* Primary metric + up to 3 secondary metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label={primaryKey.toUpperCase()} value={primaryValue} highlight />
        {metricEntries.slice(0, 3).map(([k, v]) => (
          <MetricCard key={k} label={k.toUpperCase()} value={v} />
        ))}
      </div>

      {/* Strengths & Weaknesses */}
      <div className="grid md:grid-cols-2 gap-4">
        {evaluation.strengths?.length > 0 && (
          <div className="bg-green-950/30 border border-green-800 rounded-xl p-4">
            <div className="text-green-400 font-semibold text-sm mb-2">Strengths</div>
            <ul className="space-y-1">
              {evaluation.strengths.map((s, i) => (
                <li key={i} className="text-green-300 text-sm flex gap-2">
                  <span className="shrink-0">+</span><span>{s}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {evaluation.weaknesses?.length > 0 && (
          <div className="bg-red-950/30 border border-red-800 rounded-xl p-4">
            <div className="text-red-400 font-semibold text-sm mb-2">Weaknesses</div>
            <ul className="space-y-1">
              {evaluation.weaknesses.map((s, i) => (
                <li key={i} className="text-red-300 text-sm flex gap-2">
                  <span className="shrink-0">−</span><span>{s}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Suggestions */}
      {evaluation.suggestions?.length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
          <div className="text-brand-400 font-semibold text-sm mb-2">Suggestions</div>
          <ul className="space-y-1">
            {evaluation.suggestions.map((s, i) => (
              <li key={i} className="text-gray-300 text-sm flex gap-2">
                <span className="shrink-0 text-brand-500">→</span><span>{s}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Download */}
      <a
        href={`${API}/download/${sessionId}/model`}
        download="model.pkl"
        className="inline-flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors"
      >
        ⬇ Download model.pkl
      </a>
    </div>
  );
}
