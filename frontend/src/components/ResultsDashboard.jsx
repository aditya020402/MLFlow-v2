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

function ValidationSection({ results }) {
  const metrics  = results.metrics  || {};
  const summary  = results.summary  || {};
  const isClassif = results.task_type?.includes("classif");

  const numericMetrics = Object.entries(metrics).filter(([, v]) => typeof v === "number");

  return (
    <div className="bg-purple-950/30 border border-purple-700 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <div className="text-purple-300 font-semibold text-base flex items-center gap-2">
            Held-out Validation Results
            <span className="text-xs font-normal text-purple-500 bg-purple-900/40 px-2 py-0.5 rounded-full">
              5% unseen data
            </span>
          </div>
          <p className="text-purple-400 text-sm mt-0.5">
            {summary.total ?? "?"} rows completely withheld from training
          </p>
        </div>
        {isClassif && summary.accuracy_pct !== undefined && (
          <div className="flex gap-4 text-sm">
            <span><span className="text-green-400 font-bold">{summary.correct}</span><span className="text-gray-400"> correct</span></span>
            <span><span className="text-red-400 font-bold">{summary.wrong}</span><span className="text-gray-400"> wrong</span></span>
            <span><span className="text-purple-300 font-bold">{summary.accuracy_pct}%</span><span className="text-gray-400"> accuracy</span></span>
          </div>
        )}
      </div>

      {numericMetrics.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {numericMetrics.map(([k, v]) => (
            <MetricCard key={k} label={k.toUpperCase()} value={v}
              highlight={["accuracy", "f1", "r2_score", "silhouette_score"].includes(k.toLowerCase())} />
          ))}
        </div>
      )}

      {metrics.classification_report && (
        <div className="bg-gray-900 rounded-lg p-3">
          <div className="text-gray-400 text-xs font-semibold mb-2 uppercase tracking-wide">Classification Report</div>
          <pre className="text-gray-300 text-xs font-mono overflow-x-auto whitespace-pre">
            {metrics.classification_report}
          </pre>
        </div>
      )}
    </div>
  );
}

export default function ResultsDashboard({ evaluation, algorithmInfo, sessionId, onTest, validationResults }) {
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

      {/* Validation hold-out results */}
      {validationResults && <ValidationSection results={validationResults} />}

      {/* Actions */}
      <div className="flex flex-wrap gap-3">
        <a
          href={`${API}/download/${sessionId}/model`}
          download="model.pkl"
          className="inline-flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors"
        >
          ⬇ Download model.pkl
        </a>
        {onTest && (
          <button
            onClick={onTest}
            className="inline-flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold px-6 py-3 rounded-lg transition-colors border border-gray-600"
          >
            🧪 Test on New Data
          </button>
        )}
      </div>
    </div>
  );
}
