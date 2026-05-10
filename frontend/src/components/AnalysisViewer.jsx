export default function AnalysisViewer({ data }) {
  if (!data || Object.keys(data).length === 0) return null;

  const {
    original_shape,
    cleaned_shape,
    feature_insights = [],
    top_features = [],
    recommendations = [],
    preprocessing_decisions = {},
    outlier_summary = {},
    high_correlation_pairs = [],
    class_distribution = {},
    imbalance_ratio,
    imbalance_recommendation,
    feature_target_correlations = {},
    plot_insights = [],
    ml_model_recommendations = {},
    encoding_recommendations = {},
  } = data;

  const outlierEntries = Object.entries(outlier_summary)
    .filter(([, v]) => v && (v.pct_outliers ?? 0) > 0)
    .sort((a, b) => (b[1].pct_outliers ?? 0) - (a[1].pct_outliers ?? 0));

  const topCorrelations = Array.isArray(high_correlation_pairs)
    ? high_correlation_pairs.slice(0, 6)
    : [];

  const topFeatCorr = Object.entries(feature_target_correlations)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8);

  const isClassif = Object.keys(class_distribution).length > 0;

  return (
    <div className="space-y-5 text-sm">

      {/* ── Dataset shape ─────────────────────────────────── */}
      {(original_shape || cleaned_shape) && (
        <div className="flex gap-6 flex-wrap">
          {original_shape && (
            <div>
              <span className="text-gray-500 text-xs">Original</span>
              <div className="text-white font-mono font-semibold">
                {original_shape[0].toLocaleString()} rows × {original_shape[1]} cols
              </div>
            </div>
          )}
          {cleaned_shape && (
            <div>
              <span className="text-gray-500 text-xs">After cleaning</span>
              <div className="text-green-400 font-mono font-semibold">
                {cleaned_shape[0].toLocaleString()} rows × {cleaned_shape[1]} cols
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Plot-derived insights (new) ───────────────────── */}
      {plot_insights.length > 0 && (
        <div>
          <div className="text-blue-300 font-semibold mb-1.5">Plot Insights</div>
          <ul className="space-y-1">
            {plot_insights.map((insight, i) => (
              <li key={i} className="flex gap-2 text-gray-300">
                <span className="text-blue-400 shrink-0">◆</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Feature insights ──────────────────────────────── */}
      {feature_insights.length > 0 && (
        <div>
          <div className="text-orange-300 font-semibold mb-1.5">Feature Insights</div>
          <ul className="space-y-1">
            {feature_insights.map((insight, i) => (
              <li key={i} className="flex gap-2 text-gray-300">
                <span className="text-orange-500 shrink-0">•</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── ML model recommendations (new) ───────────────── */}
      {ml_model_recommendations && Object.keys(ml_model_recommendations).length > 0 && (
        <div className="bg-indigo-950/30 border border-indigo-800/50 rounded-lg p-4 space-y-3">
          <div className="text-indigo-300 font-semibold">ML Recommendations</div>
          {ml_model_recommendations.algorithm_hints?.length > 0 && (
            <div>
              <div className="text-indigo-400 text-xs font-medium mb-1 uppercase tracking-wide">Algorithm Hints</div>
              <ul className="space-y-1">
                {ml_model_recommendations.algorithm_hints.map((h, i) => (
                  <li key={i} className="text-gray-300 flex gap-2">
                    <span className="text-indigo-400 shrink-0">→</span><span>{h}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {ml_model_recommendations.expected_challenges?.length > 0 && (
            <div>
              <div className="text-yellow-400 text-xs font-medium mb-1 uppercase tracking-wide">Expected Challenges</div>
              <ul className="space-y-1">
                {ml_model_recommendations.expected_challenges.map((c, i) => (
                  <li key={i} className="text-gray-300 flex gap-2">
                    <span className="text-yellow-500 shrink-0">⚠</span><span>{c}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {ml_model_recommendations.preprocessing_notes?.length > 0 && (
            <div>
              <div className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wide">Remaining Preprocessing</div>
              <ul className="space-y-1">
                {ml_model_recommendations.preprocessing_notes.map((n, i) => (
                  <li key={i} className="text-gray-400 flex gap-2">
                    <span className="shrink-0">–</span><span>{n}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {ml_model_recommendations.feature_importance_candidates?.length > 0 && (
            <div>
              <div className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wide">Top Feature Candidates</div>
              <div className="flex flex-wrap gap-1.5">
                {ml_model_recommendations.feature_importance_candidates.slice(0, 12).map((f) => (
                  <span key={f} className="bg-indigo-950/60 border border-indigo-800 text-indigo-200 px-2 py-0.5 rounded text-xs">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Top features + recommendations ───────────────── */}
      <div className="grid grid-cols-2 gap-4">
        {top_features.length > 0 && (
          <div>
            <div className="text-orange-300 font-semibold mb-1.5">Top Features</div>
            <div className="flex flex-wrap gap-1.5">
              {top_features.map((f) => (
                <span key={f} className="bg-orange-950/60 border border-orange-800 text-orange-200 px-2 py-0.5 rounded text-xs">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}

        {recommendations.length > 0 && (
          <div>
            <div className="text-orange-300 font-semibold mb-1.5">Recommendations</div>
            <ul className="space-y-1">
              {recommendations.map((r, i) => (
                <li key={i} className="text-gray-300 flex gap-1.5">
                  <span className="text-green-500 shrink-0">→</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* ── Class distribution (classification) ──────────── */}
      {isClassif && (
        <div>
          <div className="text-orange-300 font-semibold mb-1.5 flex items-center gap-2">
            Class Distribution
            {imbalance_ratio && imbalance_ratio > 1 && (
              <span className={`text-xs px-2 py-0.5 rounded-full border font-normal ${
                imbalance_ratio > 10
                  ? "bg-red-950/50 border-red-700 text-red-400"
                  : imbalance_ratio > 3
                  ? "bg-yellow-950/50 border-yellow-700 text-yellow-400"
                  : "bg-green-950/50 border-green-700 text-green-400"
              }`}>
                {imbalance_ratio.toFixed(1)}× imbalance
              </span>
            )}
          </div>
          <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
            {Object.entries(class_distribution).map(([cls, count]) => (
              <div key={cls} className="flex justify-between px-3 py-1.5">
                <span className="text-gray-300 font-mono text-xs">{cls}</span>
                <span className="text-gray-400 text-xs">{Number(count).toLocaleString()} rows</span>
              </div>
            ))}
          </div>
          {imbalance_recommendation && imbalance_recommendation !== "n/a" && (
            <p className="text-yellow-400 text-xs mt-1.5">
              Recommendation: {imbalance_recommendation}
            </p>
          )}
        </div>
      )}

      {/* ── Feature-target correlations (regression) ─────── */}
      {topFeatCorr.length > 0 && (
        <div>
          <div className="text-orange-300 font-semibold mb-1.5">Feature–Target Correlations</div>
          <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
            {topFeatCorr.map(([col, r]) => (
              <div key={col} className="flex justify-between px-3 py-1.5">
                <span className="text-gray-400 font-mono text-xs truncate">{col}</span>
                <span className={`text-xs font-medium ${
                  Math.abs(r) > 0.7 ? "text-green-400" :
                  Math.abs(r) > 0.4 ? "text-yellow-400" : "text-gray-400"
                }`}>
                  {Number(r).toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Preprocessing decisions ───────────────────────── */}
      {Object.keys(preprocessing_decisions).length > 0 && (
        <div>
          <div className="text-orange-300 font-semibold mb-1.5">Preprocessing Decisions</div>
          <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
            {Object.entries(preprocessing_decisions).slice(0, 10).map(([col, decision]) => (
              <div key={col} className="flex gap-3 px-3 py-1.5">
                <span className="text-gray-400 font-mono text-xs shrink-0 w-32 truncate">{col}</span>
                <span className="text-gray-300 text-xs">{typeof decision === "string" ? decision : JSON.stringify(decision)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Outliers + high-correlation pairs ────────────── */}
      <div className="grid grid-cols-2 gap-4">
        {outlierEntries.length > 0 && (
          <div>
            <div className="text-orange-300 font-semibold mb-1.5">Outlier Summary</div>
            <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
              {outlierEntries.slice(0, 6).map(([col, info]) => (
                <div key={col} className="flex justify-between px-3 py-1.5">
                  <span className="text-gray-400 font-mono text-xs truncate">{col}</span>
                  <span className={`text-xs font-medium ${
                    info.recommended_action === "cap" ? "text-yellow-400" : "text-gray-500"
                  }`}>
                    {((info.pct_outliers ?? 0) * 100).toFixed(1)}% — {info.recommended_action}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {topCorrelations.length > 0 && (
          <div>
            <div className="text-orange-300 font-semibold mb-1.5">High Correlations</div>
            <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
              {topCorrelations.map((entry, i) => {
                const pair = typeof entry === "object" ? entry.pair : String(entry);
                const r    = typeof entry === "object" ? entry.r   : null;
                return (
                  <div key={i} className="flex justify-between px-3 py-1.5">
                    <span className="text-gray-400 font-mono text-xs truncate">{pair?.replace("||", " ↔ ")}</span>
                    {r != null && (
                      <span className="text-red-400 text-xs font-medium">{Number(r).toFixed(3)}</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
