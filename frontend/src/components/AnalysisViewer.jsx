export default function AnalysisViewer({ data }) {
  if (!data || Object.keys(data).length === 0) return null;

  const {
    feature_insights = [],
    top_features = [],
    recommendations = [],
    outlier_counts = {},
    preprocessing_decisions = {},
    correlations = {},
  } = data;

  // Top 5 correlations by absolute value
  const topCorrelations = Object.entries(correlations)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 5);

  const outlierEntries = Object.entries(outlier_counts).filter(([, v]) => v > 0);

  return (
    <div className="space-y-4 text-sm">
      {/* Feature insights */}
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

      {/* Top features + recommendations side by side */}
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

      {/* Preprocessing decisions */}
      {Object.keys(preprocessing_decisions).length > 0 && (
        <div>
          <div className="text-orange-300 font-semibold mb-1.5">Preprocessing Decisions</div>
          <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
            {Object.entries(preprocessing_decisions).slice(0, 8).map(([col, decision]) => (
              <div key={col} className="flex gap-3 px-3 py-1.5">
                <span className="text-gray-400 font-mono text-xs shrink-0 w-32 truncate">{col}</span>
                <span className="text-gray-300 text-xs">{decision}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Outliers + correlations */}
      <div className="grid grid-cols-2 gap-4">
        {outlierEntries.length > 0 && (
          <div>
            <div className="text-orange-300 font-semibold mb-1.5">Outliers Detected</div>
            <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
              {outlierEntries.slice(0, 6).map(([col, count]) => (
                <div key={col} className="flex justify-between px-3 py-1.5">
                  <span className="text-gray-400 font-mono text-xs truncate">{col}</span>
                  <span className="text-yellow-400 text-xs font-medium">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {topCorrelations.length > 0 && (
          <div>
            <div className="text-orange-300 font-semibold mb-1.5">Top Correlations</div>
            <div className="bg-gray-900 rounded-lg divide-y divide-gray-700 overflow-hidden">
              {topCorrelations.map(([pair, val]) => (
                <div key={pair} className="flex justify-between px-3 py-1.5">
                  <span className="text-gray-400 font-mono text-xs truncate">{pair}</span>
                  <span className={`text-xs font-medium ${Math.abs(val) > 0.7 ? "text-red-400" : Math.abs(val) > 0.4 ? "text-yellow-400" : "text-gray-400"}`}>
                    {Number(val).toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
