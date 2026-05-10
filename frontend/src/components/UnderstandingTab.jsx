import { useState } from "react";

function Section({ title, children }) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  );
}

function Card({ children, className = "" }) {
  return (
    <div className={`bg-gray-900 border border-gray-700 rounded-xl p-4 ${className}`}>
      {children}
    </div>
  );
}

const TYPE_COLOR = {
  numeric:          "bg-blue-900/50 text-blue-300",
  categorical_low:  "bg-green-900/50 text-green-300",
  categorical_high: "bg-yellow-900/50 text-yellow-300",
  boolean:          "bg-purple-900/50 text-purple-300",
  datetime:         "bg-cyan-900/50 text-cyan-300",
  id_like:          "bg-gray-700 text-gray-400",
  constant:         "bg-red-900/50 text-red-300",
  free_text:        "bg-gray-700 text-gray-400",
};

function TypeBadge({ type }) {
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${TYPE_COLOR[type] || "bg-gray-700 text-gray-400"}`}>
      {type}
    </span>
  );
}

function DatasetOverview({ overview }) {
  if (!overview) return null;
  const items = [
    { label: "Rows",       value: overview.n_rows?.toLocaleString() },
    { label: "Columns",    value: overview.n_cols },
    { label: "Duplicates", value: overview.duplicate_rows },
    { label: "Missing",    value: overview.total_missing_pct != null ? `${(overview.total_missing_pct * 100).toFixed(1)}%` : "—" },
    { label: "Memory",     value: overview.memory_mb != null ? `${overview.memory_mb} MB` : "—" },
  ];
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
      {items.map(({ label, value }) => (
        <Card key={label} className="text-center">
          <div className="text-xl font-bold text-white">{value ?? "—"}</div>
          <div className="text-xs text-gray-400 mt-0.5">{label}</div>
        </Card>
      ))}
    </div>
  );
}

function LLMInsights({ insights }) {
  if (!insights || Object.keys(insights).length === 0) return null;
  return (
    <Card className="space-y-4">
      {insights.dataset_description && (
        <p className="text-sm text-gray-200 leading-relaxed">{insights.dataset_description}</p>
      )}
      <div className="grid md:grid-cols-2 gap-4 text-xs">
        {[
          { key: "column_relationships",      label: "Column Relationships" },
          { key: "feature_engineering_ideas", label: "Feature Engineering Ideas" },
          { key: "data_quality_concerns",     label: "Data Quality Concerns" },
          { key: "recommended_focus_areas",   label: "Recommended Focus Areas" },
        ].map(({ key, label }) =>
          insights[key]?.length > 0 ? (
            <div key={key}>
              <div className="text-gray-400 font-semibold mb-1.5">{label}</div>
              <ul className="space-y-1 text-gray-300">
                {insights[key].map((r, i) => <li key={i}>• {r}</li>)}
              </ul>
            </div>
          ) : null
        )}
      </div>
      {insights.target_column_notes && insights.target_column_notes !== "N/A" && (
        <div className="pt-2 border-t border-gray-700 text-xs">
          <span className="text-gray-400 font-semibold">Target notes: </span>
          <span className="text-gray-300">{insights.target_column_notes}</span>
        </div>
      )}
    </Card>
  );
}

function ColumnProfiles({ profiles }) {
  const [filter, setFilter] = useState("all");
  if (!profiles || Object.keys(profiles).length === 0) return null;

  const cols = Object.entries(profiles);
  const types = [...new Set(cols.map(([, p]) => p.inferred_type))].sort();
  const shown = filter === "all" ? cols : cols.filter(([, p]) => p.inferred_type === filter);

  return (
    <div className="space-y-3">
      <div className="flex gap-2 flex-wrap text-xs">
        <button
          onClick={() => setFilter("all")}
          className={`px-3 py-1 rounded-full border transition-colors ${filter === "all" ? "border-brand-500 bg-brand-900/50 text-brand-300" : "border-gray-700 text-gray-400 hover:text-gray-200"}`}
        >
          All ({cols.length})
        </button>
        {types.map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`px-3 py-1 rounded-full border transition-colors ${filter === t ? "border-brand-500 bg-brand-900/50 text-brand-300" : "border-gray-700 text-gray-400 hover:text-gray-200"}`}
          >
            {t} ({cols.filter(([, p]) => p.inferred_type === t).length})
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2 pr-4 font-medium">Column</th>
              <th className="text-left py-2 pr-4 font-medium">Type</th>
              <th className="text-right py-2 pr-4 font-medium">Missing</th>
              <th className="text-right py-2 pr-4 font-medium">Unique</th>
              <th className="text-left py-2 font-medium">Stats / Top values</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(([col, p]) => (
              <tr key={col} className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors">
                <td className="py-2 pr-4 font-mono text-gray-100 max-w-[140px] truncate" title={col}>{col}</td>
                <td className="py-2 pr-4"><TypeBadge type={p.inferred_type} /></td>
                <td className="py-2 pr-4 text-right text-gray-300">
                  {p.missing_pct != null
                    ? <span className={p.missing_pct > 0.1 ? "text-red-400" : ""}>{(p.missing_pct * 100).toFixed(1)}%</span>
                    : "—"}
                </td>
                <td className="py-2 pr-4 text-right text-gray-300">{p.nunique ?? "—"}</td>
                <td className="py-2 text-gray-400 max-w-xs truncate">
                  {p.stats ? (
                    <span>mean={p.stats.mean} · std={p.stats.std} · skew={p.stats.skewness}</span>
                  ) : p.top_values ? (
                    <span>{Object.entries(p.top_values).slice(0, 3).map(([v, c]) => `${v}: ${c}`).join("  ·  ")}</span>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CorrelationSection({ correlation }) {
  if (!correlation) return null;
  const pairs = correlation.high_correlation_pairs || [];
  const nzv   = correlation.near_zero_variance_cols || [];
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <Card>
        <div className="text-xs font-semibold text-gray-400 mb-3">High Correlation Pairs (|r| &gt; 0.9)</div>
        {pairs.length === 0 ? (
          <div className="text-xs text-gray-500 italic">None found</div>
        ) : (
          <ul className="space-y-2">
            {pairs.map((p, i) => (
              <li key={i} className="flex justify-between text-xs">
                <span className="font-mono text-gray-300">{(p.pair || "").replace("||", " ↔ ")}</span>
                <span className={`font-bold ${Math.abs(p.r) > 0.95 ? "text-red-400" : "text-yellow-400"}`}>{p.r}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
      <Card>
        <div className="text-xs font-semibold text-gray-400 mb-3">Near-Zero Variance Columns</div>
        {nzv.length === 0 ? (
          <div className="text-xs text-gray-500 italic">None found</div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {nzv.map((c, i) => (
              <span key={i} className="text-xs bg-red-900/50 text-red-300 px-2 py-0.5 rounded-full">{c}</span>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function ClassificationProfile({ data }) {
  return (
    <div className="space-y-4">
      {data.class_distribution && (
        <div>
          <div className="text-xs font-semibold text-gray-400 mb-2">Class Distribution</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.class_distribution).map(([cls, cnt]) => (
              <div key={cls} className="bg-gray-800 rounded-lg px-3 py-2 text-center min-w-[64px]">
                <div className="text-base font-bold text-white">{cnt?.toLocaleString()}</div>
                <div className="text-xs text-gray-400 mt-0.5 truncate max-w-[80px]" title={cls}>{String(cls)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {data.imbalance_ratio != null && (
        <div className="flex flex-wrap gap-6 text-xs pt-3 border-t border-gray-700">
          <div><span className="text-gray-400">Imbalance ratio: </span><span className="text-white font-semibold">{data.imbalance_ratio}</span></div>
          <div><span className="text-gray-400">Recommendation: </span><span className="text-brand-300 font-semibold font-mono">{data.imbalance_action}</span></div>
          {data.imbalance_reason && <div className="text-gray-500 flex-1">{data.imbalance_reason}</div>}
        </div>
      )}
      {data.top_discriminative_features?.length > 0 && (
        <div className="pt-3 border-t border-gray-700">
          <div className="text-xs font-semibold text-gray-400 mb-2">Top Discriminative Features</div>
          <div className="flex flex-wrap gap-1.5">
            {data.top_discriminative_features.map((f, i) => (
              <span key={i} className="text-xs font-mono bg-blue-900/40 text-blue-300 px-2 py-0.5 rounded">{f}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RegressionProfile({ data }) {
  const stats = data.target_stats || {};
  return (
    <div className="space-y-4">
      {Object.keys(stats).length > 0 && (
        <div>
          <div className="text-xs font-semibold text-gray-400 mb-2">Target Distribution</div>
          <div className="grid grid-cols-4 sm:grid-cols-7 gap-2 text-xs">
            {["mean", "std", "min", "max", "median", "skewness", "kurtosis"].map((k) => (
              <div key={k} className="bg-gray-800 rounded-lg p-2 text-center">
                <div className="font-bold text-white">{stats[k] ?? "—"}</div>
                <div className="text-gray-400 mt-0.5">{k}</div>
              </div>
            ))}
          </div>
          {data.log_transform_suggested && (
            <div className="mt-2 text-xs text-yellow-400 font-medium">
              ⚠ Log transform suggested (|skewness| &gt; 1, all values &gt; 0)
            </div>
          )}
        </div>
      )}
      {data.top_predictive_features?.length > 0 && (
        <div className="pt-3 border-t border-gray-700">
          <div className="text-xs font-semibold text-gray-400 mb-2">Top Predictive Features (by |r| with target)</div>
          <div className="flex flex-wrap gap-1.5">
            {data.top_predictive_features.map((f, i) => (
              <span key={i} className="text-xs font-mono bg-blue-900/40 text-blue-300 px-2 py-0.5 rounded">{f}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ClusteringProfile({ data }) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-6 text-xs">
        {data.scaling_sensitivity != null && (
          <div><span className="text-gray-400">Scaling sensitivity: </span><span className="text-white font-semibold">{data.scaling_sensitivity}</span></div>
        )}
        {data.scaling_required != null && (
          <div><span className="text-gray-400">Scaling required: </span><span className={`font-semibold ${data.scaling_required ? "text-yellow-400" : "text-green-400"}`}>{data.scaling_required ? "Yes" : "No"}</span></div>
        )}
        {data.pca_components_for_90pct_variance != null && (
          <div><span className="text-gray-400">PCA for 90% variance: </span><span className="text-white font-semibold">{data.pca_components_for_90pct_variance} components</span></div>
        )}
        {data.suggested_k_range && (
          <div><span className="text-gray-400">Suggested k range: </span><span className="text-white font-semibold">{data.suggested_k_range[0]}–{data.suggested_k_range[1]}</span></div>
        )}
      </div>
      {data.top_variance_features?.length > 0 && (
        <div className="pt-3 border-t border-gray-700">
          <div className="text-xs font-semibold text-gray-400 mb-2">Top Variance Features</div>
          <div className="flex flex-wrap gap-1.5">
            {data.top_variance_features.map((f, i) => (
              <span key={i} className="text-xs font-mono bg-blue-900/40 text-blue-300 px-2 py-0.5 rounded">{f}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SuggestedPlots({ plots }) {
  if (!plots?.length) return null;
  return (
    <div className="space-y-2">
      {plots.map((p, i) => (
        <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-mono font-semibold text-brand-300">{p.plot_type}</span>
            {p.features?.map((f, j) => (
              <span key={j} className="text-xs font-mono bg-gray-800 text-gray-300 px-2 py-0.5 rounded">{f}</span>
            ))}
          </div>
          {p.trigger_finding && (
            <div className="text-xs text-yellow-500/80">Trigger: {p.trigger_finding}</div>
          )}
          {p.reason && (
            <div className="text-xs text-gray-400">{p.reason}</div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function UnderstandingTab({ understandingData, profilingData, taskType, ready }) {
  if (!ready) {
    return (
      <div className="text-gray-500 text-sm italic text-center py-16">
        Data profile will appear here after the Data Understanding step completes.
      </div>
    );
  }

  const ud = understandingData || {};
  const pd = profilingData   || {};
  const isClassif = taskType?.includes("classif");
  const isRegress = taskType?.includes("regress");

  return (
    <div className="space-y-8">

      {/* Dataset Overview */}
      {ud.dataset_overview && (
        <Section title="Dataset Overview">
          <DatasetOverview overview={ud.dataset_overview} />
        </Section>
      )}

      {/* LLM Domain Insights */}
      {ud.llm_insights && Object.keys(ud.llm_insights).length > 0 && (
        <Section title="Domain Insights">
          <LLMInsights insights={ud.llm_insights} />
        </Section>
      )}

      {/* Column Profiles */}
      {ud.column_profiles && Object.keys(ud.column_profiles).length > 0 && (
        <Section title="Column Profiles">
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
            <ColumnProfiles profiles={ud.column_profiles} />
          </div>
        </Section>
      )}

      {/* Correlation Analysis */}
      {ud.correlation_analysis && (
        <Section title="Correlation Analysis">
          <CorrelationSection correlation={ud.correlation_analysis} />
        </Section>
      )}

      {/* Task-Specific Profiling */}
      {Object.keys(pd).length > 0 && (
        <Section title={`Task Profile · ${taskType?.replace(/_/g, " ") ?? ""}`}>
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
            {isClassif ? <ClassificationProfile data={pd} /> :
             isRegress  ? <RegressionProfile    data={pd} /> :
                          <ClusteringProfile    data={pd} />}
          </div>
        </Section>
      )}

      {/* Suggested Exploratory Plots (from understanding) */}
      {ud.suggested_exploratory_plots?.length > 0 && (
        <Section title={`Suggested Plots · General (${ud.suggested_exploratory_plots.length})`}>
          <SuggestedPlots plots={ud.suggested_exploratory_plots} />
        </Section>
      )}

      {/* Suggested Visualizations (from profiling) */}
      {pd.suggested_visualizations?.length > 0 && (
        <Section title={`Suggested Plots · Task-Specific (${pd.suggested_visualizations.length})`}>
          <SuggestedPlots plots={pd.suggested_visualizations} />
        </Section>
      )}

    </div>
  );
}
