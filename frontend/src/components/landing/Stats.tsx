export function Stats() {
  const stats = [
    { value: "<5s", label: "Inference Time", sublabel: "End-to-end response" },
    { value: "95%+", label: "Recall Rate", sublabel: "Anomaly detection" },
    { value: "100+", label: "Engine Units", sublabel: "Training dataset" },
  ];

  return (
    <section id="stats" className="stats">
      <div className="stats-grid">
        {stats.map((stat) => (
          <div key={stat.label} className="stat-item">
            <div className="stat-value">{stat.value}</div>
            <div className="stat-label">{stat.label}</div>
            <div className="stat-sublabel">{stat.sublabel}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
