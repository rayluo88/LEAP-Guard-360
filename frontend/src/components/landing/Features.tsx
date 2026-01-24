import { Activity, Brain, Database } from "lucide-react";

const features = [
  {
    icon: Activity,
    title: "LSTM Anomaly Detection",
    description:
      "Deep learning autoencoder trained on healthy engine data. Detects subtle degradation patterns humans miss.",
    color: "#007AFF",
  },
  {
    icon: Brain,
    title: "GenAI Diagnostics",
    description:
      "Claude Haiku 4.5 via AWS Bedrock interprets anomalies in plain English. Technical insights without the manual lookup.",
    color: "#32D583",
  },
  {
    icon: Database,
    title: "NASA CMAPSS Dataset",
    description:
      "Model trained on NASA's gold-standard turbofan degradation dataset with 100+ simulated engine run-to-failure cycles.",
    color: "#8B5CF6",
  },
];

export function Features() {
  return (
    <section id="features" className="features">
      <div className="section-header">
        <h2 className="section-title">Intelligent Engine Monitoring</h2>
        <p className="section-subtitle">
          Combining deep learning with generative AI for actionable maintenance
          insights
        </p>
      </div>

      <div className="features-grid">
        {features.map((feature) => (
          <div key={feature.title} className="feature-card">
            <div
              className="feature-icon"
              style={{ backgroundColor: `${feature.color}15` }}
            >
              <feature.icon size={28} color={feature.color} />
            </div>
            <h3 className="feature-title">{feature.title}</h3>
            <p className="feature-description">{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
