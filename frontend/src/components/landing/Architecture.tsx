import { Monitor, Server, Cpu, MessageSquare } from "lucide-react";

export function Architecture() {
  return (
    <section id="architecture" className="architecture">
      <div className="section-header">
        <h2 className="section-title">How It Works</h2>
        <p className="section-subtitle">
          Serverless architecture designed for cost-efficiency and scalability
        </p>
      </div>

      <div className="architecture-diagram">
        <div className="arch-node">
          <div className="arch-icon">
            <Monitor size={32} />
          </div>
          <div className="arch-label">React Dashboard</div>
          <div className="arch-sublabel">S3 Static Hosting</div>
        </div>

        <div className="arch-arrow">
          <span>API Call</span>
        </div>

        <div className="arch-node">
          <div className="arch-icon">
            <Server size={32} />
          </div>
          <div className="arch-label">AWS Lambda</div>
          <div className="arch-sublabel">Container Runtime</div>
        </div>

        <div className="arch-arrow">
          <span>Inference</span>
        </div>

        <div className="arch-node">
          <div className="arch-icon">
            <Cpu size={32} />
          </div>
          <div className="arch-label">LSTM Model</div>
          <div className="arch-sublabel">PyTorch Autoencoder</div>
        </div>

        <div className="arch-arrow">
          <span>Diagnosis</span>
        </div>

        <div className="arch-node">
          <div className="arch-icon">
            <MessageSquare size={32} />
          </div>
          <div className="arch-label">AWS Bedrock</div>
          <div className="arch-sublabel">Claude Haiku 4.5</div>
        </div>
      </div>

      <div className="architecture-details">
        <div className="arch-detail">
          <h4>Scale-to-Zero</h4>
          <p>Pay only when inference runs. No idle costs.</p>
        </div>
        <div className="arch-detail">
          <h4>Cold Start &lt;10s</h4>
          <p>Containerized Lambda with pre-loaded model weights.</p>
        </div>
        <div className="arch-detail">
          <h4>On-Demand GenAI</h4>
          <p>Bedrock charges per token. No reserved capacity.</p>
        </div>
      </div>
    </section>
  );
}
