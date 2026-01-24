import { Github, Linkedin, FileText } from "lucide-react";

export function Footer() {
  return (
    <footer className="landing-footer">
      <div className="footer-content">
        <div className="footer-brand">
          <div className="logo">
            <img
              src="/logo-leap.png"
              alt="LEAP-Guard 360"
              className="logo-img"
            />
            <span className="logo-text">LEAP-Guard 360</span>
          </div>
          <p className="footer-tagline">
            Predictive maintenance demo showcasing ML + GenAI capabilities
          </p>
        </div>

        <div className="footer-links">
          <div className="footer-section">
            <h4>Resources</h4>
            <a href="#features">Features</a>
            <a href="#architecture">Architecture</a>
            <a href="#stats">Performance</a>
          </div>

          <div className="footer-section">
            <h4>Technical</h4>
            <a
              href="https://github.com/rayluo88/LEAP-Guard-360"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Github size={16} /> GitHub
            </a>
            <a href="#" target="_blank" rel="noopener noreferrer">
              <FileText size={16} /> Documentation
            </a>
          </div>

          <div className="footer-section">
            <h4>Connect</h4>
            <a
              href="https://www.linkedin.com/in/raymondluoming/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Linkedin size={16} /> LinkedIn
            </a>
          </div>
        </div>
      </div>

      <div className="footer-bottom">
        <p>&copy; 2026 LEAP-Guard 360. Portfolio project by Raymond Luo.</p>
      </div>
    </footer>
  );
}
