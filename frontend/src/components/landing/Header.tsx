import { Link } from "react-router-dom";

export function Header() {
  return (
    <header className="landing-header">
      <div className="header-content">
        <div className="logo">
          <img src="/logo-leap.png" alt="LEAP-Guard 360" className="logo-img" />
          <span className="logo-text">LEAP-Guard 360</span>
        </div>

        <nav className="header-nav">
          <a href="#features">Features</a>
          <a href="#architecture">Architecture</a>
          <a href="#stats">Performance</a>
        </nav>

        <Link to="/dashboard" className="btn btn-primary">
          Launch Dashboard
        </Link>
      </div>
    </header>
  );
}
