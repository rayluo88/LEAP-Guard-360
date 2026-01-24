import {
  Header,
  Hero,
  Features,
  Architecture,
  Stats,
  Footer,
} from "../components/landing";

export function LandingPage() {
  return (
    <div className="landing-page">
      <Header />
      <Hero />
      <Features />
      <Architecture />
      <Stats />
      <Footer />
    </div>
  );
}
