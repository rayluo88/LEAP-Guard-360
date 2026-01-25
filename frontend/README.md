# LEAP-Guard 360 Frontend

React + TypeScript dashboard for predictive maintenance visualization.

**Live Demo:** [lg360.vercel.app](https://lg360.vercel.app)

## Tech Stack

- React 19 + TypeScript
- Vite 7
- Recharts (time-series visualization)
- Axios (API client)

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## Environment Variables

Create `.env.local` for local development:

```bash
VITE_API_URL=https://5r7w3jhzhhw4e43r7y36mru77q0zpamo.lambda-url.ap-southeast-1.on.aws/
```

For Vercel deployment, set `VITE_API_URL` in the Vercel dashboard.

## Project Structure

```
src/
├── components/
│   ├── dashboard/    # Dashboard components (Sidebar, SensorChart, ChatWindow)
│   └── landing/      # Landing page components (Hero, Features)
├── data/             # Test data generation
├── hooks/            # Custom hooks (useInference)
├── pages/            # Page components
├── styles/           # CSS files
└── types/            # TypeScript type definitions
```

## Deployment

Deployed to Vercel with automatic builds on push:

```bash
vercel --prod
```
