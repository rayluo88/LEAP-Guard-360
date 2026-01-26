# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LEAP-Guard 360** - A predictive maintenance demo for aviation engines, showcasing ML anomaly detection + GenAI diagnostics. This is a portfolio project targeting an MRO (Maintenance, Repair, Overhaul) engineering role at ST Engineering Aerospace.

**Goal:** Demonstrate full-stack ML/Cloud/GenAI capabilities suitable for the job described in `docs/JD.md`.

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│  React + TS     │────▶│  AWS Lambda (Docker)                │
│  (S3 Static)    │◀────│  - LSTM-Autoencoder inference       │
│  Vite + Recharts│     │  - Bedrock GenAI for diagnostics    │
└─────────────────┘     └─────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌─────────────────┐     ┌─────────────────────────────────────┐
│  test_data.json │     │  AWS Bedrock (Claude Haiku 4.5)     │
│  (S3 bucket)    │     │  Natural language anomaly diagnosis │
└─────────────────┘     └─────────────────────────────────────┘
```

**Data Flow:**
1. Frontend loads simulated engine sensor data from S3
2. User triggers analysis → Frontend calls Lambda
3. Lambda runs LSTM model → produces anomaly score
4. If anomaly detected → Lambda calls Bedrock for natural language diagnosis
5. Results returned to frontend for visualization

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | React + TypeScript + Vite | Type safety for aviation context |
| Visualization | Recharts | Time-series sensor graphs |
| Backend | Python + FastAPI/Lambda handler | ML inference |
| ML Model | PyTorch LSTM-Autoencoder | Anomaly detection |
| GenAI | AWS Bedrock (Claude Haiku 4.5) | Cost-effective diagnostics |
| Hosting | S3 (frontend) + Lambda (backend) | Serverless, near-zero idle cost |
| Container | Docker → ECR → Lambda | Portable ML environment |

## Development Phases

1. **Data & Model** - Train LSTM on NASA CMAPSS dataset in Colab, export `.pt`
2. **Backend & Cloud** - Dockerize inference, deploy to Lambda via ECR
3. **Frontend & Integration** - React dashboard connecting to Lambda

## Cost Model (Demo)

Architecture designed for <$0.10/month using AWS Free Tier + on-demand pricing. See `docs/AWS_Options.md` for enterprise vs demo comparison.

## Key Files

- `docs/PRD.md` - Full product requirements and functional specs
- `docs/AWS_Options.md` - Architecture cost comparison
- `docs/JD.md` - Target job description for alignment

## Implementation Progress Tracking

Progress is tracked in `docs/implementation-plan.md`:
- Add ✅ next to completed task/section titles
- Update verification checklists at the bottom of the file

## Design Decisions

- **Lambda over SageMaker:** Cost optimization for demo; shows FinOps awareness
- **Bedrock over OpenAI:** AWS-native, pay-per-token, cost-controlled
- **Static S3 frontend:** No server costs, CDN-ready via CloudFront
- **Docker containerization:** Production-grade deployment pattern
