# Product Requirements Document (PRD): LEAP-Guard 360

**Version:** 1.1
**Status:** Draft (Refined)  
**Owner:** Raymond Luo
**Date:** 2026-01-20  

---

## 1. Executive Summary
**LEAP-Guard 360** is a conceptual Predictive Maintenance Dashboard designed to demonstrate full-stack Machine Learning and Generative AI capabilities for the aviation MRO sector. 

**Objective:** To serve as a technical portfolio piece for the "Machine Learning Engineer" application at ST Engineering Aerospace. The project demonstrates proficiency in **AWS Cloud Architecture, Generative AI (Bedrock), Predictive Modeling (Deep Learning), and Modern Frontend Development.**

---

## 2. Assumptions & Prerequisites

### Development Environment
| Requirement | Version/Details |
|-------------|-----------------|
| Node.js | 18+ (for frontend build) |
| Python | 3.12 (match Colab for pickle compatibility) |
| Docker Desktop | Latest (for containerized Lambda) |
| AWS CLI | v2, configured with credentials |

### AWS Account Setup
- [ ] AWS account with billing enabled
- [ ] Bedrock model access approved (Claude/Titan requires manual request in AWS Console)
- [ ] IAM user with programmatic access configured locally
- [ ] Default region set to `us-east-1` (best Bedrock model availability)

### Knowledge Assumptions
- Basic familiarity with React, Python, and AWS Console
- Understanding of Docker containerization
- Access to Google Colab for model training

---

## 3. Out of Scope

To maintain focus and prevent scope creep, the following are **explicitly excluded**:

- User authentication/login system
- Database persistence (all data is static/simulated)
- Real-time streaming (WebSocket) — simulation uses polling
- Mobile responsiveness (desktop-only)
- Multi-engine support (single engine demo)
- Historical trend analysis across multiple flights
- Internationalization (English only)
- Accessibility (WCAG) compliance

---

## 4. Problem Statement
Aircraft technicians and engineers face two critical challenges:
1.  **Data Overload:** Modern engines (like the CFM LEAP) generate terabytes of sensor data. Detecting subtle anomalies before they trigger a hard warning is difficult.
2.  **Interpretation Gap:** Even when an anomaly is detected, interpreting the raw sensor codes requires digging through thousands of pages of technical manuals.

## 5. Goals & Success Metrics
* **Goal 1:** Accurately detect anomalies in simulated engine sensor data using an LSTM-Autoencoder.
* **Goal 2:** Translate numerical anomalies into natural language diagnostic reports using GenAI.
* **Goal 3:** Deploy the solution on AWS using a cost-optimized, serverless architecture ("Scale-to-Zero").
* **Success Metric:** The system successfully processes a user request and returns a visualized anomaly + text explanation in under 5 seconds.

---

## 6. User Personas
### Primary Persona: The MRO Engineer (User)
* **Goal:** Quickly identify which engine component is degrading and understand *why*.
* **Pain Point:** "I see a vibration spike in Spool 2, but I don't know if it's a sensor error or a bearing failure."

### Secondary Persona: The Interviewer (Evaluator)
* **Goal:** Verify the candidate's skills in Python, AWS, GenAI, and Frontend.
* **Checklist:** * Is the architecture sound?
    * Is the code clean?
    * Does it actually work?

---

## 7. Functional Requirements

### 7.1 Real-Time Engine Health Monitoring
* **FR-01:** The system shall visualize "Live" sensor data (simulated stream) for an aircraft engine.
* **FR-02:** The system shall display "Actual" vs "Predicted" values on a time-series graph.
* **FR-03:** The system shall highlight regions where the reconstruction error (anomaly score) exceeds the threshold.

### 7.2 GenAI Diagnostics (The "Copilot")
* **FR-04:** Users can click on an anomaly event to trigger a "Diagnose" action.
* **FR-05:** The system shall send the specific sensor readings to AWS Bedrock.
* **FR-06:** The system shall return a natural language explanation citing potential root causes (e.g., "Possible seal degradation in High-Pressure Compressor").

### 7.3 UI Layout Specifications
* **Layout:** Desktop-only, minimum 1200px viewport width
* **Sidebar:** Engine selector dropdown (single engine for v1), anomaly threshold slider (0.5–1.0)
* **Main Graph:** Recharts `LineChart` with dual-axis (actual vs predicted), anomaly regions shaded in red (`ReferenceArea`)
* **Chat Window:** Click anomaly region → triggers diagnosis → displays response in chat bubble format
* **Color Scheme:** Dark theme with aviation-inspired blues and alert reds

---

## 8. Technical Architecture & Tech Stack

### 8.1 Cost-Optimized "Demo" Architecture
*Design Philosophy: Serverless & Static Hosting to minimize idle costs.*

1.  **Frontend (UI):**
    * **Tech:** React.js + TypeScript + Vite.
    * **Visualization:** Recharts (for sensor graphs).
    * **Hosting:** **AWS S3** (Static Website Hosting) + **CloudFront** (optional, for SSL).
    * **Cost Strategy:** Costs only occur on storage (<$0.10/mo).

2.  **Backend (Inference API):**
    * **Tech:** Python (FastAPI or pure Lambda handler).
    * **Compute:** **AWS Lambda** (Container Image support).
    * **Cost Strategy:** Free tier provides 400,000 GB-seconds per month. No cost when not running.

3.  **Machine Learning Model:**
    * **Tech:** PyTorch (LSTM-Autoencoder).
    * **Training:** Google Colab (Free GPU).
    * **Storage:** Model weights saved as `.pt` inside the Lambda Docker image.

4.  **Generative AI:**
    * **Service:** **AWS Bedrock**.
    * **Model:** Claude 3 Haiku (Fast & Cheap) or Titan Text.
    * **Cost Strategy:** On-Demand pricing. Only pays for tokens used during the demo.

### 8.2 AWS Infrastructure Configuration

| Component | Configuration |
|-----------|---------------|
| **Lambda** | Memory: 1024MB, Timeout: 30s, Runtime: Container (Python 3.12) |
| **ECR** | Private repository for inference Docker image |
| **Endpoint** | Lambda Function URL (simpler than API Gateway, no additional cost) |
| **IAM Role** | Lambda execution role with `bedrock:InvokeModel`, `logs:CreateLogGroup`, `logs:PutLogEvents` |
| **Bedrock** | Region: `us-east-1`, Model: `anthropic.claude-3-haiku-20240307-v1:0` |
| **S3** | Public bucket for frontend, private bucket for test data |

#### First-Time AWS Setup Checklist
- [ ] Create IAM user with `AdministratorAccess` (for demo simplicity)
- [ ] Run `aws configure` with access keys
- [ ] Request Bedrock model access in AWS Console → Bedrock → Model access
- [ ] Create ECR repository: `aws ecr create-repository --repository-name leap-guard-inference`
- [ ] Create S3 bucket for frontend: `leap-guard-frontend-{account-id}`

### 8.3 Data Flow Diagram
1.  **Frontend** loads `test_data.json` (simulated engine cycle) from S3.
2.  **Frontend** sends a slice of data to **API Gateway / Lambda URL**.
3.  **Lambda** runs the LSTM model -> returns `Anomaly Score`.
4.  If Score > Threshold, **Lambda** calls **AWS Bedrock** with a prompt: *"Sensor X is reading Y, expected Z. Explain implications."*
5.  **Lambda** returns JSON `{ prediction: [...], diagnosis: "..." }` to Frontend.

---

## 9. Data Schemas

### 9.1 Training Data: NASA CMAPSS
The model is trained on the **NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset, which simulates turbofan engine degradation. This dataset serves as a proxy for CFM LEAP sensor patterns.

| Property | Details |
|----------|---------|
| **Source** | NASA Prognostics Data Repository |
| **Engines** | 100 simulated turbofan units |
| **Sensors** | 21 sensor channels (temperature, pressure, speeds) |
| **Cycles** | Run-to-failure trajectories (varying 128–362 cycles) |
| **Relevance** | Simulates real degradation patterns in high-bypass turbofans |

### 9.2 Test Data Format (`test_data.json`)
```json
{
  "engine_id": "LEAP-1A-001",
  "metadata": {
    "aircraft_type": "A320neo",
    "total_cycles": 250
  },
  "cycles": [
    {
      "cycle": 1,
      "sensors": {
        "T24": 642.15,
        "T30": 1589.70,
        "T50": 1406.36,
        "P30": 554.36,
        "Nf": 2388.06,
        "Nc": 9046.19,
        "Ps30": 47.47,
        "phi": 521.66
      }
    }
  ]
}
```

### 9.3 API Contract

#### Request: `POST /predict`
```json
{
  "sensor_readings": [[642.15, 1589.70, 1406.36, ...], ...],
  "window_size": 50,
  "threshold": 0.7
}
```

#### Response: Success (200)
```json
{
  "anomaly_score": 0.85,
  "threshold": 0.7,
  "is_anomaly": true,
  "diagnosis": "Elevated T30 readings suggest possible HPC seal degradation. Recommend borescope inspection at next C-check.",
  "sensor_contributions": {
    "T30": 0.42,
    "P30": 0.28,
    "Nc": 0.18
  }
}
```

#### Response: Error (4xx/5xx)
```json
{
  "error": "Invalid input format",
  "detail": "sensor_readings must be a 2D array with shape (window_size, n_features)"
}
```

---

## 10. Development Roadmap

### Phase 1: Data & Model
* Download NASA CMAPSS dataset
* Train LSTM-Autoencoder in Colab
* Validate that the model correctly identifies failures in the test set
* Save model artifacts (`.pt` format)

**✓ Phase 1 Verification Checklist:**
- [ ] Model trains without errors in Colab
- [ ] Reconstruction error clearly separates healthy vs degraded cycles
- [ ] Model file exported and downloadable (`leap_guard_model.pth`, < 50MB)
- [ ] Test inference runs locally with sample data

### Phase 2: Backend & Cloud
* Set up AWS Account & Bedrock access
* Create Python script to load model and run inference
* Dockerize the Python script
* Push to AWS ECR and connect to AWS Lambda
* Test Lambda with a mock JSON payload

**✓ Phase 2 Verification Checklist:**
- [ ] Docker container builds and runs locally
- [ ] Lambda responds to test payload via SAM CLI
- [ ] Lambda deployed and returns 200 from Function URL
- [ ] Bedrock call works and returns diagnosis text
- [ ] CloudWatch logs visible and showing execution details

### Phase 3: Frontend & Integration
* Initialize React + TypeScript project
* Build the dashboard layout (Sidebar, Main Graph, Chat Window)
* Connect Frontend to Lambda Function URL
* Deploy frontend build to S3 bucket

**✓ Phase 3 Verification Checklist:**
- [ ] `npm run dev` starts local server without errors
- [ ] Graph renders with sample data
- [ ] "Diagnose" button triggers Lambda call and displays response
- [ ] Frontend deployed to S3 and accessible via public URL
- [ ] CORS configured correctly (no console errors)

---

## 11. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| **NFR-01** | Cold start latency | < 10s (acceptable for demo) |
| **NFR-02** | Inference + GenAI response time | < 5s (warm Lambda) |
| **NFR-03** | CORS | Enabled for S3 frontend origin |
| **NFR-04** | Authentication | None (demo simplicity) |
| **NFR-05** | Graceful degradation | System works if Bedrock unavailable |
| **NFR-06** | Browser support | Chrome/Edge latest (desktop only) |

---

## 12. Error Handling Strategy

| Failure Scenario | Handling | User Experience |
|-----------------|----------|-----------------|
| Model inference fails | Return 500 with error JSON | "Analysis unavailable. Please retry." |
| Bedrock timeout (>10s) | Return prediction without diagnosis | Anomaly shown, chat displays "GenAI unavailable" |
| Bedrock quota exceeded | Fallback to cached generic response | "Contact support for diagnosis" |
| Invalid input data | Return 400 with validation message | Form validation error displayed |
| Lambda cold start | No special handling | User sees loading spinner (up to 10s) |
| S3 data fetch fails | Frontend shows error state | "Unable to load sensor data" |

---

## 13. Local Development

### Running Lambda Locally (SAM CLI)
```bash
# Install SAM CLI
brew install aws-sam-cli

# Build and invoke locally
cd backend/
sam build
sam local invoke InferenceFunction -e events/test_event.json
```

### Mock Bedrock for Offline Development
Set environment variable to skip Bedrock calls:
```bash
export MOCK_BEDROCK=true
```
When enabled, Lambda returns a canned diagnosis response.

### Frontend Dev Server
```bash
cd frontend/
npm install
npm run dev  # Starts on http://localhost:5173
```

Configure `.env.local` for local backend:
```
VITE_API_URL=http://localhost:3000
```

---

## 14. Financial Estimation (Monthly)

| Service | Estimated Usage (Demo) | Estimated Cost |
| :--- | :--- | :--- |
| **AWS Lambda** | 500 requests | $0.00 (Free Tier) |
| **AWS S3** | 100MB Storage | $0.01 |
| **AWS Bedrock** | 100 queries (Claude Haiku) | < $0.05 |
| **Data Transfer** | < 1GB | $0.00 (Free Tier) |
| **Total** | | **~ $0.10 USD / Month** |

---

## 15. Interview Talking Points (Why this matters)
* **"Why Lambda over SageMaker Endpoints?"** → "For a production environment with constant traffic, I would use SageMaker. For this internal tool/demo, I chose Lambda to optimize costs, demonstrating my ability to architect for **FinOps** and **Operational Efficiency**."
* **"Why TypeScript?"** → "Aviation software requires high reliability. TypeScript adds static typing to prevent runtime errors, ensuring the dashboard is as robust as the backend."
* **"Why NASA CMAPSS data?"** → "CMAPSS is the gold standard for turbofan degradation research. Its 21 sensor channels closely mirror real engine telemetry, making it an appropriate proxy for CFM LEAP data."
* **"How would you scale this?"** → "Replace Lambda with SageMaker Endpoints for consistent latency, add API Gateway with throttling, implement proper auth via Cognito, and switch to streaming inference for real-time monitoring."