# AWS Deployment Guide

**Last Updated:** 2026-01-25
**Region:** ap-southeast-1 (Singapore)
**Account ID:** 377133361984

---

## Prerequisites

- AWS CLI v2 configured (`aws configure`)
- SAM CLI installed (`brew install aws-sam-cli`)
- Docker Desktop running
- Bedrock model access approved (Claude Haiku 4.5)

---

## Resource Summary

| Resource | Name/ID | Purpose |
|----------|---------|---------|
| ECR Repository | `leap-guard-inference` | Stores Docker container image |
| Lambda Function | `leap-guard-inference-InferenceFunction` | Runs ML inference |
| Lambda Function URL | `https://5r7w3jhzhhw4e43r7y36mru77q0zpamo.lambda-url.ap-southeast-1.on.aws/` | Public HTTPS endpoint |
| IAM Role | (auto-created by SAM) | Grants Lambda access to Bedrock |

### Lambda Configuration

| Setting | Value |
|---------|-------|
| Runtime | Python 3.11 (Docker) |
| Memory | 2048 MB |
| Timeout | 60 seconds |
| Architecture | x86_64 |
| Cold start | ~57 seconds |
| Warm invocation | ~3-4 seconds |

---

## Deployment Commands

### Step 1: Authenticate Docker to ECR

```bash
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin \
  377133361984.dkr.ecr.ap-southeast-1.amazonaws.com
```

### Step 2: Build Docker Image

```bash
cd backend/
docker build --platform linux/amd64 --provenance=false -t leap-guard-inference .
```

> **Important:** The `--platform linux/amd64` ensures compatibility with Lambda. The `--provenance=false` disables OCI attestations that Lambda doesn't support.

### Step 3: Tag Image for ECR

```bash
docker tag leap-guard-inference:latest \
  377133361984.dkr.ecr.ap-southeast-1.amazonaws.com/leap-guard-inference:latest
```

### Step 4: Push to ECR

```bash
docker push 377133361984.dkr.ecr.ap-southeast-1.amazonaws.com/leap-guard-inference:latest
```

### Step 5: Deploy Lambda with SAM

```bash
# First time (interactive prompts)
sam build
sam deploy --guided

# Subsequent deploys
sam build && sam deploy
```

**SAM guided prompts (recommended answers):**
| Prompt | Answer |
|--------|--------|
| Stack Name | `leap-guard-inference` |
| AWS Region | `ap-southeast-1` |
| Confirm changes before deploy | `Y` |
| Allow SAM CLI IAM role creation | `Y` |
| Save arguments to config file | `Y` |

### Step 6: Get Function URL

```bash
aws cloudformation describe-stacks \
  --stack-name leap-guard-inference \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" \
  --output text --region ap-southeast-1
```

---

## Update Frontend

After deployment, update the frontend environment:

```bash
# frontend/.env.local (development)
VITE_API_URL=https://5r7w3jhzhhw4e43r7y36mru77q0zpamo.lambda-url.ap-southeast-1.on.aws/
```

For Vercel deployment, add the environment variable in Vercel dashboard:
- **Key:** `VITE_API_URL`
- **Value:** `https://5r7w3jhzhhw4e43r7y36mru77q0zpamo.lambda-url.ap-southeast-1.on.aws/`

---

## Testing

### Test with curl

The model expects **8 features** per sensor reading and `window_size >= 10`.

```bash
curl -X POST https://5r7w3jhzhhw4e43r7y36mru77q0zpamo.lambda-url.ap-southeast-1.on.aws/ \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_readings": [
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ],
    "window_size": 10,
    "threshold": 0.7
  }'
```

**Expected response:**
```json
{
  "anomaly_score": 6973830.76,
  "threshold": 0.7,
  "is_anomaly": true,
  "diagnosis": null,
  "sensor_contributions": {"N1_Fan_RPM": 0.998, "T25_LPC_Temp": 0.001, "N2_Core_RPM": 0.001}
}
```

> **Note:** First invocation (cold start) takes ~57 seconds. Subsequent requests are ~3-4 seconds.

### View Logs

```bash
aws logs tail /aws/lambda/leap-guard-inference-InferenceFunction \
  --follow --region ap-southeast-1
```

---

## Cleanup / Teardown

To stop all costs and remove resources:

### 1. Delete Lambda Stack

Removes Lambda function, IAM roles, and Function URL.

```bash
sam delete --stack-name leap-guard-inference --region ap-southeast-1
```

### 2. Delete ECR Images

Removes stored container images (stops storage costs).

```bash
aws ecr batch-delete-image \
  --repository-name leap-guard-inference \
  --region ap-southeast-1 \
  --image-ids imageTag=latest
```

### 3. Delete ECR Repository (Optional)

Completely removes the container registry.

```bash
aws ecr delete-repository \
  --repository-name leap-guard-inference \
  --region ap-southeast-1 \
  --force
```

---

## Cost Estimate

| Service | Usage (Demo) | Monthly Cost |
|---------|--------------|--------------|
| Lambda | 500 requests | $0.00 (Free Tier) |
| ECR | ~500MB storage | ~$0.05 |
| Bedrock (Haiku) | 100 queries | ~$0.05 |
| CloudWatch Logs | Minimal | ~$0.00 |
| **Total** | | **~$0.10/month** |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `403 Forbidden` from Bedrock | Check IAM policy includes `bedrock:InvokeModel`; verify model access is approved in Bedrock console |
| Cold start timeout | Increase Lambda timeout to 60s in `template.yaml` |
| CORS errors in browser | Verify `AllowOrigins` in `template.yaml` includes frontend URL |
| Out of memory | Increase `MemorySize` to 2048MB in `template.yaml` |
| Docker push fails | Re-run ECR login command (token expires after 12 hours) |
| "Image manifest not supported" | Rebuild with `--provenance=false` flag (disables OCI attestations) |
| "Expected shape (10, 8)" | Model expects 8 features per sensor reading, not 14 |
| "window_size >= 10" error | Input must have at least 10 rows of sensor readings |

---

## Useful Commands

```bash
# Check Lambda function status
aws lambda get-function --function-name leap-guard-inference-InferenceFunction \
  --region ap-southeast-1

# List ECR images
aws ecr list-images --repository-name leap-guard-inference --region ap-southeast-1

# View CloudFormation stack events (for debugging deploy issues)
aws cloudformation describe-stack-events \
  --stack-name leap-guard-inference --region ap-southeast-1

# Invoke Lambda directly (bypass Function URL)
aws lambda invoke \
  --function-name leap-guard-inference-InferenceFunction \
  --payload '{"body": "{\"sensor_readings\": [[0.5]], \"window_size\": 1}"}' \
  --region ap-southeast-1 \
  output.json && cat output.json
```
