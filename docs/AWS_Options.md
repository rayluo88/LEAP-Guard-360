# AWS Architecture Options

| Component | Enterprise Plan (Standard) | Demo Plan (Low-Cost / Free Tier) | Why this works for the Interview |
|-----------|---------------------------|----------------------------------|----------------------------------|
| **Model Hosting** | SageMaker Endpoint: Always-on server. Cost: ~$50-$100/month. | AWS Lambda (Containerized): "Serverless." Runs only when clicked. Cost: $0 (Free Tier covers 400k GB-seconds). | Shows you understand containerization (Docker) and serverless architecture. |
| **Model Training** | SageMaker Training Job: Renting GPU instances. Cost: $1-$5 per hour. | Google Colab / Local: Train locally, save model artifact .pt, upload to S3. Cost: $0. | Shows resourcefulness. You only need SageMaker deployment skills for the JD. |
| **Frontend Hosting** | EC2 or Amplify: Virtual machine hosting. Cost: $10-$20/month. | AWS S3 Static Hosting: Host the React build files on S3. Cost: <$0.10/month. | Standard industry practice for Single Page Apps (SPAs). |
| **Database** | AWS RDS / DynamoDB: Managed database. Cost: $15+/month. | S3 JSON Files: Store the "test data" in a simple JSON file on S3. Cost: <$0.01/month. | For a demo, you don't need a live DB. You just need to fetch data. |
| **GenAI** | Provisioned Throughput: Dedicated capacity. Cost: $$$$. | Bedrock On-Demand: Pay per token. Cost: ~$0.05 for hundreds of demo queries. | Only pays for what you use during the demo. |
