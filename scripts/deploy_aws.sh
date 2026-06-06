#!/usr/bin/env bash
# =============================================================================
#  deploy_aws.sh  —  Build, push to ECR, and deploy to EC2
#
#  Prerequisites (run once on your local machine):
#    1. AWS CLI configured:   aws configure
#    2. EC2 key pair created: aws ec2 create-key-pair --key-name drivable-key
#    3. Set the four variables below to match your AWS account
#
#  Usage:
#    chmod +x scripts/deploy_aws.sh
#    ./scripts/deploy_aws.sh
#
#  What this script does:
#    Step 1  — Create ECR repository (idempotent — safe to re-run)
#    Step 2  — Build Docker image locally
#    Step 3  — Push image to ECR
#    Step 4  — Resolve the latest Amazon Linux 2 AMI for the target region
#              (fixes hardcoded us-east-1 AMI that fails in other regions)
#    Step 5  — Launch EC2 t2.micro with user-data bootstrap
#    Step 6  — Print the public URL
# =============================================================================

set -euo pipefail

# ── Configuration — edit these four values ────────────────────────────────────
AWS_REGION="eu-central-1"          # Frankfurt — closest to Bavaria :)
AWS_ACCOUNT_ID="YOUR_ACCOUNT_ID"   # 12-digit number from AWS console
ECR_REPO_NAME="drivable-area-detection"
EC2_KEY_NAME="drivable-key"        # Key pair name (without .pem)
# ─────────────────────────────────────────────────────────────────────────────

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_TAG="latest"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Drivable Area Detection — AWS Deployment       ║"
echo "╚══════════════════════════════════════════════════╝"
echo "  Region  : ${AWS_REGION}"
echo "  ECR URI : ${ECR_URI}"
echo ""

# ── Step 1: Create ECR repository (idempotent) ────────────────────────────────
echo "▶ Step 1/6  Creating ECR repository (if not exists) …"
aws ecr describe-repositories \
    --repository-names "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    > /dev/null 2>&1 \
|| aws ecr create-repository \
    --repository-name "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    > /dev/null
echo "  ✓ ECR repository ready"

# ── Step 2: Build Docker image ────────────────────────────────────────────────
echo ""
echo "▶ Step 2/6  Building Docker image …"
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .
echo "  ✓ Image built"

# ── Step 3: Push to ECR ───────────────────────────────────────────────────────
echo ""
echo "▶ Step 3/6  Pushing image to ECR …"
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"
echo "  ✓ Image pushed → ${ECR_URI}:${IMAGE_TAG}"

# ── Step 4: Resolve latest Amazon Linux 2 AMI for target region ──────────────
# AMI IDs are region-specific — a hardcoded us-east-1 AMI will fail in
# eu-central-1 or any other region. This query always gets the latest
# Amazon Linux 2 AMI available in the configured AWS_REGION.
echo ""
echo "▶ Step 4/6  Resolving latest Amazon Linux 2 AMI for ${AWS_REGION} …"
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters \
        "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text \
    --region "${AWS_REGION}")

if [[ -z "${AMI_ID}" || "${AMI_ID}" == "None" ]]; then
    echo "  ERROR: Could not resolve Amazon Linux 2 AMI in ${AWS_REGION}" >&2
    exit 1
fi
echo "  ✓ AMI resolved: ${AMI_ID}"

# ── Step 5: Launch EC2 t2.micro ───────────────────────────────────────────────
echo ""
echo "▶ Step 5/6  Launching EC2 instance …"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
yum update -y
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user

# Authenticate to ECR using the attached IAM role
aws ecr get-login-password --region ${AWS_REGION} \
    | docker login --username AWS --password-stdin \
      ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Pull and run the Streamlit app
docker pull ${ECR_URI}:${IMAGE_TAG}
docker run -d \
    --name drivable-app \
    --restart unless-stopped \
    -p 8501:8501 \
    ${ECR_URI}:${IMAGE_TAG}
USERDATA
)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type t2.micro \
    --key-name "${EC2_KEY_NAME}" \
    --user-data "${USER_DATA}" \
    --iam-instance-profile Name=EC2ECRReadRole \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=drivable-area-detection}]" \
    --security-group-ids $(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=drivable-sg" \
        --query "SecurityGroups[0].GroupId" \
        --output text 2>/dev/null || \
        aws ec2 create-security-group \
            --group-name drivable-sg \
            --description "Drivable Area Detection app" \
            --query "GroupId" --output text) \
    --query "Instances[0].InstanceId" \
    --output text \
    --region "${AWS_REGION}")

echo "  ✓ Instance launched: ${INSTANCE_ID}"

# Open port 8501 on the security group
aws ec2 authorize-security-group-ingress \
    --group-name drivable-sg \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0 \
    --region "${AWS_REGION}" \
    > /dev/null 2>&1 || true   # ignore if rule already exists

# ── Step 6: Wait and print URL ────────────────────────────────────────────────
echo ""
echo "▶ Step 6/6  Waiting for instance to get a public IP …"
sleep 15

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
    --region "${AWS_REGION}")

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Deployment complete ✓                          ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║   Instance ID : ${INSTANCE_ID}"
echo "║   AMI used    : ${AMI_ID}"
echo "║   Public IP   : ${PUBLIC_IP}"
echo "║"
echo "║   App URL (ready in ~2 min after boot):"
echo "║   http://${PUBLIC_IP}:8501"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "To SSH into the instance:"
echo "  ssh -i ~/.ssh/${EC2_KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "To stop and terminate the instance:"
echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
