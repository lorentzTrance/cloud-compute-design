#!/bin/bash
# ============================================
# CI/CD Pipeline: Build → Push SWR → Deploy to CCE
# ============================================
set -e

echo "=========================================="
echo "Stage 1/4: Checkout code"
echo "=========================================="
echo "Branch: main"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'local')"
echo "SUCCESS ✓"

echo ""
echo "=========================================="
echo "Stage 2/4: Build Docker images"
echo "=========================================="
docker build -f backend/Dockerfile.backend -t backend:latest backend/
docker build -f frontend/Dockerfile.frontend -t frontend:latest frontend/
echo "SUCCESS ✓"

echo ""
echo "=========================================="
echo "Stage 3/4: Push images to SWR"
echo "=========================================="
SWR="swr.cn-north-4.myhuaweicloud.com/cloud-56"
echo "Tagging..."
docker tag backend:latest ${SWR}/backend:latest
docker tag frontend:latest ${SWR}/frontend:latest
echo "Pushing..."
docker push ${SWR}/backend:latest
docker push ${SWR}/frontend:latest
echo "SUCCESS ✓"

echo ""
echo "=========================================="
echo "Stage 4/4: Deploy to CCE"
echo "=========================================="
echo "Updating backend deployment..."
kubectl set image deployment/backend backend=${SWR}/backend:latest -n default
kubectl rollout status deployment/backend -n default --timeout=120s
echo "Updating frontend deployment..."
kubectl set image deployment/frontend frontend=${SWR}/frontend:latest -n default
kubectl rollout status deployment/frontend -n default --timeout=120s
echo "SUCCESS ✓"

echo ""
echo "=========================================="
echo "All stages PASSED"
echo "Deployment completed!"
echo "=========================================="
kubectl get pods -o wide
