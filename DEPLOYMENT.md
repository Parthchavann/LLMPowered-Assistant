# 🚀 Enterprise Deployment Guide

Complete guide for deploying the LLM-Powered Customer Support Assistant in production environments.

## 🎯 Deployment Options

### 1. Development (Single Machine)
Perfect for development and testing.

```bash
# Quick start
./scripts/setup.sh
docker-compose up --build
```

### 2. Production (Multi-Container)
Scalable deployment with service separation.

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Kubernetes (Enterprise Scale)
For enterprise deployments with auto-scaling.

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

### 4. Cloud Deployment
Platform-specific deployment guides.

## 📋 Pre-Deployment Checklist

### System Requirements

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **Storage** | 50 GB SSD | 200 GB SSD | 1+ TB NVMe |
| **GPU** | Optional | RTX 3060 | RTX 4090/A100 |

### Dependencies

```bash
# Install Docker & Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2:3b
ollama pull llama3.2:7b  # For advanced features
```

## 🔒 Security Configuration

### 1. Authentication Setup

```bash
# Generate JWT secret
export JWT_SECRET_KEY=$(openssl rand -hex 32)

# Configure SSO (optional)
export SSO_PROVIDER=azure-ad  # or okta, google
export SSO_CLIENT_ID=your-client-id
export SSO_CLIENT_SECRET=your-client-secret
```

### 2. Network Security

```yaml
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Firewall Configuration

```bash
# UFW setup
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Block direct access to internal services
sudo ufw deny 6333/tcp   # Qdrant (internal only)
sudo ufw deny 8000/tcp   # API (behind proxy)
sudo ufw deny 8501/tcp   # Streamlit (behind proxy)
```

## 📊 Monitoring Setup

### 1. Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
```

### 2. Log Aggregation

```yaml
# ELK Stack
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false

logstash:
  image: docker.elastic.co/logstash/logstash:8.8.0
  volumes:
    - ./logstash/pipeline:/usr/share/logstash/pipeline

kibana:
  image: docker.elastic.co/kibana/kibana:8.8.0
  ports:
    - "5601:5601"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: |
          python -m pytest tests/
          
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker Images
        run: |
          docker build -t rag-backend -f Dockerfile.backend .
          docker build -t rag-frontend -f Dockerfile.frontend .
          
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Production
        run: |
          docker-compose -f docker-compose.prod.yml up -d
```

## 🏢 Enterprise Features Configuration

### 1. Multi-Tenant Setup

```python
# enterprise_config.py
TENANT_CONFIGS = {
    "default": {
        "max_users": 100,
        "max_documents": 10000,
        "max_queries_per_hour": 1000,
        "storage_quota_gb": 100,
        "features_enabled": ["basic_rag", "analytics"]
    },
    "enterprise": {
        "max_users": 1000,
        "max_documents": 100000,
        "max_queries_per_hour": 10000,
        "storage_quota_gb": 1000,
        "features_enabled": ["advanced_rag", "multimodal", "ab_testing", "sso"]
    }
}
```

### 2. Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup Qdrant data
docker exec qdrant /opt/qdrant/qdrant-backup $BACKUP_DIR/qdrant

# Backup tenant data
rsync -av /data/tenants/ $BACKUP_DIR/tenants/

# Backup configuration
cp -r /app/config $BACKUP_DIR/

# Upload to cloud storage (AWS S3 example)
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/$(date +%Y%m%d)/

# Cleanup old backups (keep 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} \;
```

## ⚡ Performance Optimization

### 1. Caching Strategy

```python
# Redis caching configuration
REDIS_CONFIG = {
    "host": "redis",
    "port": 6379,
    "db": 0,
    "max_connections": 100,
    "cache_ttl": 3600,  # 1 hour
    "embedding_cache_ttl": 86400  # 24 hours
}
```

### 2. Load Balancing

```nginx
# nginx load balancer
upstream backend {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=3;
    server backend3:8000 weight=2;
}

upstream frontend {
    ip_hash;  # Sticky sessions for Streamlit
    server frontend1:8501;
    server frontend2:8501;
}
```

### 3. Database Optimization

```python
# Qdrant optimization
QDRANT_CONFIG = {
    "collection_config": {
        "vectors": {
            "size": 384,  # MiniLM embedding size
            "distance": "Cosine"
        },
        "optimizers_config": {
            "default_segment_number": 2,
            "max_segment_size": 50000,
            "memmap_threshold": 20000
        },
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100,
            "full_scan_threshold": 10000
        }
    }
}
```

## 📈 Scaling Guidelines

### Horizontal Scaling

| Users | Backend Instances | Frontend Instances | Qdrant Nodes |
|-------|-------------------|-------------------|---------------|
| 1-100 | 1 | 1 | 1 |
| 100-1K | 2-3 | 1-2 | 1 |
| 1K-10K | 3-5 | 2-3 | 2-3 |
| 10K+ | 5+ | 3+ | 3+ (cluster) |

### Vertical Scaling

```yaml
# Resource allocation per service
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
          
  qdrant:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
```

## 🔍 Health Checks & Monitoring

### 1. Health Check Endpoints

```python
# Health check configuration
HEALTH_CHECKS = {
    "/health": "Basic health check",
    "/health/db": "Database connectivity",
    "/health/llm": "LLM service availability",
    "/health/detailed": "Comprehensive system status"
}
```

### 2. Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: rag-system
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

## 🔄 Disaster Recovery

### Recovery Procedures

1. **Service Recovery**
   ```bash
   # Restart services
   docker-compose restart
   
   # Check service health
   curl http://localhost:8000/health
   ```

2. **Data Recovery**
   ```bash
   # Restore from backup
   ./scripts/restore.sh backup_20240815_120000
   
   # Verify data integrity
   python scripts/verify_data.py
   ```

3. **Full System Recovery**
   ```bash
   # Complete system restore
   ./scripts/disaster_recovery.sh
   ```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Update models
ollama pull llama3.2:3b

# Clean up logs
find /var/log -name "*.log" -mtime +7 -delete

# Optimize database
docker exec qdrant /opt/qdrant/optimize

# Generate health report
python scripts/health_report.py
```

### Troubleshooting Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High memory usage | Large embeddings cache | Reduce cache size or add RAM |
| Slow responses | Network latency | Add caching, optimize queries |
| Model errors | Ollama service down | Restart Ollama service |
| Database errors | Qdrant connection | Check network, restart Qdrant |

## 📋 Production Checklist

- [ ] SSL certificates configured
- [ ] Firewall rules applied
- [ ] Monitoring dashboards set up
- [ ] Backup strategy implemented
- [ ] Load balancer configured
- [ ] Health checks enabled
- [ ] Alerting rules configured
- [ ] Security scanning completed
- [ ] Performance testing done
- [ ] Disaster recovery tested
- [ ] Documentation updated
- [ ] Team training completed

## 🔗 Additional Resources

- [Kubernetes Deployment Guide](k8s/README.md)
- [AWS Deployment Guide](cloud/aws/README.md)
- [Azure Deployment Guide](cloud/azure/README.md)
- [GCP Deployment Guide](cloud/gcp/README.md)
- [Performance Tuning Guide](docs/performance.md)
- [Security Best Practices](docs/security.md)