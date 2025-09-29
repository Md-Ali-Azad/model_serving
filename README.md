# GitHub Stars Predictor - ML Model Serving Platform

A scalable machine learning platform that predicts GitHub repository star counts using distributed infrastructure, containerization, and CI/CD pipelines. The system leverages multiple ML models to predict repository popularity based on various GitHub metrics and features.

## 🚀 Features

- **Machine Learning Pipeline**: Comprehensive model training and evaluation with 100+ algorithms
- **Containerized Deployment**: Docker-based microservices architecture with load balancing
- **Distributed Processing**: Celery workers with RabbitMQ for asynchronous task processing
- **CI/CD Pipeline**: Git hooks for automated model deployment and updates
- **Infrastructure as Code**: Ansible playbooks for automated server provisioning
- **Monitoring & Observability**: Prometheus metrics and Grafana dashboards
- **Load Balancing**: Nginx reverse proxy with multiple Flask instances

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │     Client      │    │   Production    │
│     Server      │    │     Machine     │    │     Server      │
│                 │    │                 │    │                 │
│ • Model Training│    │ • Ansible Host  │    │ • Flask Apps    │
│ • Data Prep     │    │ • OpenStack API │    │ • Celery Workers│
│ • Git Repository│    │ • SSH Keys      │    │ • RabbitMQ      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Monitoring    │
                    │                 │
                    │ • Prometheus    │
                    │ • Grafana       │
                    │ • Nginx LB      │
                    └─────────────────┘
```

### Container Architecture

- **Web Layer**: 3x Flask applications (ports 5101-5103) behind Nginx load balancer
- **Message Queue**: RabbitMQ for task distribution
- **Workers**: Scalable Celery workers for ML predictions
- **Monitoring**: Prometheus + Grafana stack
- **Reverse Proxy**: Nginx for load balancing and SSL termination

## 📊 Machine Learning Pipeline

### Dataset Features
The model uses GitHub repository metadata including:
- Repository metrics: forks, watchers, open issues, size
- Temporal features: project age, days since last update/push
- Repository attributes: language, license, wiki, projects
- Derived features: forks per day, issues per day, update rate
- Community metrics: subscribers, contributors, commits count

### Model Training
The system evaluates 100+ machine learning algorithms including:
- **Linear Models**: Ridge, Lasso, ElasticNet, Bayesian Ridge
- **Tree-Based**: Random Forest, Gradient Boosting, Extra Trees
- **Advanced Boosting**: XGBoost, LightGBM, CatBoost
- **Neural Networks**: Multi-layer Perceptrons with various configurations
- **Support Vector Machines**: Various kernels and parameters
- **Ensemble Methods**: Voting, Stacking, Bagging regressors

## 🛠️ Quick Start

### Prerequisites
- OpenStack environment (SSC Cloud)
- Ubuntu 20.04+ client machine
- Python 3.9+
- Docker & Docker Compose
- Ansible

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Set up OpenStack credentials
source your-project-openrc.sh

# Install OpenStack CLI tools
sudo apt install python3-openstackclient python3-novaclient python3-keystoneclient

# Install Ansible
sudo apt-add-repository ppa:ansible/ansible
sudo apt update && sudo apt install ansible
```

### 2. SSH Key Generation

```bash
# Generate SSH keys for cluster communication
mkdir -p /home/ubuntu/cluster-keys
ssh-keygen -t rsa -f /home/ubuntu/cluster-keys/cluster-key
# (Press Enter twice for no passphrase)
```

### 3. Deploy Infrastructure

```bash
# Navigate to deployment directory
cd openstack-client/single_node_with_docker_ansible_client

# Update cloud config files with your SSH public key
cat /home/ubuntu/cluster-keys/cluster-key.pub
# Copy the output to prod-cloud-cfg.txt and dev-cloud-cfg.txt

# Launch VMs
python3 start_instances.py

# Configure Ansible inventory with VM IPs
sudo nano /etc/ansible/hosts
```

Add to `/etc/ansible/hosts`:
```ini
[servers]
prodserver ansible_host=<PRODUCTION_SERVER_IP>
devserver ansible_host=<DEVELOPMENT_SERVER_IP>

[all:vars]
ansible_python_interpreter=/usr/bin/python3

[prodserver]
prodserver ansible_connection=ssh ansible_user=appuser

[devserver]
devserver ansible_connection=ssh ansible_user=appuser
```

### 4. Run Ansible Deployment

```bash
# Deploy services to both servers
export ANSIBLE_HOST_KEY_CHECKING=False
ansible-playbook configuration.yml --private-key=/home/ubuntu/cluster-keys/cluster-key
```

### 5. Access the Application

After deployment, attach floating IPs to your servers and access:

- **Main Application**: `http://<PRODUCTION_IP>`
- **Prediction Interface**: `http://<PRODUCTION_IP>/predict`
- **RabbitMQ Management**: `http://<PRODUCTION_IP>:15672`
- **Prometheus**: `http://<PRODUCTION_IP>:9090`
- **Grafana**: `http://<PRODUCTION_IP>:3000`

## 🔄 CI/CD Pipeline

### Git Hooks Setup

1. **Production Server Setup**:
```bash
# Create Git repository
mkdir /home/appuser/my_project
cd /home/appuser/my_project
git init --bare

# Create post-receive hook
nano hooks/post-receive
chmod +x hooks/post-receive
```

2. **Development Server Setup**:
```bash
# Initialize development repository
mkdir /home/appuser/my_project
cd /home/appuser/my_project
git init

# Add production as remote
git remote add production appuser@<PRODUCTION_IP>:/home/appuser/my_project
```

3. **Model Updates**:
```bash
# Train new model
cd /model_serving/ci_cd/development_server
python3 train.py

# Deploy to production
cp model* /home/appuser/my_project/
cd /home/appuser/my_project
git add .
git commit -m "Updated model"
git push production master
```

## 🐳 Container Management

### Build and Start Services
```bash
cd ci_cd/production_server

# Build and start all services
docker compose build
docker compose up -d

# Scale workers
docker compose up -d --scale worker_1=3 --scale worker_2=3

# Check status
docker compose ps
docker compose logs
```

### Service Scaling
```bash
# Scale up workers
docker compose up --scale worker_1=5 -d

# Scale down workers
docker compose up --scale worker_1=2 -d
```

## 📈 Monitoring

### Prometheus Metrics
- Application performance metrics
- Container resource usage
- Request rates and response times
- Error rates and availability

### Grafana Dashboards
- Real-time system monitoring
- ML model performance tracking
- Infrastructure health metrics
- Custom alerting rules

## 🔧 Configuration

### Environment Variables
```bash
# Flask Configuration
FLASK_ENV=production

# Celery Configuration
CELERY_BROKER_URL=amqp://rabbitmq:rabbitmq@rabbit:5672/
CELERY_RESULT_BACKEND=rpc://

# RabbitMQ Configuration
RABBITMQ_DEFAULT_USER=rabbitmq
RABBITMQ_DEFAULT_PASS=rabbitmq
```

### Model Configuration
Models are automatically selected based on R² score performance. The system:
1. Trains 100+ different algorithms
2. Evaluates using cross-validation
3. Selects the best performing model
4. Saves the model pipeline with preprocessing

## 🚀 API Usage

### Predict Single Repository
```bash
curl -X GET http://<PRODUCTION_IP>/predict
```

### Predict Multiple Repositories
```bash
curl -X POST http://<PRODUCTION_IP>/predict \
  -H "Content-Type: application/json" \
  -d '{"repositories": [...]}'
```

## 📝 Development

### Local Development Setup
```bash
# Install dependencies
pip install -r ci_cd/production_server/requirements.txt

# Run Flask app locally
cd ci_cd/production_server
python app.py
```

### Model Training
```bash
cd ci_cd/development_server
python train.py
```

This will:
- Load and preprocess the GitHub dataset
- Train multiple ML algorithms
- Evaluate model performance
- Save the best model as `best_model.pkl`
- Generate performance comparison charts

## 🔒 Security

- SSH key-based authentication between servers
- Container isolation with Docker
- Network segmentation with Docker networks
- Secure credential management
- Regular security updates via Ansible

## 📊 Performance

- **Horizontal Scaling**: Multiple Flask instances and Celery workers
- **Load Balancing**: Nginx distributes requests across instances
- **Asynchronous Processing**: Celery handles ML predictions asynchronously
- **Caching**: Model loaded once per worker for efficiency
- **Monitoring**: Real-time performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Docker Build Failures**:
```bash
# Clean Docker system
docker system prune -a --volumes
```

**Ansible Connection Issues**:
```bash
# Test SSH connectivity
ssh -i /home/ubuntu/cluster-keys/cluster-key appuser@<SERVER_IP>
```

**Model Loading Errors**:
- Ensure `best_model.pkl` exists in the production server
- Check file permissions and ownership
- Verify all dependencies are installed

**Service Discovery Issues**:
- Check Docker network configuration
- Verify service names in docker-compose.yml
- Ensure all containers are running

### Logs and Debugging
```bash
# View application logs
docker compose logs web1

# View worker logs
docker compose logs worker_1

# View RabbitMQ logs
docker compose logs rabbit

# System resource usage
docker stats
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review container logs
3. Verify network connectivity
4. Check Ansible playbook execution logs