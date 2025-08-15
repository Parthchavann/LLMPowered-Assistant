"""
ML Model Lifecycle Management

Comprehensive MLOps system including:
- Model versioning and registry
- Automated deployment pipelines
- Performance drift detection
- A/B testing integration for models
- Rollback capabilities
- Model monitoring and alerts
- Resource optimization
"""

import json
import logging
import pickle
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"
    FAILED = "failed"

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    user_satisfaction: float
    custom_metrics: Dict[str, float]

@dataclass
class ModelVersion:
    """Model version information"""
    id: str
    name: str
    version: str
    status: ModelStatus
    model_type: str  # 'llm', 'embedding', 'classifier'
    framework: str
    created_at: datetime
    created_by: str
    description: str
    tags: List[str]
    metrics: Optional[ModelMetrics]
    artifacts_path: str
    config: Dict[str, Any]
    parent_version: Optional[str]
    experiments: List[str]

@dataclass
class Deployment:
    """Deployment information"""
    id: str
    model_version_id: str
    environment: str  # 'staging', 'production'
    strategy: DeploymentStrategy
    traffic_percentage: float
    deployed_at: datetime
    deployed_by: str
    status: str
    health_check_url: str
    rollback_version: Optional[str]

class PerformanceDriftDetector:
    """Detect performance drift in deployed models"""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_metrics = {}
        self.recent_metrics = []
        
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline performance metrics"""
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline metrics set: {metrics}")
    
    def add_measurement(self, metrics: Dict[str, float]):
        """Add new performance measurement"""
        self.recent_metrics.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Keep only recent measurements
        if len(self.recent_metrics) > self.window_size:
            self.recent_metrics = self.recent_metrics[-self.window_size:]
    
    def detect_drift(self) -> Dict[str, Any]:
        """Detect performance drift"""
        if not self.baseline_metrics or len(self.recent_metrics) < 10:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Calculate recent averages
        recent_avg = {}
        for metric_name in self.baseline_metrics:
            values = [m['metrics'].get(metric_name, 0) for m in self.recent_metrics[-10:]]
            recent_avg[metric_name] = np.mean(values)
        
        # Check for drift
        drift_detected = False
        drift_details = {}
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            recent_value = recent_avg.get(metric_name, 0)
            
            # Calculate relative change
            if baseline_value != 0:
                relative_change = abs(recent_value - baseline_value) / baseline_value
            else:
                relative_change = 1.0 if recent_value != 0 else 0.0
            
            drift_details[metric_name] = {
                'baseline': baseline_value,
                'recent': recent_value,
                'relative_change': relative_change,
                'drift': relative_change > self.threshold
            }
            
            if relative_change > self.threshold:
                drift_detected = True
        
        return {
            'drift_detected': drift_detected,
            'drift_details': drift_details,
            'threshold': self.threshold,
            'window_size': len(self.recent_metrics)
        }

class ModelRegistry:
    """Central model registry for version management"""
    
    def __init__(self, registry_path: str = "mlops/model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelVersion] = {}
        self.deployments: Dict[str, Deployment] = {}
        
        self._load_registry()
    
    def register_model(
        self,
        name: str,
        model_type: str,
        framework: str,
        artifacts_path: str,
        description: str = "",
        tags: List[str] = None,
        config: Dict[str, Any] = None,
        metrics: Optional[ModelMetrics] = None,
        created_by: str = "system"
    ) -> str:
        """Register a new model version"""
        
        # Generate version
        existing_versions = [m for m in self.models.values() if m.name == name]
        version_number = len(existing_versions) + 1
        version = f"v{version_number}"
        
        model_id = str(uuid.uuid4())
        
        model_version = ModelVersion(
            id=model_id,
            name=name,
            version=version,
            status=ModelStatus.STAGING,
            model_type=model_type,
            framework=framework,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or [],
            metrics=metrics,
            artifacts_path=artifacts_path,
            config=config or {},
            parent_version=self._get_latest_version_id(name),
            experiments=[]
        )
        
        self.models[model_id] = model_version
        self._save_registry()
        
        logger.info(f"Registered model {name} {version} with ID {model_id}")
        return model_id
    
    def promote_model(self, model_id: str, target_status: ModelStatus) -> bool:
        """Promote model to different stage"""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        model = self.models[model_id]
        old_status = model.status
        model.status = target_status
        
        self._save_registry()
        logger.info(f"Promoted model {model.name} from {old_status.value} to {target_status.value}")
        return True
    
    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def get_latest_model(self, name: str, status: ModelStatus = None) -> Optional[ModelVersion]:
        """Get latest model version by name"""
        matching_models = [
            m for m in self.models.values() 
            if m.name == name and (status is None or m.status == status)
        ]
        
        if not matching_models:
            return None
        
        return max(matching_models, key=lambda m: m.created_at)
    
    def list_models(self, name: str = None, status: ModelStatus = None) -> List[ModelVersion]:
        """List models with optional filters"""
        models = list(self.models.values())
        
        if name:
            models = [m for m in models if m.name == name]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def retire_model(self, model_id: str) -> bool:
        """Retire a model version"""
        return self.promote_model(model_id, ModelStatus.RETIRED)
    
    def _get_latest_version_id(self, name: str) -> Optional[str]:
        """Get ID of latest version for a model name"""
        latest = self.get_latest_model(name)
        return latest.id if latest else None
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = self.registry_path / "registry.json"
        
        data = {
            'models': {k: asdict(v) for k, v in self.models.items()},
            'deployments': {k: asdict(v) for k, v in self.deployments.items()}
        }
        
        # Convert datetime objects to strings
        for model_data in data['models'].values():
            model_data['created_at'] = model_data['created_at'].isoformat() if isinstance(model_data['created_at'], datetime) else model_data['created_at']
            model_data['status'] = model_data['status'].value if hasattr(model_data['status'], 'value') else model_data['status']
        
        for deployment_data in data['deployments'].values():
            deployment_data['deployed_at'] = deployment_data['deployed_at'].isoformat() if isinstance(deployment_data['deployed_at'], datetime) else deployment_data['deployed_at']
            deployment_data['strategy'] = deployment_data['strategy'].value if hasattr(deployment_data['strategy'], 'value') else deployment_data['strategy']
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_registry(self):
        """Load registry from disk"""
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
            
            # Load models
            for model_id, model_data in data.get('models', {}).items():
                model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                model_data['status'] = ModelStatus(model_data['status'])
                
                if model_data.get('metrics'):
                    model_data['metrics'] = ModelMetrics(**model_data['metrics'])
                
                self.models[model_id] = ModelVersion(**model_data)
            
            # Load deployments
            for deployment_id, deployment_data in data.get('deployments', {}).items():
                deployment_data['deployed_at'] = datetime.fromisoformat(deployment_data['deployed_at'])
                deployment_data['strategy'] = DeploymentStrategy(deployment_data['strategy'])
                
                self.deployments[deployment_id] = Deployment(**deployment_data)
                
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")

class DeploymentManager:
    """Manage model deployments and rollbacks"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_deployments = {}
        self.deployment_monitors = {}
    
    def deploy_model(
        self,
        model_id: str,
        environment: str,
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        traffic_percentage: float = 100.0,
        deployed_by: str = "system"
    ) -> str:
        """Deploy a model version"""
        
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if model.status not in [ModelStatus.STAGING, ModelStatus.PRODUCTION]:
            raise ValueError(f"Model must be in staging or production status, got {model.status}")
        
        deployment_id = str(uuid.uuid4())
        
        # Get current production deployment for rollback
        current_deployment = self._get_current_deployment(environment)
        rollback_version = current_deployment.model_version_id if current_deployment else None
        
        deployment = Deployment(
            id=deployment_id,
            model_version_id=model_id,
            environment=environment,
            strategy=strategy,
            traffic_percentage=traffic_percentage,
            deployed_at=datetime.now(),
            deployed_by=deployed_by,
            status="deploying",
            health_check_url=f"/health/{deployment_id}",
            rollback_version=rollback_version
        )
        
        try:
            # Perform deployment based on strategy
            if strategy == DeploymentStrategy.BLUE_GREEN:
                self._deploy_blue_green(deployment, model)
            elif strategy == DeploymentStrategy.CANARY:
                self._deploy_canary(deployment, model)
            elif strategy == DeploymentStrategy.ROLLING:
                self._deploy_rolling(deployment, model)
            else:  # IMMEDIATE
                self._deploy_immediate(deployment, model)
            
            deployment.status = "deployed"
            self.registry.deployments[deployment_id] = deployment
            self.active_deployments[environment] = deployment_id
            
            # Start monitoring
            self._start_deployment_monitoring(deployment_id)
            
            # Promote model to production if successful
            if environment == "production":
                self.registry.promote_model(model_id, ModelStatus.PRODUCTION)
            
            logger.info(f"Successfully deployed model {model.name} {model.version} to {environment}")
            return deployment_id
            
        except Exception as e:
            deployment.status = "failed"
            logger.error(f"Deployment failed: {str(e)}")
            raise
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback to previous deployment"""
        deployment = self.registry.deployments.get(deployment_id)
        if not deployment:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        if not deployment.rollback_version:
            logger.error("No rollback version available")
            return False
        
        try:
            # Deploy the rollback version
            rollback_deployment_id = self.deploy_model(
                model_id=deployment.rollback_version,
                environment=deployment.environment,
                strategy=DeploymentStrategy.IMMEDIATE,
                deployed_by="system_rollback"
            )
            
            # Mark original deployment as rolled back
            deployment.status = "rolled_back"
            
            logger.info(f"Successfully rolled back deployment {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False
    
    def _deploy_blue_green(self, deployment: Deployment, model: ModelVersion):
        """Blue-green deployment strategy"""
        logger.info("Executing blue-green deployment")
        
        # In a real implementation, this would:
        # 1. Spin up new environment (green)
        # 2. Deploy model to green environment
        # 3. Run health checks
        # 4. Switch traffic from blue to green
        # 5. Keep blue as backup for quick rollback
        
        time.sleep(2)  # Simulate deployment time
    
    def _deploy_canary(self, deployment: Deployment, model: ModelVersion):
        """Canary deployment strategy"""
        logger.info(f"Executing canary deployment with {deployment.traffic_percentage}% traffic")
        
        # In a real implementation, this would:
        # 1. Deploy to subset of infrastructure
        # 2. Route small percentage of traffic to new version
        # 3. Monitor metrics and user feedback
        # 4. Gradually increase traffic if metrics look good
        # 5. Full rollout or rollback based on results
        
        time.sleep(1)  # Simulate deployment time
    
    def _deploy_rolling(self, deployment: Deployment, model: ModelVersion):
        """Rolling deployment strategy"""
        logger.info("Executing rolling deployment")
        
        # In a real implementation, this would:
        # 1. Deploy to one instance at a time
        # 2. Health check each instance
        # 3. Continue to next instance if healthy
        # 4. Rollback if any instance fails
        
        time.sleep(1.5)  # Simulate deployment time
    
    def _deploy_immediate(self, deployment: Deployment, model: ModelVersion):
        """Immediate deployment strategy"""
        logger.info("Executing immediate deployment")
        
        # In a real implementation, this would:
        # 1. Stop current version
        # 2. Deploy new version
        # 3. Start new version
        # 4. Run health checks
        
        time.sleep(0.5)  # Simulate deployment time
    
    def _get_current_deployment(self, environment: str) -> Optional[Deployment]:
        """Get currently active deployment for environment"""
        deployment_id = self.active_deployments.get(environment)
        if deployment_id:
            return self.registry.deployments.get(deployment_id)
        return None
    
    def _start_deployment_monitoring(self, deployment_id: str):
        """Start monitoring deployment health"""
        monitor = PerformanceDriftDetector()
        self.deployment_monitors[deployment_id] = monitor
        
        # In production, this would start background monitoring tasks
        logger.info(f"Started monitoring for deployment {deployment_id}")

class ModelManager:
    """Main model lifecycle management class"""
    
    def __init__(self, registry_path: str = "mlops/model_registry"):
        self.registry = ModelRegistry(registry_path)
        self.deployment_manager = DeploymentManager(self.registry)
        self.model_cache = {}
        
    def create_model_version(
        self,
        name: str,
        model_type: str,
        model_config: Dict[str, Any],
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """Create new model version"""
        
        # Create artifacts directory
        artifacts_path = f"mlops/artifacts/{name}/v{len(self.registry.list_models(name)) + 1}"
        Path(artifacts_path).mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config_path = Path(artifacts_path) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Register model
        model_id = self.registry.register_model(
            name=name,
            model_type=model_type,
            framework="ollama" if model_type == "llm" else "scikit-learn",
            artifacts_path=artifacts_path,
            description=description,
            tags=tags,
            config=model_config
        )
        
        logger.info(f"Created model version {name} with ID {model_id}")
        return model_id
    
    def evaluate_model(self, model_id: str, test_data: List[Dict[str, Any]]) -> ModelMetrics:
        """Evaluate model performance"""
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Simulate model evaluation
        # In real implementation, this would load and test the actual model
        
        metrics = ModelMetrics(
            accuracy=0.85 + np.random.normal(0, 0.05),
            precision=0.83 + np.random.normal(0, 0.05),
            recall=0.87 + np.random.normal(0, 0.05),
            f1_score=0.85 + np.random.normal(0, 0.05),
            response_time_ms=150 + np.random.normal(0, 20),
            throughput_rps=50 + np.random.normal(0, 5),
            error_rate=0.02 + np.random.normal(0, 0.01),
            user_satisfaction=4.2 + np.random.normal(0, 0.3),
            custom_metrics={}
        )
        
        # Update model with metrics
        model.metrics = metrics
        self.registry._save_registry()
        
        logger.info(f"Evaluated model {model.name} - F1: {metrics.f1_score:.3f}")
        return metrics
    
    def load_model_for_inference(self, model_id: str) -> Any:
        """Load model for inference"""
        if model_id in self.model_cache:
            return self.model_cache[model_id]
        
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if model.status not in [ModelStatus.STAGING, ModelStatus.PRODUCTION]:
            raise ValueError(f"Model not ready for inference: {model.status}")
        
        # Load model based on type
        if model.model_type == "llm":
            # For LLM models, return configuration for Ollama
            model_instance = {
                'type': 'llm',
                'config': model.config,
                'version': model.version,
                'id': model.id
            }
        else:
            # For other models, load from artifacts
            try:
                artifacts_path = Path(model.artifacts_path)
                model_file = artifacts_path / "model.pkl"
                
                if model_file.exists():
                    model_instance = joblib.load(model_file)
                else:
                    # Fallback to configuration-based model
                    model_instance = {
                        'type': model.model_type,
                        'config': model.config,
                        'version': model.version,
                        'id': model.id
                    }
                    
            except Exception as e:
                logger.error(f"Error loading model artifacts: {str(e)}")
                raise
        
        # Cache the model
        self.model_cache[model_id] = model_instance
        
        logger.info(f"Loaded model {model.name} {model.version} for inference")
        return model_instance
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        comparison = {
            'models': {},
            'best_model': None,
            'comparison_metrics': {}
        }
        
        models_data = []
        for model_id in model_ids:
            model = self.registry.get_model(model_id)
            if model and model.metrics:
                models_data.append({
                    'id': model_id,
                    'name': model.name,
                    'version': model.version,
                    'metrics': model.metrics
                })
                comparison['models'][model_id] = {
                    'name': f"{model.name} {model.version}",
                    'metrics': asdict(model.metrics)
                }
        
        if not models_data:
            return comparison
        
        # Find best model by F1 score
        best_model = max(models_data, key=lambda m: m['metrics'].f1_score)
        comparison['best_model'] = best_model['id']
        
        # Calculate metric comparisons
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'response_time_ms']
        
        for metric_name in metrics_names:
            values = [getattr(m['metrics'], metric_name) for m in models_data]
            comparison['comparison_metrics'][metric_name] = {
                'best': max(values) if metric_name != 'response_time_ms' else min(values),
                'worst': min(values) if metric_name != 'response_time_ms' else max(values),
                'average': np.mean(values),
                'std': np.std(values)
            }
        
        return comparison
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get model version lineage"""
        model = self.registry.get_model(model_id)
        if not model:
            return {}
        
        lineage = {
            'current': {
                'id': model.id,
                'name': model.name,
                'version': model.version,
                'status': model.status.value,
                'created_at': model.created_at.isoformat()
            },
            'parents': [],
            'children': []
        }
        
        # Find parent versions
        current_model = model
        while current_model.parent_version:
            parent = self.registry.get_model(current_model.parent_version)
            if parent:
                lineage['parents'].append({
                    'id': parent.id,
                    'version': parent.version,
                    'status': parent.status.value,
                    'created_at': parent.created_at.isoformat()
                })
                current_model = parent
            else:
                break
        
        # Find child versions
        for other_model in self.registry.models.values():
            if other_model.parent_version == model_id:
                lineage['children'].append({
                    'id': other_model.id,
                    'version': other_model.version,
                    'status': other_model.status.value,
                    'created_at': other_model.created_at.isoformat()
                })
        
        return lineage
    
    def monitor_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance monitoring data"""
        deployment_id = None
        
        # Find active deployment for this model
        for dep_id, deployment in self.registry.deployments.items():
            if deployment.model_version_id == model_id and deployment.status == "deployed":
                deployment_id = dep_id
                break
        
        if not deployment_id:
            return {'error': 'No active deployment found for model'}
        
        monitor = self.deployment_manager.deployment_monitors.get(deployment_id)
        if not monitor:
            return {'error': 'No monitor found for deployment'}
        
        # Get drift detection results
        drift_analysis = monitor.detect_drift()
        
        return {
            'model_id': model_id,
            'deployment_id': deployment_id,
            'drift_analysis': drift_analysis,
            'recent_measurements': len(monitor.recent_metrics),
            'baseline_set': bool(monitor.baseline_metrics)
        }
    
    def get_deployment_status(self, environment: str = "production") -> Dict[str, Any]:
        """Get current deployment status"""
        current_deployment = self.deployment_manager._get_current_deployment(environment)
        
        if not current_deployment:
            return {'environment': environment, 'status': 'no_deployment'}
        
        model = self.registry.get_model(current_deployment.model_version_id)
        
        return {
            'environment': environment,
            'deployment_id': current_deployment.id,
            'model_name': model.name if model else 'unknown',
            'model_version': model.version if model else 'unknown',
            'deployed_at': current_deployment.deployed_at.isoformat(),
            'status': current_deployment.status,
            'traffic_percentage': current_deployment.traffic_percentage,
            'strategy': current_deployment.strategy.value,
            'has_rollback': current_deployment.rollback_version is not None
        }