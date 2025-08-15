"""
A/B Testing Infrastructure for RAG System

Features:
- Experiment management and configuration
- Traffic splitting and user assignment
- Statistical significance testing
- Performance comparison between variants
- Automated experiment lifecycle management
- Real-time results monitoring
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentVariant:
    """Configuration for an experiment variant"""
    name: str
    description: str
    traffic_percentage: float
    configuration: Dict[str, Any]
    is_control: bool = False

@dataclass
class ExperimentMetric:
    """Metric definition for experiments"""
    name: str
    description: str
    metric_type: str  # 'conversion', 'continuous', 'count'
    goal: str  # 'increase', 'decrease'
    primary: bool = False

@dataclass
class ExperimentResult:
    """Single experimental observation"""
    experiment_id: str
    variant: str
    user_id: str
    session_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class Experiment:
    """Complete experiment configuration"""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    min_sample_size: int
    confidence_level: float
    created_at: datetime
    updated_at: datetime
    results: List[ExperimentResult]

class StatisticalAnalyzer:
    """Statistical analysis for A/B test results"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_sample_size(
        self, 
        baseline_rate: float, 
        minimum_detectable_effect: float,
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for experiment"""
        # Using formula for two-proportion z-test
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        p_pooled = (p1 + p2) / 2
        
        n = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
        
        return int(np.ceil(n))
    
    def analyze_conversion_metrics(
        self, 
        control_data: List[float], 
        treatment_data: List[float]
    ) -> Dict[str, Any]:
        """Analyze conversion rate metrics"""
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        
        if n_control == 0 or n_treatment == 0:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate conversion rates
        control_rate = np.mean(control_data)
        treatment_rate = np.mean(treatment_data)
        
        # Perform two-proportion z-test
        successes_control = int(np.sum(control_data))
        successes_treatment = int(np.sum(treatment_data))
        
        # Pooled proportion
        p_pooled = (successes_control + successes_treatment) / (n_control + n_treatment)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        
        # Z-score
        z_score = (treatment_rate - control_rate) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval for difference
        diff = treatment_rate - control_rate
        se_diff = np.sqrt((control_rate * (1 - control_rate) / n_control) + 
                         (treatment_rate * (1 - treatment_rate) / n_treatment))
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        # Statistical significance
        is_significant = p_value < self.alpha
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_difference': diff,
            'relative_difference': diff / control_rate if control_rate > 0 else 0,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_interval': [ci_lower, ci_upper],
            'sample_sizes': {'control': n_control, 'treatment': n_treatment}
        }
    
    def analyze_continuous_metrics(
        self, 
        control_data: List[float], 
        treatment_data: List[float]
    ) -> Dict[str, Any]:
        """Analyze continuous metrics using t-test"""
        if len(control_data) == 0 or len(treatment_data) == 0:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate basic statistics
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        control_std = np.std(control_data, ddof=1)
        treatment_std = np.std(treatment_data, ddof=1)
        
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * control_std**2 + 
                             (len(treatment_data) - 1) * treatment_std**2) / 
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference
        diff = treatment_mean - control_mean
        se_diff = np.sqrt(control_std**2/len(control_data) + treatment_std**2/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'absolute_difference': diff,
            'relative_difference': diff / control_mean if control_mean != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'confidence_interval': [ci_lower, ci_upper],
            'sample_sizes': {'control': len(control_data), 'treatment': len(treatment_data)}
        }

class ExperimentManager:
    """Manages A/B testing experiments"""
    
    def __init__(self, storage_path: str = "experiments/data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, Experiment] = {}
        self.analyzer = StatisticalAnalyzer()
        
        self._load_experiments()
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
        min_sample_size: int = 1000,
        confidence_level: float = 0.95
    ) -> str:
        """Create a new experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        # Validate traffic percentages
        total_traffic = sum(v['traffic_percentage'] for v in variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}")
        
        # Create variant objects
        variant_objects = []
        for v in variants:
            variant_objects.append(ExperimentVariant(
                name=v['name'],
                description=v['description'],
                traffic_percentage=v['traffic_percentage'],
                configuration=v['configuration'],
                is_control=v.get('is_control', False)
            ))
        
        # Create metric objects
        metric_objects = []
        for m in metrics:
            metric_objects.append(ExperimentMetric(
                name=m['name'],
                description=m['description'],
                metric_type=m['metric_type'],
                goal=m['goal'],
                primary=m.get('primary', False)
            ))
        
        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            variants=variant_objects,
            metrics=metric_objects,
            start_date=None,
            end_date=None,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            results=[]
        )
        
        self.experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Can only start experiments in DRAFT status, current: {experiment.status}")
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()
        experiment.updated_at = datetime.now()
        
        self._save_experiment(experiment)
        
        logger.info(f"Started experiment: {experiment.name}")
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now()
        experiment.updated_at = datetime.now()
        
        self._save_experiment(experiment)
        
        logger.info(f"Stopped experiment: {experiment.name}")
        return True
    
    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign a user to a variant"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            return None
        
        # Use consistent hashing for assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        assignment_value = (hash_value % 10000) / 100.0  # 0-100 range
        
        # Find variant based on traffic percentage
        cumulative_percentage = 0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if assignment_value <= cumulative_percentage:
                return variant.name
        
        # Fallback to control variant
        control_variant = next((v for v in experiment.variants if v.is_control), experiment.variants[0])
        return control_variant.name
    
    def record_result(
        self,
        experiment_id: str,
        user_id: str,
        session_id: str,
        variant: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Record an experimental result"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            logger.warning(f"Recording result for non-running experiment: {experiment.status}")
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            metrics=metrics,
            metadata=metadata or {}
        )
        
        experiment.results.append(result)
        experiment.updated_at = datetime.now()
        
        # Save every 10 results to avoid excessive I/O
        if len(experiment.results) % 10 == 0:
            self._save_experiment(experiment)
        
        return True
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results"""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        
        if not experiment.results:
            return {'error': 'No results to analyze'}
        
        # Group results by variant
        variant_data = {}
        for result in experiment.results:
            if result.variant not in variant_data:
                variant_data[result.variant] = {'results': [], 'metrics': {}}
            variant_data[result.variant]['results'].append(result)
        
        # Find control variant
        control_variant = None
        for variant in experiment.variants:
            if variant.is_control:
                control_variant = variant.name
                break
        
        if not control_variant:
            control_variant = experiment.variants[0].name
        
        # Analyze each metric
        metric_analyses = {}
        
        for metric in experiment.metrics:
            metric_name = metric.name
            
            # Extract metric values for each variant
            for variant_name, data in variant_data.items():
                values = []
                for result in data['results']:
                    if metric_name in result.metrics:
                        values.append(result.metrics[metric_name])
                data['metrics'][metric_name] = values
            
            # Compare with control
            if control_variant in variant_data and metric_name in variant_data[control_variant]['metrics']:
                control_values = variant_data[control_variant]['metrics'][metric_name]
                
                for variant_name, data in variant_data.items():
                    if variant_name == control_variant or metric_name not in data['metrics']:
                        continue
                    
                    treatment_values = data['metrics'][metric_name]
                    
                    # Choose analysis method based on metric type
                    if metric.metric_type == 'conversion':
                        analysis_result = self.analyzer.analyze_conversion_metrics(
                            control_values, treatment_values
                        )
                    else:  # continuous or count
                        analysis_result = self.analyzer.analyze_continuous_metrics(
                            control_values, treatment_values
                        )
                    
                    metric_analyses[f"{metric_name}_{variant_name}_vs_{control_variant}"] = {
                        'metric_name': metric_name,
                        'variant': variant_name,
                        'control': control_variant,
                        'metric_type': metric.metric_type,
                        'is_primary': metric.primary,
                        'analysis': analysis_result
                    }
        
        # Overall experiment summary
        total_participants = len(set(r.user_id for r in experiment.results))
        
        variant_summary = {}
        for variant_name, data in variant_data.items():
            variant_summary[variant_name] = {
                'participants': len(set(r.user_id for r in data['results'])),
                'observations': len(data['results']),
                'traffic_percentage': next(v.traffic_percentage for v in experiment.variants if v.name == variant_name)
            }
        
        # Check if experiment has reached minimum sample size
        min_sample_reached = total_participants >= experiment.min_sample_size
        
        # Check for statistical significance on primary metrics
        has_significant_result = any(
            analysis['analysis'].get('is_significant', False) 
            for analysis in metric_analyses.values() 
            if analysis['is_primary']
        )
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'status': experiment.status.value,
            'total_participants': total_participants,
            'min_sample_reached': min_sample_reached,
            'has_significant_result': has_significant_result,
            'variant_summary': variant_summary,
            'metric_analyses': metric_analyses,
            'duration_days': (datetime.now() - experiment.start_date).days if experiment.start_date else 0,
            'recommendation': self._generate_recommendation(experiment, metric_analyses, min_sample_reached)
        }
    
    def _generate_recommendation(
        self, 
        experiment: Experiment, 
        metric_analyses: Dict[str, Any], 
        min_sample_reached: bool
    ) -> str:
        """Generate recommendation based on experiment results"""
        
        if not min_sample_reached:
            return "Continue experiment - minimum sample size not reached"
        
        # Check primary metrics
        primary_analyses = [a for a in metric_analyses.values() if a['is_primary']]
        
        if not primary_analyses:
            return "Continue experiment - no primary metrics defined"
        
        significant_improvements = []
        significant_degradations = []
        
        for analysis in primary_analyses:
            if analysis['analysis'].get('is_significant', False):
                metric_type = analysis['analysis']
                goal = next(m.goal for m in experiment.metrics if m.name == analysis['metric_name'])
                
                if metric_type.get('relative_difference', 0) > 0:
                    if goal == 'increase':
                        significant_improvements.append(analysis)
                    else:
                        significant_degradations.append(analysis)
                else:
                    if goal == 'increase':
                        significant_degradations.append(analysis)
                    else:
                        significant_improvements.append(analysis)
        
        if significant_improvements and not significant_degradations:
            return "Recommend: Deploy winning variant - significant improvement detected"
        elif significant_degradations:
            return "Recommend: Stop experiment - significant degradation detected"
        else:
            return "Recommend: Continue experiment - no conclusive results yet"
    
    def get_experiment_config(self, experiment_id: str, variant: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific variant"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        for v in experiment.variants:
            if v.name == variant:
                return v.configuration
        
        return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        return [
            {
                'id': exp.id,
                'name': exp.name,
                'status': exp.status.value,
                'created_at': exp.created_at.isoformat(),
                'participants': len(set(r.user_id for r in exp.results)),
                'variants': len(exp.variants)
            }
            for exp in self.experiments.values()
        ]
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk"""
        file_path = self.storage_path / f"experiment_{experiment.id}.json"
        
        # Convert to serializable format
        data = asdict(experiment)
        
        # Convert datetime objects
        for key in ['created_at', 'updated_at', 'start_date', 'end_date']:
            if data[key]:
                data[key] = data[key].isoformat() if hasattr(data[key], 'isoformat') else data[key]
        
        # Convert results
        for result in data['results']:
            result['timestamp'] = result['timestamp'].isoformat() if hasattr(result['timestamp'], 'isoformat') else result['timestamp']
        
        # Convert status enum
        data['status'] = data['status'].value if hasattr(data['status'], 'value') else data['status']
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_experiments(self):
        """Load experiments from disk"""
        for file_path in self.storage_path.glob("experiment_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert datetime strings back to objects
                for key in ['created_at', 'updated_at', 'start_date', 'end_date']:
                    if data[key]:
                        data[key] = datetime.fromisoformat(data[key])
                
                # Convert results
                results = []
                for result_data in data['results']:
                    result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp'])
                    results.append(ExperimentResult(**result_data))
                
                data['results'] = results
                
                # Convert status
                data['status'] = ExperimentStatus(data['status'])
                
                # Convert variants
                variants = []
                for variant_data in data['variants']:
                    variants.append(ExperimentVariant(**variant_data))
                data['variants'] = variants
                
                # Convert metrics
                metrics = []
                for metric_data in data['metrics']:
                    metrics.append(ExperimentMetric(**metric_data))
                data['metrics'] = metrics
                
                experiment = Experiment(**data)
                self.experiments[experiment.id] = experiment
                
            except Exception as e:
                logger.error(f"Error loading experiment from {file_path}: {str(e)}")

# Example RAG experiment configurations
RAG_EXPERIMENT_CONFIGS = {
    "retrieval_methods": {
        "name": "RAG Retrieval Methods Comparison",
        "description": "Compare standard retrieval vs HyDE vs query expansion",
        "variants": [
            {
                "name": "control_standard",
                "description": "Standard semantic search",
                "traffic_percentage": 40.0,
                "is_control": True,
                "configuration": {
                    "use_hyde": False,
                    "use_expansion": False,
                    "top_k": 5,
                    "score_threshold": 0.3
                }
            },
            {
                "name": "treatment_hyde",
                "description": "HyDE retrieval method",
                "traffic_percentage": 30.0,
                "configuration": {
                    "use_hyde": True,
                    "use_expansion": False,
                    "top_k": 5,
                    "score_threshold": 0.3
                }
            },
            {
                "name": "treatment_expansion",
                "description": "Query expansion method",
                "traffic_percentage": 30.0,
                "configuration": {
                    "use_hyde": False,
                    "use_expansion": True,
                    "top_k": 8,
                    "score_threshold": 0.25
                }
            }
        ],
        "metrics": [
            {
                "name": "answer_quality_rating",
                "description": "User rating of answer quality (1-5)",
                "metric_type": "continuous",
                "goal": "increase",
                "primary": True
            },
            {
                "name": "response_time_ms",
                "description": "Response generation time",
                "metric_type": "continuous",
                "goal": "decrease"
            },
            {
                "name": "user_satisfied",
                "description": "User satisfaction (thumbs up/down)",
                "metric_type": "conversion",
                "goal": "increase",
                "primary": True
            }
        ]
    },
    
    "model_parameters": {
        "name": "Model Parameter Optimization",
        "description": "Test different temperature and max_tokens settings",
        "variants": [
            {
                "name": "control_balanced",
                "description": "Balanced parameters",
                "traffic_percentage": 50.0,
                "is_control": True,
                "configuration": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.9
                }
            },
            {
                "name": "treatment_creative",
                "description": "More creative parameters",
                "traffic_percentage": 50.0,
                "configuration": {
                    "temperature": 0.9,
                    "max_tokens": 3072,
                    "top_p": 0.95
                }
            }
        ],
        "metrics": [
            {
                "name": "answer_helpfulness",
                "description": "Perceived helpfulness (1-5)",
                "metric_type": "continuous",
                "goal": "increase",
                "primary": True
            },
            {
                "name": "answer_length",
                "description": "Response length in characters",
                "metric_type": "continuous",
                "goal": "increase"
            }
        ]
    }
}