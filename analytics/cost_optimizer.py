"""
Cost Optimization Engine
Advanced cost management and optimization features:
- Token usage optimization
- Model selection based on cost/performance
- Caching strategies to reduce API calls
- Resource allocation optimization
- Cost prediction and budgeting
- Automated cost alerts
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, deque
import threading
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    timestamp: datetime
    tokens_used: int
    api_calls: int
    processing_time: float
    model_used: str
    estimated_cost: float
    cache_hit_rate: float
    efficiency_score: float

@dataclass
class OptimizationRule:
    rule_id: str
    condition: str
    action: str
    priority: int
    savings_potential: float
    enabled: bool

class TokenOptimizer:
    """Optimizes token usage across the RAG pipeline"""
    
    def __init__(self):
        self.token_history = deque(maxlen=10000)
        self.optimization_rules = []
        self.cache_stats = defaultdict(int)
        
    def optimize_prompt(self, prompt: str, context: List[str], max_tokens: int) -> Tuple[str, List[str]]:
        """Optimize prompt and context for minimal token usage"""
        # Calculate token estimates
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        context_tokens = sum(len(ctx.split()) * 1.3 for ctx in context)
        
        total_tokens = prompt_tokens + context_tokens
        
        if total_tokens <= max_tokens * 0.8:  # Leave room for response
            return prompt, context
        
        # Optimize context
        optimized_context = self._optimize_context(context, max_tokens - prompt_tokens)
        
        # Optimize prompt if still too long
        if prompt_tokens + sum(len(ctx.split()) * 1.3 for ctx in optimized_context) > max_tokens * 0.8:
            optimized_prompt = self._optimize_prompt(prompt, max_tokens * 0.3)
            return optimized_prompt, optimized_context
        
        return prompt, optimized_context
    
    def _optimize_context(self, context: List[str], max_tokens: float) -> List[str]:
        """Prioritize and truncate context based on relevance"""
        # Score each context piece
        scored_context = []
        for i, ctx in enumerate(context):
            score = self._calculate_relevance_score(ctx, i)
            scored_context.append((score, ctx))
        
        # Sort by relevance
        scored_context.sort(reverse=True, key=lambda x: x[0])
        
        # Select context within token limit
        optimized = []
        current_tokens = 0
        for score, ctx in scored_context:
            ctx_tokens = len(ctx.split()) * 1.3
            if current_tokens + ctx_tokens <= max_tokens:
                optimized.append(ctx)
                current_tokens += ctx_tokens
            else:
                # Truncate if partially fits
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # Minimum useful context
                    words = ctx.split()
                    truncated_words = int(remaining_tokens / 1.3)
                    optimized.append(' '.join(words[:truncated_words]))
                break
        
        return optimized
    
    def _optimize_prompt(self, prompt: str, max_tokens: float) -> str:
        """Compress prompt while maintaining meaning"""
        words = prompt.split()
        max_words = int(max_tokens / 1.3)
        
        if len(words) <= max_words:
            return prompt
        
        # Keep essential parts: question and instructions
        lines = prompt.split('\n')
        essential_lines = []
        current_words = 0
        
        for line in lines:
            line_words = len(line.split())
            if current_words + line_words <= max_words:
                essential_lines.append(line)
                current_words += line_words
            else:
                break
        
        return '\n'.join(essential_lines)
    
    def _calculate_relevance_score(self, context: str, position: int) -> float:
        """Calculate relevance score for context ranking"""
        # Higher score for earlier positions (search ranking)
        position_score = 1.0 / (position + 1)
        
        # Length penalty for very long context
        length_penalty = 1.0 if len(context) < 1000 else 0.8
        
        # Keyword density (simple heuristic)
        keywords = ['error', 'problem', 'solution', 'help', 'support', 'issue']
        keyword_score = sum(1 for word in keywords if word.lower() in context.lower()) / len(keywords)
        
        return (position_score * 0.6) + (length_penalty * 0.3) + (keyword_score * 0.1)

class ModelSelector:
    """Intelligently selects models based on cost/performance trade-offs"""
    
    def __init__(self):
        self.model_performance = {
            'llama3.2:1b': {'cost_per_1k': 0.0, 'avg_quality': 0.7, 'avg_latency': 0.5},
            'llama3.2:3b': {'cost_per_1k': 0.0, 'avg_quality': 0.8, 'avg_latency': 0.8},
            'llama3.2:7b': {'cost_per_1k': 0.0, 'avg_quality': 0.9, 'avg_latency': 1.5},
        }
        self.query_complexity_cache = {}
        
    def select_optimal_model(self, query: str, budget_constraint: float = None) -> str:
        """Select the most cost-effective model for the query"""
        complexity = self._analyze_query_complexity(query)
        
        # Calculate cost-effectiveness for each model
        best_model = None
        best_score = 0
        
        for model, metrics in self.model_performance.items():
            if budget_constraint and metrics['cost_per_1k'] > budget_constraint:
                continue
            
            # Score based on quality vs cost trade-off
            quality_score = metrics['avg_quality']
            cost_penalty = metrics['cost_per_1k'] * 0.1  # Adjust weight
            latency_penalty = metrics['avg_latency'] * 0.05
            
            # Adjust for complexity
            if complexity < 0.3:  # Simple query
                quality_weight = 0.6
            elif complexity > 0.7:  # Complex query
                quality_weight = 0.9
            else:
                quality_weight = 0.75
            
            score = (quality_score * quality_weight) - cost_penalty - latency_penalty
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model or 'llama3.2:3b'  # Default fallback
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity to guide model selection"""
        # Cache for performance
        if query in self.query_complexity_cache:
            return self.query_complexity_cache[query]
        
        complexity_indicators = {
            'length': len(query.split()) / 100,  # Longer = more complex
            'questions': query.count('?') * 0.1,
            'keywords': sum(1 for word in ['analyze', 'compare', 'explain', 'why', 'how'] 
                          if word.lower() in query.lower()) * 0.1,
            'technical_terms': sum(1 for word in ['API', 'database', 'configuration', 'error'] 
                                 if word in query) * 0.1
        }
        
        complexity = min(1.0, sum(complexity_indicators.values()))
        self.query_complexity_cache[query] = complexity
        
        return complexity

class CacheOptimizer:
    """Manages intelligent caching strategies"""
    
    def __init__(self, max_cache_size: int = 10000):
        self.response_cache = {}
        self.embedding_cache = {}
        self.max_cache_size = max_cache_size
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(list)
        
    def get_cache_key(self, query: str, context_hash: str) -> str:
        """Generate cache key for query + context combination"""
        return f"{hash(query)}_{context_hash}"
    
    def should_cache(self, query: str, processing_time: float) -> bool:
        """Decide whether to cache based on processing time and patterns"""
        # Cache expensive operations
        if processing_time > 2.0:
            return True
        
        # Cache frequently accessed queries
        query_hash = str(hash(query))
        self.access_counts[query_hash] += 1
        
        if self.access_counts[query_hash] > 3:
            return True
        
        # Cache during high-traffic periods
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            return processing_time > 1.0
        
        return False
    
    def evict_cache_entries(self):
        """Intelligent cache eviction based on LRU + frequency"""
        if len(self.response_cache) <= self.max_cache_size:
            return
        
        # Score entries for eviction
        eviction_scores = {}
        current_time = time.time()
        
        for key in self.response_cache:
            access_count = self.access_counts.get(key, 0)
            last_access = max(self.access_times.get(key, [0]))
            time_since_access = current_time - last_access
            
            # Lower score = higher priority for eviction
            score = access_count * 0.7 - (time_since_access / 3600) * 0.3
            eviction_scores[key] = score
        
        # Remove lowest scoring entries
        to_remove = sorted(eviction_scores.items(), key=lambda x: x[1])[:len(self.response_cache) - self.max_cache_size]
        
        for key, _ in to_remove:
            del self.response_cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.access_times:
                del self.access_times[key]

class CostPredictor:
    """Predicts future costs and resource usage"""
    
    def __init__(self):
        self.usage_history = deque(maxlen=10000)
        self.seasonal_patterns = {}
        
    def predict_monthly_cost(self, current_usage: List[CostMetrics]) -> Dict[str, Any]:
        """Predict monthly costs based on current trends"""
        if not current_usage:
            return {'predicted_cost': 0, 'confidence': 0}
        
        # Calculate daily averages
        daily_tokens = defaultdict(list)
        daily_costs = defaultdict(list)
        
        for metric in current_usage[-30:]:  # Last 30 entries
            day = metric.timestamp.date()
            daily_tokens[day].append(metric.tokens_used)
            daily_costs[day].append(metric.estimated_cost)
        
        if not daily_tokens:
            return {'predicted_cost': 0, 'confidence': 0}
        
        # Calculate trends
        avg_daily_tokens = np.mean([np.sum(tokens) for tokens in daily_tokens.values()])
        avg_daily_cost = np.mean([np.sum(costs) for costs in daily_costs.values()])
        
        # Project to monthly
        monthly_prediction = avg_daily_cost * 30
        
        # Adjust for seasonal patterns
        current_month = datetime.now().month
        seasonal_multiplier = self.seasonal_patterns.get(current_month, 1.0)
        adjusted_prediction = monthly_prediction * seasonal_multiplier
        
        # Calculate confidence based on data consistency
        cost_variance = np.var([np.sum(costs) for costs in daily_costs.values()])
        confidence = max(0.1, 1.0 - (cost_variance / avg_daily_cost))
        
        return {
            'predicted_cost': adjusted_prediction,
            'confidence': confidence,
            'daily_average': avg_daily_cost,
            'monthly_tokens_projected': avg_daily_tokens * 30
        }
    
    def identify_cost_anomalies(self, recent_metrics: List[CostMetrics]) -> List[Dict[str, Any]]:
        """Identify unusual cost patterns"""
        if len(recent_metrics) < 10:
            return []
        
        anomalies = []
        
        # Calculate moving averages
        costs = [m.estimated_cost for m in recent_metrics]
        tokens = [m.tokens_used for m in recent_metrics]
        
        avg_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        for i, metric in enumerate(recent_metrics[-5:], len(recent_metrics)-5):
            # Cost anomaly
            if metric.estimated_cost > avg_cost + 2 * std_cost:
                anomalies.append({
                    'type': 'high_cost',
                    'timestamp': metric.timestamp,
                    'value': metric.estimated_cost,
                    'threshold': avg_cost + 2 * std_cost,
                    'severity': 'high' if metric.estimated_cost > avg_cost + 3 * std_cost else 'medium'
                })
            
            # Efficiency anomaly
            if metric.efficiency_score < 0.5:
                anomalies.append({
                    'type': 'low_efficiency',
                    'timestamp': metric.timestamp,
                    'value': metric.efficiency_score,
                    'severity': 'high' if metric.efficiency_score < 0.3 else 'medium'
                })
        
        return anomalies

class CostOptimizationEngine:
    """Main cost optimization coordinator"""
    
    def __init__(self):
        self.token_optimizer = TokenOptimizer()
        self.model_selector = ModelSelector()
        self.cache_optimizer = CacheOptimizer()
        self.cost_predictor = CostPredictor()
        self.optimization_rules = []
        self.cost_alerts = []
        
    def optimize_request(self, query: str, context: List[str], budget: float = None) -> Dict[str, Any]:
        """Optimize a single request for cost efficiency"""
        start_time = time.time()
        
        # Select optimal model
        optimal_model = self.model_selector.select_optimal_model(query, budget)
        
        # Optimize tokens
        optimized_prompt, optimized_context = self.token_optimizer.optimize_prompt(
            query, context, max_tokens=2048
        )
        
        # Check cache
        context_hash = str(hash(str(optimized_context)))
        cache_key = self.cache_optimizer.get_cache_key(optimized_prompt, context_hash)
        
        optimization_time = time.time() - start_time
        
        return {
            'original_query': query,
            'optimized_prompt': optimized_prompt,
            'optimized_context': optimized_context,
            'selected_model': optimal_model,
            'cache_key': cache_key,
            'optimization_time': optimization_time,
            'estimated_savings': self._calculate_savings(query, optimized_prompt, context, optimized_context)
        }
    
    def _calculate_savings(self, original_query: str, optimized_query: str, 
                         original_context: List[str], optimized_context: List[str]) -> Dict[str, float]:
        """Calculate estimated cost savings from optimization"""
        original_tokens = len(original_query.split()) + sum(len(ctx.split()) for ctx in original_context)
        optimized_tokens = len(optimized_query.split()) + sum(len(ctx.split()) for ctx in optimized_context)
        
        token_reduction = max(0, original_tokens - optimized_tokens)
        percentage_saved = (token_reduction / original_tokens) * 100 if original_tokens > 0 else 0
        
        return {
            'tokens_saved': token_reduction,
            'percentage_saved': percentage_saved,
            'estimated_cost_reduction': token_reduction * 0.001  # Approximate cost per token
        }
    
    def generate_cost_report(self, metrics: List[CostMetrics]) -> Dict[str, Any]:
        """Generate comprehensive cost analysis report"""
        if not metrics:
            return {'error': 'No metrics available'}
        
        total_cost = sum(m.estimated_cost for m in metrics)
        total_tokens = sum(m.tokens_used for m in metrics)
        avg_efficiency = np.mean([m.efficiency_score for m in metrics])
        
        # Predictions
        monthly_prediction = self.cost_predictor.predict_monthly_cost(metrics)
        anomalies = self.cost_predictor.identify_cost_anomalies(metrics)
        
        # Optimization opportunities
        low_efficiency_count = sum(1 for m in metrics if m.efficiency_score < 0.7)
        cache_hit_rate = np.mean([m.cache_hit_rate for m in metrics])
        
        return {
            'summary': {
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'average_efficiency': avg_efficiency,
                'cache_hit_rate': cache_hit_rate
            },
            'predictions': monthly_prediction,
            'anomalies': anomalies,
            'optimization_opportunities': {
                'low_efficiency_queries': low_efficiency_count,
                'cache_improvement_potential': max(0, 0.8 - cache_hit_rate),
                'estimated_monthly_savings': total_cost * 0.15  # Conservative estimate
            },
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: List[CostMetrics]) -> List[str]:
        """Generate actionable cost optimization recommendations"""
        recommendations = []
        
        avg_efficiency = np.mean([m.efficiency_score for m in metrics])
        if avg_efficiency < 0.7:
            recommendations.append("Implement query preprocessing to improve efficiency")
        
        cache_hit_rate = np.mean([m.cache_hit_rate for m in metrics])
        if cache_hit_rate < 0.5:
            recommendations.append("Increase cache size and optimize caching strategy")
        
        high_cost_queries = [m for m in metrics if m.estimated_cost > np.mean([m.estimated_cost for m in metrics]) * 2]
        if len(high_cost_queries) > len(metrics) * 0.1:
            recommendations.append("Review and optimize high-cost query patterns")
        
        if any(m.tokens_used > 3000 for m in metrics):
            recommendations.append("Implement context truncation for long documents")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    optimizer = CostOptimizationEngine()
    
    # Example optimization
    sample_query = "How do I reset my password and update my billing information?"
    sample_context = [
        "Password reset instructions: Go to login page, click forgot password...",
        "Billing information can be updated in your account settings...",
        "For security reasons, you'll need to verify your identity..."
    ]
    
    result = optimizer.optimize_request(sample_query, sample_context, budget=0.01)
    print("Optimization Result:")
    print(json.dumps(result, indent=2, default=str))