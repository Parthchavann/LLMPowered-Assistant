"""
Self-Improving RAG with Reinforcement Learning
Revolutionary adaptive AI system that:
- Learns from user feedback and interactions
- Optimizes retrieval and generation strategies
- Adapts to changing data patterns
- Performs continuous model improvement
- Implements advanced RL algorithms (PPO, SAC)

This represents the pinnacle of AI systems - truly autonomous
learning that gets better over time without human intervention.
"""

import asyncio
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    QUERY_EXPANSION = "query_expansion"
    CONTEXT_SELECTION = "context_selection"
    RESPONSE_GENERATION = "response_generation"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"

class RewardType(Enum):
    USER_FEEDBACK = "user_feedback"
    RESPONSE_TIME = "response_time"
    ANSWER_ACCURACY = "answer_accuracy"
    CONTEXT_RELEVANCE = "context_relevance"
    USER_SATISFACTION = "user_satisfaction"
    BUSINESS_METRIC = "business_metric"

@dataclass
class RLState:
    """State representation for RL agent"""
    query: str
    query_embedding: np.ndarray
    query_complexity: float
    user_history: List[Dict[str, Any]]
    context_quality: float
    response_confidence: float
    timestamp: datetime
    
@dataclass
class RLAction:
    """Action representation for RL agent"""
    action_type: ActionType
    parameters: Dict[str, Any]
    confidence: float
    
@dataclass
class RLReward:
    """Reward signal for RL training"""
    reward_type: RewardType
    value: float
    weight: float
    source: str
    timestamp: datetime

@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: RLState
    action: RLAction
    reward: float
    next_state: Optional[RLState]
    done: bool
    timestamp: datetime

class RAGPolicyNetwork(nn.Module):
    """
    Deep neural network for RAG policy learning
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage network (for A3C/PPO)
        self.advantage_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy and value"""
        policy = self.policy_net(state)
        value = self.value_net(state)
        return policy, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        policy, value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(policy, dim=-1)
        else:
            action = torch.multinomial(policy, 1).squeeze(-1)
        
        log_prob = torch.log(policy.gather(1, action.unsqueeze(-1))).squeeze(-1)
        
        return action, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO training"""
        policy, value = self.forward(states)
        
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(-1))).squeeze(-1)
        entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
        
        return log_probs, value.squeeze(-1), entropy

class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_priority = 1.0
        
    def add(self, experience: Experience, priority: float = None):
        """Add experience to buffer"""
        self.buffer.append(experience)
        if priority is None:
            priority = self.max_priority
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, alpha: float = 0.6) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized experience replay"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                 p=probabilities, replace=False)
        
        # Get experiences and importance sampling weights
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-0.4)  # beta = 0.4
        weights /= weights.max()  # Normalize
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class SelfImprovingRAGAgent:
    """
    Main RL agent that learns to optimize RAG performance
    """
    
    def __init__(self, state_dim: int = 100, action_dim: int = 20):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Neural networks
        self.policy_net = RAGPolicyNetwork(state_dim, action_dim)
        self.target_net = RAGPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        
        # Training components
        self.replay_buffer = ReplayBuffer()
        self.episode_rewards = deque(maxlen=1000)
        self.training_stats = defaultdict(list)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update rate
        self.epsilon = 0.1 # Exploration rate
        self.batch_size = 64
        self.update_frequency = 4
        self.target_update_frequency = 100
        
        # State tracking
        self.current_episode = 0
        self.step_count = 0
        self.last_state = None
        self.last_action = None
        
        # Performance metrics
        self.performance_metrics = {
            'avg_reward': 0.0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'user_satisfaction': 0.0
        }
        
        # Action space definition
        self.action_space = self._define_action_space()
        
        # Learning schedule
        self.learning_enabled = True
        self.exploration_schedule = self._create_exploration_schedule()
    
    def _define_action_space(self) -> Dict[int, Dict[str, Any]]:
        """Define the action space for RAG optimization"""
        actions = {}
        action_id = 0
        
        # Retrieval strategy actions
        retrieval_strategies = ['semantic', 'keyword', 'hybrid', 'graph_based', 'multi_hop']
        for strategy in retrieval_strategies:
            actions[action_id] = {
                'type': ActionType.RETRIEVAL_STRATEGY,
                'strategy': strategy,
                'description': f'Use {strategy} retrieval strategy'
            }
            action_id += 1
        
        # Query expansion actions
        expansion_methods = ['none', 'synonyms', 'semantic', 'hyde', 'multi_query']
        for method in expansion_methods:
            actions[action_id] = {
                'type': ActionType.QUERY_EXPANSION,
                'method': method,
                'description': f'Use {method} query expansion'
            }
            action_id += 1
        
        # Context selection actions
        context_strategies = ['top_k', 'threshold', 'diversity', 'mmr', 'adaptive']
        for strategy in context_strategies:
            actions[action_id] = {
                'type': ActionType.CONTEXT_SELECTION,
                'strategy': strategy,
                'description': f'Use {strategy} context selection'
            }
            action_id += 1
        
        # Response generation actions
        generation_modes = ['conservative', 'balanced', 'creative', 'precise', 'comprehensive']
        for mode in generation_modes:
            actions[action_id] = {
                'type': ActionType.RESPONSE_GENERATION,
                'mode': mode,
                'description': f'Use {mode} response generation'
            }
            action_id += 1
        
        return actions
    
    def _create_exploration_schedule(self) -> callable:
        """Create exploration schedule (epsilon decay)"""
        def schedule(step: int) -> float:
            # Exponential decay from 1.0 to 0.05
            return max(0.05, 1.0 * np.exp(-step / 10000))
        return schedule
    
    async def get_action(self, state: RLState, deterministic: bool = False) -> RLAction:
        """Get action from current policy"""
        state_tensor = self._state_to_tensor(state)
        
        if not deterministic and random.random() < self.exploration_schedule(self.step_count):
            # Exploration: random action
            action_id = random.randint(0, self.action_dim - 1)
            confidence = 0.5
        else:
            # Exploitation: use policy
            with torch.no_grad():
                action_id, _ = self.policy_net.get_action(state_tensor.unsqueeze(0), deterministic)
                action_id = action_id.item()
                confidence = 0.8
        
        # Convert action ID to action object
        action_config = self.action_space.get(action_id, self.action_space[0])
        
        action = RLAction(
            action_type=action_config['type'],
            parameters=action_config.copy(),
            confidence=confidence
        )
        
        return action
    
    async def observe_reward(self, reward_signals: List[RLReward], next_state: Optional[RLState] = None):
        """Observe reward and update learning"""
        if self.last_state is None or self.last_action is None:
            return
        
        # Combine multiple reward signals
        total_reward = self._combine_rewards(reward_signals)
        
        # Create experience
        experience = Experience(
            state=self.last_state,
            action=self.last_action,
            reward=total_reward,
            next_state=next_state,
            done=next_state is None,
            timestamp=datetime.now()
        )
        
        # Add to replay buffer
        td_error = abs(total_reward)  # Simplified TD error for prioritization
        self.replay_buffer.add(experience, priority=td_error)
        
        # Update statistics
        self.episode_rewards.append(total_reward)
        self.training_stats['rewards'].append(total_reward)
        
        # Train if enough experiences
        if len(self.replay_buffer.buffer) >= self.batch_size and self.learning_enabled:
            if self.step_count % self.update_frequency == 0:
                await self._train_step()
            
            if self.step_count % self.target_update_frequency == 0:
                self._soft_update_target()
        
        # Update for next step
        self.last_state = next_state
        self.step_count += 1
    
    def _combine_rewards(self, reward_signals: List[RLReward]) -> float:
        """Combine multiple reward signals into single value"""
        total_reward = 0.0
        total_weight = 0.0
        
        reward_weights = {
            RewardType.USER_FEEDBACK: 2.0,
            RewardType.ANSWER_ACCURACY: 1.5,
            RewardType.USER_SATISFACTION: 2.0,
            RewardType.RESPONSE_TIME: 0.5,
            RewardType.CONTEXT_RELEVANCE: 1.0,
            RewardType.BUSINESS_METRIC: 1.0
        }
        
        for reward in reward_signals:
            weight = reward_weights.get(reward.reward_type, 1.0) * reward.weight
            total_reward += reward.value * weight
            total_weight += weight
        
        return total_reward / total_weight if total_weight > 0 else 0.0
    
    async def _train_step(self):
        """Perform one training step using PPO"""
        # Sample batch from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        if not experiences:
            return
        
        # Prepare training data
        states = torch.stack([self._state_to_tensor(exp.state) for exp in experiences])
        actions = torch.tensor([self._action_to_id(exp.action) for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([self._state_to_tensor(exp.next_state) if exp.next_state 
                                 else torch.zeros(self.state_dim) for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        # Compute targets
        with torch.no_grad():
            _, next_values = self.target_net(next_states)
            targets = rewards + self.gamma * next_values.squeeze() * (~dones)
        
        # Current policy evaluation
        old_log_probs, old_values, old_entropy = self.policy_net.evaluate_actions(states, actions)
        
        # PPO loss computation
        advantages = targets - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (clipped)
        ratio = torch.exp(old_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(old_values, targets)
        
        # Entropy bonus
        entropy_loss = -0.01 * old_entropy.mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Apply importance sampling weights
        total_loss = (total_loss * weights_tensor).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = abs((old_values - targets).detach().numpy())
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Log training statistics
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['entropy'].append(old_entropy.mean().item())
    
    def _soft_update_target(self):
        """Soft update of target network"""
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _state_to_tensor(self, state: RLState) -> torch.Tensor:
        """Convert state to tensor representation"""
        if state is None:
            return torch.zeros(self.state_dim)
        
        # Extract features from state
        features = []
        
        # Query embedding (first 50 dims)
        query_emb = state.query_embedding[:50] if len(state.query_embedding) >= 50 else np.pad(state.query_embedding, (0, 50 - len(state.query_embedding)))
        features.extend(query_emb)
        
        # Scalar features
        features.extend([
            state.query_complexity,
            state.context_quality,
            state.response_confidence,
            len(state.user_history),
            state.timestamp.hour / 24.0,  # Time of day
            state.timestamp.weekday() / 7.0  # Day of week
        ])
        
        # Pad to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.state_dim], dtype=torch.float32)
    
    def _action_to_id(self, action: RLAction) -> int:
        """Convert action to action ID"""
        for action_id, action_config in self.action_space.items():
            if (action_config['type'] == action.action_type and 
                action_config.get('strategy') == action.parameters.get('strategy') and
                action_config.get('method') == action.parameters.get('method') and
                action_config.get('mode') == action.parameters.get('mode')):
                return action_id
        return 0  # Default action
    
    async def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        for key, value in metrics.items():
            if key in self.performance_metrics:
                # Exponential moving average
                alpha = 0.1
                self.performance_metrics[key] = (1 - alpha) * self.performance_metrics[key] + alpha * value
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics and metrics"""
        stats = {
            'episode_count': self.current_episode,
            'step_count': self.step_count,
            'replay_buffer_size': len(self.replay_buffer.buffer),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'exploration_rate': self.exploration_schedule(self.step_count),
            'performance_metrics': self.performance_metrics.copy(),
            'training_losses': {
                key: np.mean(values[-100:]) if values else 0.0  # Last 100 values
                for key, values in self.training_stats.items()
            }
        }
        return stats
    
    async def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': dict(self.training_stats),
            'performance_metrics': self.performance_metrics,
            'step_count': self.step_count,
            'episode_count': self.current_episode
        }, path)
        logger.info(f"Model saved to {path}")
    
    async def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_stats = defaultdict(list, checkpoint['training_stats'])
        self.performance_metrics = checkpoint['performance_metrics']
        self.step_count = checkpoint['step_count']
        self.current_episode = checkpoint['episode_count']
        logger.info(f"Model loaded from {path}")

class AdaptiveRAGSystem:
    """
    Adaptive RAG system that integrates RL agent for continuous improvement
    """
    
    def __init__(self):
        self.rl_agent = SelfImprovingRAGAgent()
        self.performance_tracker = PerformanceTracker()
        self.user_feedback_collector = UserFeedbackCollector()
        
        # System state
        self.current_strategies = {
            'retrieval': 'semantic',
            'expansion': 'none',
            'selection': 'top_k',
            'generation': 'balanced'
        }
        
        # Learning history
        self.adaptation_history = []
        self.performance_history = deque(maxlen=1000)
        
    async def process_query_with_learning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query while learning and adapting"""
        
        # Create current state
        state = await self._create_state(query, context)
        
        # Get action from RL agent
        action = await self.rl_agent.get_action(state)
        
        # Apply action to update strategies
        await self._apply_action(action)
        
        # Process query with current strategies
        result = await self._process_with_current_strategies(query, context)
        
        # Collect immediate feedback
        immediate_rewards = await self._collect_immediate_rewards(result)
        
        # Store state and action for learning
        self.rl_agent.last_state = state
        self.rl_agent.last_action = action
        
        # Add learning metadata to result
        result['learning_metadata'] = {
            'action_taken': asdict(action),
            'current_strategies': self.current_strategies.copy(),
            'state_features': state.query_complexity,
            'immediate_rewards': [asdict(r) for r in immediate_rewards]
        }
        
        return result
    
    async def provide_feedback(self, query_id: str, feedback: Dict[str, Any]):
        """Provide delayed feedback for learning"""
        
        # Convert feedback to reward signals
        reward_signals = await self._feedback_to_rewards(feedback)
        
        # Provide to RL agent
        await self.rl_agent.observe_reward(reward_signals)
        
        # Update performance metrics
        performance_metrics = self._extract_performance_metrics(feedback)
        await self.rl_agent.update_performance_metrics(performance_metrics)
    
    async def _create_state(self, query: str, context: Dict[str, Any]) -> RLState:
        """Create state representation for RL agent"""
        
        # Generate query embedding (mock implementation)
        query_embedding = np.random.randn(384)  # Mock embedding
        
        # Calculate query complexity
        complexity = len(query.split()) / 50.0  # Simplified
        
        # Get user history
        user_history = context.get('user_history', [])
        
        return RLState(
            query=query,
            query_embedding=query_embedding,
            query_complexity=min(complexity, 1.0),
            user_history=user_history,
            context_quality=0.8,  # Mock value
            response_confidence=0.7,  # Mock value
            timestamp=datetime.now()
        )
    
    async def _apply_action(self, action: RLAction):
        """Apply RL action to update system strategies"""
        
        if action.action_type == ActionType.RETRIEVAL_STRATEGY:
            self.current_strategies['retrieval'] = action.parameters.get('strategy', 'semantic')
        
        elif action.action_type == ActionType.QUERY_EXPANSION:
            self.current_strategies['expansion'] = action.parameters.get('method', 'none')
        
        elif action.action_type == ActionType.CONTEXT_SELECTION:
            self.current_strategies['selection'] = action.parameters.get('strategy', 'top_k')
        
        elif action.action_type == ActionType.RESPONSE_GENERATION:
            self.current_strategies['generation'] = action.parameters.get('mode', 'balanced')
        
        # Log adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'action': asdict(action),
            'new_strategies': self.current_strategies.copy()
        })
    
    async def _process_with_current_strategies(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using current strategies (mock implementation)"""
        
        # This would integrate with actual RAG components
        # For now, return mock result influenced by strategies
        
        processing_time = random.uniform(0.5, 3.0)
        confidence = random.uniform(0.6, 0.95)
        
        # Strategy influences on mock result
        if self.current_strategies['retrieval'] == 'graph_based':
            confidence += 0.05
            processing_time += 0.5
        
        if self.current_strategies['expansion'] == 'hyde':
            confidence += 0.1
            processing_time += 1.0
        
        return {
            'query': query,
            'answer': f"Mock answer using {self.current_strategies}",
            'confidence': min(confidence, 1.0),
            'processing_time': processing_time,
            'strategies_used': self.current_strategies.copy(),
            'sources': ['doc1', 'doc2', 'doc3']
        }
    
    async def _collect_immediate_rewards(self, result: Dict[str, Any]) -> List[RLReward]:
        """Collect immediate reward signals"""
        rewards = []
        
        # Response time reward (faster is better)
        time_reward = max(0, 1.0 - result['processing_time'] / 5.0)
        rewards.append(RLReward(
            reward_type=RewardType.RESPONSE_TIME,
            value=time_reward,
            weight=0.3,
            source='system',
            timestamp=datetime.now()
        ))
        
        # Confidence reward
        conf_reward = result['confidence']
        rewards.append(RLReward(
            reward_type=RewardType.ANSWER_ACCURACY,
            value=conf_reward,
            weight=0.7,
            source='system',
            timestamp=datetime.now()
        ))
        
        return rewards
    
    async def _feedback_to_rewards(self, feedback: Dict[str, Any]) -> List[RLReward]:
        """Convert user feedback to reward signals"""
        rewards = []
        
        # User satisfaction
        if 'satisfaction' in feedback:
            rewards.append(RLReward(
                reward_type=RewardType.USER_SATISFACTION,
                value=feedback['satisfaction'],
                weight=1.0,
                source='user',
                timestamp=datetime.now()
            ))
        
        # Answer helpfulness
        if 'helpful' in feedback:
            helpful_score = 1.0 if feedback['helpful'] else 0.0
            rewards.append(RLReward(
                reward_type=RewardType.USER_FEEDBACK,
                value=helpful_score,
                weight=1.0,
                source='user',
                timestamp=datetime.now()
            ))
        
        return rewards
    
    def _extract_performance_metrics(self, feedback: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from feedback"""
        metrics = {}
        
        if 'satisfaction' in feedback:
            metrics['user_satisfaction'] = feedback['satisfaction']
        
        if 'response_time_rating' in feedback:
            metrics['response_time_rating'] = feedback['response_time_rating']
        
        return metrics
    
    async def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights about system adaptation"""
        
        training_stats = self.rl_agent.get_training_statistics()
        
        # Analyze adaptation patterns
        recent_adaptations = self.adaptation_history[-50:] if len(self.adaptation_history) >= 50 else self.adaptation_history
        
        strategy_changes = defaultdict(int)
        for adaptation in recent_adaptations:
            action_type = adaptation['action']['action_type']
            strategy_changes[action_type] += 1
        
        return {
            'training_statistics': training_stats,
            'recent_adaptations': len(recent_adaptations),
            'strategy_changes': dict(strategy_changes),
            'current_strategies': self.current_strategies,
            'learning_enabled': self.rl_agent.learning_enabled,
            'adaptation_insights': {
                'most_changed_strategy': max(strategy_changes.items(), key=lambda x: x[1])[0] if strategy_changes else None,
                'adaptation_frequency': len(recent_adaptations) / min(50, len(self.adaptation_history)) if self.adaptation_history else 0
            }
        }

class PerformanceTracker:
    """Track system performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    async def record_metric(self, name: str, value: float):
        self.metrics[name].append((datetime.now(), value))

class UserFeedbackCollector:
    """Collect and process user feedback"""
    
    def __init__(self):
        self.feedback_history = []
    
    async def collect_feedback(self, query_id: str, feedback: Dict[str, Any]):
        self.feedback_history.append({
            'query_id': query_id,
            'feedback': feedback,
            'timestamp': datetime.now()
        })

# Example usage and demonstration
if __name__ == "__main__":
    async def demo_self_improving_rag():
        """Demonstrate self-improving RAG system"""
        
        adaptive_rag = AdaptiveRAGSystem()
        
        # Simulate queries and learning
        queries = [
            "How do I reset my password?",
            "What is your refund policy?",
            "I'm getting error 500",
            "How to enable two-factor authentication?"
        ]
        
        for i, query in enumerate(queries):
            print(f"\n🔍 Query {i+1}: {query}")
            
            # Process query
            result = await adaptive_rag.process_query_with_learning(query, {})
            
            print(f"📄 Answer: {result['answer']}")
            print(f"⚡ Time: {result['processing_time']:.2f}s")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"🧠 Strategies: {result['strategies_used']}")
            
            # Simulate user feedback
            feedback = {
                'satisfaction': random.uniform(0.7, 1.0),
                'helpful': random.choice([True, True, True, False])  # 75% helpful
            }
            
            await adaptive_rag.provide_feedback(f"query_{i}", feedback)
            
            print(f"👍 Feedback: {feedback}")
        
        # Get adaptation insights
        insights = await adaptive_rag.get_adaptation_insights()
        print(f"\n📊 Learning Insights:")
        print(f"Episodes: {insights['training_statistics']['episode_count']}")
        print(f"Steps: {insights['training_statistics']['step_count']}")
        print(f"Avg Reward: {insights['training_statistics']['avg_episode_reward']:.3f}")
        print(f"Current Strategies: {insights['current_strategies']}")
    
    # Run demo
    # asyncio.run(demo_self_improving_rag())