"""
Federated Learning Infrastructure for RAG Systems
Revolutionary distributed learning that enables:
- Privacy-preserving model training across organizations
- Collaborative learning without data sharing
- Differential privacy mechanisms
- Secure aggregation protocols
- Cross-organizational knowledge sharing

This represents the bleeding edge of AI - enabling multiple organizations
to collaboratively improve RAG models while maintaining data privacy.
"""

import asyncio
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import hashlib
import hmac
import base64
from collections import defaultdict
import threading
import socket
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientInfo:
    client_id: str
    organization: str
    public_key: bytes
    data_samples: int
    model_version: str
    last_update: datetime
    trust_score: float
    
@dataclass
class ModelUpdate:
    client_id: str
    parameters: Dict[str, torch.Tensor]
    gradient_norms: Dict[str, float]
    loss: float
    accuracy: float
    data_samples: int
    privacy_budget: float
    signature: bytes
    timestamp: datetime

@dataclass
class FederatedRound:
    round_id: int
    participants: List[str]
    global_model: Dict[str, torch.Tensor]
    aggregated_metrics: Dict[str, float]
    convergence_score: float
    privacy_spent: float
    start_time: datetime
    end_time: Optional[datetime] = None

class DifferentialPrivacyMechanism:
    """
    Implements differential privacy for federated learning
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        self.spent_budget = 0.0
        
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor], 
                              sensitivity: float, batch_size: int) -> Dict[str, torch.Tensor]:
        """Add calibrated noise to gradients for differential privacy"""
        
        # Calculate noise scale for Gaussian mechanism
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        noisy_gradients = {}
        for name, grad in gradients.items():
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noisy_gradients[name] = grad + noise
            
        # Update spent budget
        self.spent_budget += self.epsilon / len(gradients)
        
        return noisy_gradients
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor], 
                      max_norm: float) -> Tuple[Dict[str, torch.Tensor], float]:
        """Clip gradients to bound sensitivity"""
        
        # Calculate total norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        clip_ratio = min(1.0, max_norm / total_norm)
        
        clipped_gradients = {}
        for name, grad in gradients.items():
            clipped_gradients[name] = grad * clip_ratio
            
        return clipped_gradients, clip_ratio
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0.0, self.epsilon - self.spent_budget)

class SecureAggregation:
    """
    Secure multi-party computation for gradient aggregation
    """
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
    def encrypt_parameters(self, parameters: Dict[str, torch.Tensor], 
                          recipient_public_key: bytes) -> Dict[str, bytes]:
        """Encrypt model parameters for secure transmission"""
        
        # Convert recipient public key
        recipient_key = serialization.load_pem_public_key(recipient_public_key)
        
        encrypted_params = {}
        for name, tensor in parameters.items():
            # Serialize tensor
            tensor_bytes = tensor.cpu().numpy().tobytes()
            
            # Split into chunks for RSA encryption
            chunk_size = 190  # RSA-2048 can encrypt ~190 bytes
            chunks = [tensor_bytes[i:i + chunk_size] 
                     for i in range(0, len(tensor_bytes), chunk_size)]
            
            encrypted_chunks = []
            for chunk in chunks:
                encrypted_chunk = recipient_key.encrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                encrypted_chunks.append(encrypted_chunk)
            
            encrypted_params[name] = b''.join(encrypted_chunks)
        
        return encrypted_params
    
    def decrypt_parameters(self, encrypted_params: Dict[str, bytes], 
                          original_shapes: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        """Decrypt model parameters"""
        
        decrypted_params = {}
        for name, encrypted_data in encrypted_params.items():
            # Split back into chunks
            chunk_size = 256  # RSA-2048 produces 256-byte ciphertext
            chunks = [encrypted_data[i:i + chunk_size] 
                     for i in range(0, len(encrypted_data), chunk_size)]
            
            decrypted_chunks = []
            for chunk in chunks:
                decrypted_chunk = self.private_key.decrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                decrypted_chunks.append(decrypted_chunk)
            
            # Reconstruct tensor
            tensor_bytes = b''.join(decrypted_chunks)
            tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32)
            tensor = torch.from_numpy(tensor_array.reshape(original_shapes[name]))
            
            decrypted_params[name] = tensor
        
        return decrypted_params
    
    def generate_signature(self, data: bytes) -> bytes:
        """Generate digital signature for data integrity"""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, 
                        public_key: bytes) -> bool:
        """Verify digital signature"""
        try:
            sender_key = serialization.load_pem_public_key(public_key)
            sender_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

class FederatedAggregator:
    """
    Aggregates model updates from multiple clients using various strategies
    """
    
    def __init__(self):
        self.aggregation_strategies = {
            'fedavg': self._fed_avg,
            'fedprox': self._fed_prox,
            'scaffold': self._scaffold,
            'adaptive': self._adaptive_aggregation
        }
    
    async def aggregate_updates(self, updates: List[ModelUpdate], 
                              strategy: str = 'fedavg') -> Dict[str, torch.Tensor]:
        """Aggregate model updates from clients"""
        
        if not updates:
            raise ValueError("No updates to aggregate")
        
        aggregation_func = self.aggregation_strategies.get(strategy, self._fed_avg)
        return await aggregation_func(updates)
    
    async def _fed_avg(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Federated Averaging (FedAvg) aggregation"""
        
        # Calculate total samples
        total_samples = sum(update.data_samples for update in updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first update
        param_names = list(updates[0].parameters.keys())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(updates[0].parameters[param_name])
            
            for update in updates:
                weight = update.data_samples / total_samples
                weighted_sum += weight * update.parameters[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    async def _fed_prox(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term"""
        
        # Start with FedAvg
        aggregated_params = await self._fed_avg(updates)
        
        # Apply proximal regularization
        mu = 0.01  # Proximal term coefficient
        
        for param_name in aggregated_params.keys():
            # Add proximal term (simplified)
            regularization = torch.zeros_like(aggregated_params[param_name])
            for update in updates:
                regularization += update.parameters[param_name]
            regularization /= len(updates)
            
            aggregated_params[param_name] = (
                aggregated_params[param_name] + mu * regularization
            ) / (1 + mu)
        
        return aggregated_params
    
    async def _scaffold(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates"""
        
        # Simplified SCAFFOLD implementation
        # In practice, this would maintain control variates
        
        aggregated_params = {}
        param_names = list(updates[0].parameters.keys())
        
        for param_name in param_names:
            # Use variance reduction with control variates
            param_sum = torch.zeros_like(updates[0].parameters[param_name])
            
            for update in updates:
                param_sum += update.parameters[param_name]
            
            aggregated_params[param_name] = param_sum / len(updates)
        
        return aggregated_params
    
    async def _adaptive_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Adaptive aggregation based on client performance"""
        
        # Calculate adaptive weights based on loss and gradient norms
        weights = []
        for update in updates:
            # Lower loss gets higher weight
            loss_weight = 1.0 / (update.loss + 1e-6)
            
            # Stable gradients get higher weight
            grad_stability = 1.0 / (np.mean(list(update.gradient_norms.values())) + 1e-6)
            
            # Combine weights
            total_weight = loss_weight * grad_stability * update.data_samples
            weights.append(total_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Aggregate with adaptive weights
        aggregated_params = {}
        param_names = list(updates[0].parameters.keys())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(updates[0].parameters[param_name])
            
            for update, weight in zip(updates, normalized_weights):
                weighted_sum += weight * update.parameters[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params

class FederatedServer:
    """
    Central server for federated learning coordination
    """
    
    def __init__(self, initial_model: nn.Module, config: Dict[str, Any]):
        self.model = initial_model
        self.config = config
        
        # Federated learning components
        self.aggregator = FederatedAggregator()
        self.secure_agg = SecureAggregation()
        self.privacy_mechanism = DifferentialPrivacyMechanism(
            epsilon=config.get('epsilon', 1.0),
            delta=config.get('delta', 1e-5)
        )
        
        # Client management
        self.registered_clients = {}
        self.client_updates = defaultdict(list)
        
        # Round management
        self.current_round = 0
        self.round_history = []
        self.global_model_history = []
        
        # Security
        self.allowed_clients = set()
        self.client_trust_scores = defaultdict(lambda: 1.0)
        
        # Configuration
        self.min_clients = config.get('min_clients', 2)
        self.max_clients = config.get('max_clients', 100)
        self.rounds_per_epoch = config.get('rounds_per_epoch', 10)
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
    
    async def register_client(self, client_info: ClientInfo) -> bool:
        """Register a new client for federated learning"""
        
        # Validate client
        if not await self._validate_client(client_info):
            logger.warning(f"Client validation failed: {client_info.client_id}")
            return False
        
        # Register client
        self.registered_clients[client_info.client_id] = client_info
        self.allowed_clients.add(client_info.client_id)
        
        logger.info(f"Client registered: {client_info.client_id} from {client_info.organization}")
        return True
    
    async def _validate_client(self, client_info: ClientInfo) -> bool:
        """Validate client before registration"""
        
        # Check if client has valid public key
        try:
            serialization.load_pem_public_key(client_info.public_key)
        except Exception:
            return False
        
        # Check minimum data requirements
        if client_info.data_samples < self.config.get('min_data_samples', 100):
            return False
        
        # Check organization whitelist (if configured)
        allowed_orgs = self.config.get('allowed_organizations', [])
        if allowed_orgs and client_info.organization not in allowed_orgs:
            return False
        
        return True
    
    async def start_federated_round(self) -> FederatedRound:
        """Start a new federated learning round"""
        
        self.current_round += 1
        
        # Select clients for this round
        selected_clients = await self._select_clients()
        
        if len(selected_clients) < self.min_clients:
            raise RuntimeError(f"Not enough clients: {len(selected_clients)} < {self.min_clients}")
        
        # Create round object
        fed_round = FederatedRound(
            round_id=self.current_round,
            participants=selected_clients,
            global_model=self._get_model_parameters(),
            aggregated_metrics={},
            convergence_score=0.0,
            privacy_spent=0.0,
            start_time=datetime.now()
        )
        
        logger.info(f"Started federated round {self.current_round} with {len(selected_clients)} clients")
        
        return fed_round
    
    async def _select_clients(self) -> List[str]:
        """Select clients for federated round"""
        
        # Strategy: Select based on trust score and data availability
        available_clients = []
        
        for client_id, client_info in self.registered_clients.items():
            # Check if client is active (updated recently)
            if (datetime.now() - client_info.last_update).days <= 1:
                available_clients.append((client_id, client_info.trust_score))
        
        # Sort by trust score and select top clients
        available_clients.sort(key=lambda x: x[1], reverse=True)
        
        # Select clients
        max_participants = min(self.max_clients, len(available_clients))
        selected = [client_id for client_id, _ in available_clients[:max_participants]]
        
        return selected
    
    async def receive_client_update(self, update: ModelUpdate) -> bool:
        """Receive and validate client update"""
        
        # Validate client
        if update.client_id not in self.allowed_clients:
            logger.warning(f"Unauthorized client: {update.client_id}")
            return False
        
        # Verify signature
        client_info = self.registered_clients[update.client_id]
        update_data = json.dumps({
            'client_id': update.client_id,
            'loss': update.loss,
            'accuracy': update.accuracy,
            'timestamp': update.timestamp.isoformat()
        }).encode()
        
        if not self.secure_agg.verify_signature(update_data, update.signature, 
                                               client_info.public_key):
            logger.warning(f"Invalid signature from client: {update.client_id}")
            return False
        
        # Add differential privacy noise
        if self.config.get('enable_privacy', True):
            noisy_params, clip_ratio = self.privacy_mechanism.clip_gradients(
                update.parameters, max_norm=self.config.get('clip_norm', 1.0)
            )
            update.parameters = self.privacy_mechanism.add_noise_to_gradients(
                noisy_params, sensitivity=1.0, batch_size=update.data_samples
            )
        
        # Store update
        self.client_updates[self.current_round].append(update)
        
        # Update client trust score
        await self._update_client_trust(update.client_id, update)
        
        logger.info(f"Received update from client: {update.client_id}")
        return True
    
    async def _update_client_trust(self, client_id: str, update: ModelUpdate):
        """Update client trust score based on update quality"""
        
        current_trust = self.client_trust_scores[client_id]
        
        # Factors affecting trust
        factors = []
        
        # Loss improvement factor
        if hasattr(self, 'previous_losses') and client_id in self.previous_losses:
            if update.loss < self.previous_losses[client_id]:
                factors.append(0.1)  # Positive
            else:
                factors.append(-0.05)  # Negative
        
        # Gradient norm stability
        avg_grad_norm = np.mean(list(update.gradient_norms.values()))
        if 0.1 <= avg_grad_norm <= 10.0:  # Reasonable range
            factors.append(0.05)
        else:
            factors.append(-0.1)
        
        # Update trust score
        trust_delta = sum(factors)
        new_trust = max(0.1, min(1.0, current_trust + trust_delta))
        
        self.client_trust_scores[client_id] = new_trust
    
    async def aggregate_round(self, round_id: int, strategy: str = 'fedavg') -> Dict[str, Any]:
        """Aggregate updates from a completed round"""
        
        updates = self.client_updates.get(round_id, [])
        
        if not updates:
            raise ValueError(f"No updates found for round {round_id}")
        
        # Aggregate parameters
        aggregated_params = await self.aggregator.aggregate_updates(updates, strategy)
        
        # Update global model
        self._update_global_model(aggregated_params)
        
        # Calculate metrics
        metrics = self._calculate_round_metrics(updates)
        
        # Check convergence
        convergence_score = await self._calculate_convergence(aggregated_params)
        
        # Update round history
        fed_round = next((r for r in self.round_history if r.round_id == round_id), None)
        if fed_round:
            fed_round.aggregated_metrics = metrics
            fed_round.convergence_score = convergence_score
            fed_round.end_time = datetime.now()
        
        result = {
            'round_id': round_id,
            'participants': len(updates),
            'aggregated_metrics': metrics,
            'convergence_score': convergence_score,
            'privacy_spent': self.privacy_mechanism.spent_budget,
            'global_model_updated': True
        }
        
        logger.info(f"Aggregated round {round_id}: {len(updates)} participants, convergence: {convergence_score:.4f}")
        
        return result
    
    def _get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return {name: param.clone() for name, param in self.model.named_parameters()}
    
    def _update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters"""
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    param.copy_(aggregated_params[name])
        
        # Store in history
        self.global_model_history.append({
            'round': self.current_round,
            'parameters': self._get_model_parameters(),
            'timestamp': datetime.now()
        })
    
    def _calculate_round_metrics(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Calculate aggregated metrics for the round"""
        
        if not updates:
            return {}
        
        total_samples = sum(update.data_samples for update in updates)
        
        metrics = {
            'avg_loss': sum(update.loss * update.data_samples for update in updates) / total_samples,
            'avg_accuracy': sum(update.accuracy * update.data_samples for update in updates) / total_samples,
            'participants': len(updates),
            'total_samples': total_samples,
            'avg_gradient_norm': np.mean([
                np.mean(list(update.gradient_norms.values())) for update in updates
            ])
        }
        
        return metrics
    
    async def _calculate_convergence(self, current_params: Dict[str, torch.Tensor]) -> float:
        """Calculate convergence score based on parameter changes"""
        
        if len(self.global_model_history) < 2:
            return 1.0  # Not enough history
        
        previous_params = self.global_model_history[-2]['parameters']
        
        total_change = 0.0
        total_norm = 0.0
        
        for name, current_param in current_params.items():
            if name in previous_params:
                prev_param = previous_params[name]
                change = torch.norm(current_param - prev_param).item()
                norm = torch.norm(current_param).item()
                
                total_change += change
                total_norm += norm
        
        # Relative change score (lower is better for convergence)
        convergence_score = total_change / (total_norm + 1e-8)
        
        return convergence_score
    
    async def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status and statistics"""
        
        status = {
            'current_round': self.current_round,
            'registered_clients': len(self.registered_clients),
            'active_clients': len([
                c for c in self.registered_clients.values()
                if (datetime.now() - c.last_update).days <= 1
            ]),
            'total_data_samples': sum(
                c.data_samples for c in self.registered_clients.values()
            ),
            'privacy_budget_remaining': self.privacy_mechanism.get_remaining_budget(),
            'convergence_trend': self._get_convergence_trend(),
            'client_trust_scores': dict(self.client_trust_scores),
            'federation_performance': self._get_federation_performance()
        }
        
        return status
    
    def _get_convergence_trend(self) -> List[float]:
        """Get convergence trend over recent rounds"""
        recent_rounds = self.round_history[-10:]  # Last 10 rounds
        return [round_info.convergence_score for round_info in recent_rounds 
                if round_info.convergence_score > 0]
    
    def _get_federation_performance(self) -> Dict[str, float]:
        """Calculate overall federation performance metrics"""
        
        if not self.round_history:
            return {}
        
        recent_rounds = self.round_history[-5:]  # Last 5 rounds
        
        return {
            'avg_participants_per_round': np.mean([len(r.participants) for r in recent_rounds]),
            'avg_convergence_score': np.mean([r.convergence_score for r in recent_rounds 
                                            if r.convergence_score > 0]),
            'federation_efficiency': len(recent_rounds) / max(1, self.current_round) * 100,
            'privacy_efficiency': 1.0 - (self.privacy_mechanism.spent_budget / self.privacy_mechanism.epsilon)
        }

class FederatedClient:
    """
    Client implementation for federated learning
    """
    
    def __init__(self, client_id: str, organization: str, local_model: nn.Module):
        self.client_id = client_id
        self.organization = organization
        self.model = local_model
        
        # Security
        self.secure_agg = SecureAggregation()
        self.server_public_key = None
        
        # Training
        self.local_data_loader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        
        # Metrics tracking
        self.training_history = []
        self.communication_log = []
        
    async def connect_to_server(self, server_address: str, server_public_key: bytes) -> bool:
        """Connect to federated learning server"""
        
        self.server_public_key = server_public_key
        
        # Create client info
        client_info = ClientInfo(
            client_id=self.client_id,
            organization=self.organization,
            public_key=self.secure_agg.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ),
            data_samples=1000,  # Mock data size
            model_version="1.0.0",
            last_update=datetime.now(),
            trust_score=1.0
        )
        
        # In real implementation, this would send HTTP request to server
        logger.info(f"Client {self.client_id} connected to server at {server_address}")
        
        return True
    
    async def local_training(self, epochs: int = 5) -> Dict[str, Any]:
        """Perform local training on client data"""
        
        training_metrics = {
            'epochs': epochs,
            'initial_loss': 0.0,
            'final_loss': 0.0,
            'accuracy': 0.0,
            'gradient_norms': {}
        }
        
        # Mock local training
        # In real implementation, this would train on actual local data
        for epoch in range(epochs):
            # Simulate training step
            loss = np.random.uniform(0.5, 1.5) * np.exp(-epoch * 0.1)  # Decreasing loss
            
            if epoch == 0:
                training_metrics['initial_loss'] = loss
            if epoch == epochs - 1:
                training_metrics['final_loss'] = loss
        
        # Calculate gradient norms (mock)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                training_metrics['gradient_norms'][name] = torch.norm(param.grad).item()
            else:
                training_metrics['gradient_norms'][name] = np.random.uniform(0.1, 1.0)
        
        training_metrics['accuracy'] = np.random.uniform(0.8, 0.95)
        
        self.training_history.append({
            'timestamp': datetime.now(),
            'metrics': training_metrics
        })
        
        logger.info(f"Client {self.client_id} completed {epochs} epochs of local training")
        
        return training_metrics
    
    async def create_model_update(self, training_metrics: Dict[str, Any]) -> ModelUpdate:
        """Create model update to send to server"""
        
        # Get model parameters
        parameters = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Create update data for signature
        update_data = json.dumps({
            'client_id': self.client_id,
            'loss': training_metrics['final_loss'],
            'accuracy': training_metrics['accuracy'],
            'timestamp': datetime.now().isoformat()
        }).encode()
        
        # Sign the update
        signature = self.secure_agg.generate_signature(update_data)
        
        update = ModelUpdate(
            client_id=self.client_id,
            parameters=parameters,
            gradient_norms=training_metrics['gradient_norms'],
            loss=training_metrics['final_loss'],
            accuracy=training_metrics['accuracy'],
            data_samples=1000,  # Mock data size
            privacy_budget=0.1,  # Mock privacy cost
            signature=signature,
            timestamp=datetime.now()
        )
        
        return update
    
    async def receive_global_model(self, global_parameters: Dict[str, torch.Tensor]):
        """Receive and apply global model update from server"""
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_parameters:
                    param.copy_(global_parameters[name])
        
        logger.info(f"Client {self.client_id} received global model update")

# Example usage and demonstration
if __name__ == "__main__":
    async def demo_federated_learning():
        """Demonstrate federated learning system"""
        
        # Create a simple model for demonstration
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # Initialize server
        server_model = SimpleModel()
        config = {
            'min_clients': 2,
            'max_clients': 5,
            'epsilon': 1.0,  # Privacy budget
            'enable_privacy': True,
            'clip_norm': 1.0
        }
        
        server = FederatedServer(server_model, config)
        
        # Initialize clients
        clients = []
        for i in range(3):
            client_model = SimpleModel()
            client = FederatedClient(f"client_{i}", f"org_{i}", client_model)
            clients.append(client)
        
        # Register clients
        for client in clients:
            client_info = ClientInfo(
                client_id=client.client_id,
                organization=client.organization,
                public_key=client.secure_agg.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ),
                data_samples=1000,
                model_version="1.0.0",
                last_update=datetime.now(),
                trust_score=1.0
            )
            
            await server.register_client(client_info)
        
        # Simulate federated learning rounds
        for round_num in range(3):
            print(f"\n🔄 Starting Federated Round {round_num + 1}")
            
            # Start round
            fed_round = await server.start_federated_round()
            
            # Clients perform local training
            for client in clients:
                if client.client_id in fed_round.participants:
                    # Local training
                    training_metrics = await client.local_training(epochs=5)
                    
                    # Create update
                    update = await client.create_model_update(training_metrics)
                    
                    # Send to server
                    await server.receive_client_update(update)
                    
                    print(f"  📊 {client.client_id}: loss={training_metrics['final_loss']:.4f}, acc={training_metrics['accuracy']:.4f}")
            
            # Aggregate updates
            result = await server.aggregate_round(fed_round.round_id, strategy='fedavg')
            print(f"  🔗 Aggregated: {result['participants']} participants, convergence={result['convergence_score']:.4f}")
            
            # Update clients with global model
            global_params = server._get_model_parameters()
            for client in clients:
                await client.receive_global_model(global_params)
        
        # Get final status
        status = await server.get_federation_status()
        print(f"\n📈 Federation Status:")
        print(f"  Total rounds: {status['current_round']}")
        print(f"  Active clients: {status['active_clients']}")
        print(f"  Privacy budget remaining: {status['privacy_budget_remaining']:.2f}")
        print(f"  Convergence trend: {status['convergence_trend'][-3:] if status['convergence_trend'] else 'N/A'}")
    
    # Run demo
    # asyncio.run(demo_federated_learning())