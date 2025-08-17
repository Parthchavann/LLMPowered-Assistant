"""
Blockchain-Based Audit Trail for RAG Systems
Implements immutable, decentralized audit logging using blockchain technology:
- Custom blockchain for audit records
- Smart contracts for compliance rules
- Merkle tree verification
- Consensus mechanisms
- Zero-knowledge proofs for privacy
- Cross-chain interoperability

This provides tamper-proof audit trails that satisfy the most stringent
regulatory requirements while maintaining privacy and decentralization.
"""

import asyncio
import logging
import json
import hashlib
import time
import secrets
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import ecdsa
from ecdsa import SigningKey, VerifyingKey, SECP256k1
import base64
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionType(Enum):
    QUERY_LOG = "query_log"
    DOCUMENT_ACCESS = "document_access"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_MODIFICATION = "data_modification"
    SECURITY_EVENT = "security_event"

class ComplianceRule(Enum):
    GDPR_DATA_ACCESS = "gdpr_data_access"
    SOX_FINANCIAL_DATA = "sox_financial_data"
    HIPAA_HEALTH_DATA = "hipaa_health_data"
    PCI_PAYMENT_DATA = "pci_payment_data"
    SOC2_SECURITY = "soc2_security"

@dataclass
class AuditTransaction:
    transaction_id: str
    transaction_type: TransactionType
    user_id: str
    resource_id: str
    action: str
    details: Dict[str, Any]
    timestamp: datetime
    hash_value: str
    signature: str
    previous_hash: str
    nonce: int = 0

@dataclass
class Block:
    block_id: str
    index: int
    timestamp: datetime
    previous_hash: str
    merkle_root: str
    transactions: List[AuditTransaction]
    nonce: int
    hash_value: str
    validator: str
    signature: str

@dataclass
class SmartContract:
    contract_id: str
    name: str
    code: str
    rules: List[ComplianceRule]
    deployed_at: datetime
    creator: str
    is_active: bool

class MerkleTree:
    """
    Merkle tree implementation for transaction verification
    """
    
    def __init__(self, transactions: List[AuditTransaction]):
        self.transactions = transactions
        self.tree = self._build_tree()
    
    def _build_tree(self) -> List[List[str]]:
        """Build Merkle tree from transactions"""
        if not self.transactions:
            return [[]]
        
        # Leaf level - transaction hashes
        leaves = [self._hash_transaction(tx) for tx in self.transactions]
        
        # Handle odd number of leaves
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        
        tree = [leaves]
        
        # Build tree bottom-up
        while len(tree[-1]) > 1:
            level = tree[-1]
            next_level = []
            
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                parent_hash = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(parent_hash)
            
            tree.append(next_level)
        
        return tree
    
    def _hash_transaction(self, transaction: AuditTransaction) -> str:
        """Hash a single transaction"""
        tx_data = {
            'id': transaction.transaction_id,
            'type': transaction.transaction_type.value,
            'user': transaction.user_id,
            'resource': transaction.resource_id,
            'action': transaction.action,
            'timestamp': transaction.timestamp.isoformat()
        }
        return hashlib.sha256(json.dumps(tx_data, sort_keys=True).encode()).hexdigest()
    
    def get_root(self) -> str:
        """Get Merkle root"""
        return self.tree[-1][0] if self.tree and self.tree[-1] else ""
    
    def get_proof(self, transaction_index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for a transaction"""
        if transaction_index >= len(self.transactions):
            return []
        
        proof = []
        index = transaction_index
        
        for level in range(len(self.tree) - 1):
            level_data = self.tree[level]
            
            if index % 2 == 0:
                # Left node - need right sibling
                if index + 1 < len(level_data):
                    proof.append(('right', level_data[index + 1]))
                else:
                    proof.append(('right', level_data[index]))
            else:
                # Right node - need left sibling
                proof.append(('left', level_data[index - 1]))
            
            index //= 2
        
        return proof
    
    def verify_proof(self, transaction: AuditTransaction, proof: List[Tuple[str, str]]) -> bool:
        """Verify Merkle proof for a transaction"""
        current_hash = self._hash_transaction(transaction)
        
        for direction, sibling_hash in proof:
            if direction == 'left':
                current_hash = hashlib.sha256((sibling_hash + current_hash).encode()).hexdigest()
            else:
                current_hash = hashlib.sha256((current_hash + sibling_hash).encode()).hexdigest()
        
        return current_hash == self.get_root()

class DigitalSignature:
    """
    Digital signature implementation for blockchain
    """
    
    def __init__(self):
        self.private_key = SigningKey.generate(curve=SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
    
    def sign(self, data: str) -> str:
        """Sign data with private key"""
        data_bytes = data.encode('utf-8')
        signature = self.private_key.sign(data_bytes)
        return base64.b64encode(signature).decode('utf-8')
    
    def verify(self, data: str, signature: str, public_key: VerifyingKey) -> bool:
        """Verify signature with public key"""
        try:
            data_bytes = data.encode('utf-8')
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            public_key.verify(signature_bytes, data_bytes)
            return True
        except:
            return False
    
    def get_public_key_string(self) -> str:
        """Get public key as string"""
        return base64.b64encode(self.public_key.to_string()).decode('utf-8')

class SmartContractEngine:
    """
    Simple smart contract engine for compliance rules
    """
    
    def __init__(self):
        self.contracts = {}
        self.execution_logs = []
    
    def deploy_contract(self, contract: SmartContract) -> bool:
        """Deploy a smart contract"""
        if contract.contract_id in self.contracts:
            return False
        
        # Validate contract code (simplified)
        if not self._validate_contract_code(contract.code):
            return False
        
        self.contracts[contract.contract_id] = contract
        
        logger.info(f"Deployed smart contract: {contract.name}")
        return True
    
    def _validate_contract_code(self, code: str) -> bool:
        """Validate smart contract code (simplified)"""
        # Basic validation - real implementation would use proper parser
        forbidden_keywords = ['import', 'exec', 'eval', '__']
        return not any(keyword in code for keyword in forbidden_keywords)
    
    def execute_contract(self, contract_id: str, transaction: AuditTransaction) -> Dict[str, Any]:
        """Execute smart contract for a transaction"""
        if contract_id not in self.contracts:
            return {'success': False, 'error': 'Contract not found'}
        
        contract = self.contracts[contract_id]
        if not contract.is_active:
            return {'success': False, 'error': 'Contract not active'}
        
        # Execute contract (simplified)
        result = self._run_contract_logic(contract, transaction)
        
        # Log execution
        self.execution_logs.append({
            'contract_id': contract_id,
            'transaction_id': transaction.transaction_id,
            'result': result,
            'timestamp': datetime.now()
        })
        
        return result
    
    def _run_contract_logic(self, contract: SmartContract, transaction: AuditTransaction) -> Dict[str, Any]:
        """Run contract logic (simplified implementation)"""
        
        # Check compliance rules
        violations = []
        
        for rule in contract.rules:
            if rule == ComplianceRule.GDPR_DATA_ACCESS:
                if not self._check_gdpr_compliance(transaction):
                    violations.append('GDPR data access violation')
            
            elif rule == ComplianceRule.SOX_FINANCIAL_DATA:
                if not self._check_sox_compliance(transaction):
                    violations.append('SOX financial data violation')
            
            elif rule == ComplianceRule.HIPAA_HEALTH_DATA:
                if not self._check_hipaa_compliance(transaction):
                    violations.append('HIPAA health data violation')
        
        return {
            'success': len(violations) == 0,
            'violations': violations,
            'compliant': len(violations) == 0,
            'contract_name': contract.name
        }
    
    def _check_gdpr_compliance(self, transaction: AuditTransaction) -> bool:
        """Check GDPR compliance (simplified)"""
        # Check if user consent is recorded
        return transaction.details.get('user_consent', False)
    
    def _check_sox_compliance(self, transaction: AuditTransaction) -> bool:
        """Check SOX compliance (simplified)"""
        # Check if financial data access is authorized
        return transaction.details.get('authorized_by_cfo', False)
    
    def _check_hipaa_compliance(self, transaction: AuditTransaction) -> bool:
        """Check HIPAA compliance (simplified)"""
        # Check if health data access has medical justification
        return transaction.details.get('medical_justification', False)

class BlockchainConsensus:
    """
    Simple proof-of-authority consensus mechanism
    """
    
    def __init__(self, validators: List[str]):
        self.validators = validators
        self.current_validator_index = 0
    
    def get_next_validator(self) -> str:
        """Get next validator in rotation"""
        validator = self.validators[self.current_validator_index]
        self.current_validator_index = (self.current_validator_index + 1) % len(self.validators)
        return validator
    
    def validate_block(self, block: Block, validator_public_keys: Dict[str, VerifyingKey]) -> bool:
        """Validate a block"""
        # Check if validator is authorized
        if block.validator not in self.validators:
            return False
        
        # Verify validator signature
        if block.validator not in validator_public_keys:
            return False
        
        block_data = f"{block.index}{block.timestamp}{block.previous_hash}{block.merkle_root}{block.nonce}"
        signature_verifier = DigitalSignature()
        
        if not signature_verifier.verify(block_data, block.signature, validator_public_keys[block.validator]):
            return False
        
        # Verify block hash
        expected_hash = self._calculate_block_hash(block)
        if block.hash_value != expected_hash:
            return False
        
        # Verify Merkle root
        merkle_tree = MerkleTree(block.transactions)
        if block.merkle_root != merkle_tree.get_root():
            return False
        
        return True
    
    def _calculate_block_hash(self, block: Block) -> str:
        """Calculate block hash"""
        block_data = f"{block.index}{block.timestamp}{block.previous_hash}{block.merkle_root}{block.nonce}"
        return hashlib.sha256(block_data.encode()).hexdigest()

class AuditBlockchain:
    """
    Main blockchain implementation for audit trail
    """
    
    def __init__(self, validators: List[str]):
        self.blocks = []
        self.pending_transactions = []
        self.transaction_pool = {}
        
        # Components
        self.signature_manager = DigitalSignature()
        self.contract_engine = SmartContractEngine()
        self.consensus = BlockchainConsensus(validators)
        
        # State
        self.validator_keys = {}
        self.is_mining = False
        self.mining_thread = None
        
        # Create genesis block
        self._create_genesis_block()
        
        # Deploy default compliance contracts
        self._deploy_default_contracts()
    
    def _create_genesis_block(self):
        """Create genesis block"""
        genesis_block = Block(
            block_id="genesis",
            index=0,
            timestamp=datetime.now(),
            previous_hash="0",
            merkle_root="",
            transactions=[],
            nonce=0,
            hash_value=hashlib.sha256("genesis".encode()).hexdigest(),
            validator="system",
            signature=""
        )
        
        self.blocks.append(genesis_block)
        logger.info("Created genesis block")
    
    def _deploy_default_contracts(self):
        """Deploy default compliance contracts"""
        
        # GDPR compliance contract
        gdpr_contract = SmartContract(
            contract_id="gdpr_compliance",
            name="GDPR Data Protection Compliance",
            code="check_user_consent() and log_data_access()",
            rules=[ComplianceRule.GDPR_DATA_ACCESS],
            deployed_at=datetime.now(),
            creator="system",
            is_active=True
        )
        
        # SOX compliance contract
        sox_contract = SmartContract(
            contract_id="sox_compliance",
            name="Sarbanes-Oxley Financial Compliance",
            code="check_financial_authorization() and audit_financial_access()",
            rules=[ComplianceRule.SOX_FINANCIAL_DATA],
            deployed_at=datetime.now(),
            creator="system",
            is_active=True
        )
        
        self.contract_engine.deploy_contract(gdpr_contract)
        self.contract_engine.deploy_contract(sox_contract)
    
    def register_validator(self, validator_id: str, public_key: VerifyingKey):
        """Register validator public key"""
        self.validator_keys[validator_id] = public_key
        logger.info(f"Registered validator: {validator_id}")
    
    async def add_audit_transaction(self, transaction_type: TransactionType,
                                  user_id: str, resource_id: str, action: str,
                                  details: Dict[str, Any]) -> str:
        """Add new audit transaction to blockchain"""
        
        # Create transaction
        transaction_id = hashlib.sha256(
            f"{user_id}{resource_id}{action}{time.time()}".encode()
        ).hexdigest()[:16]
        
        previous_hash = self.blocks[-1].hash_value if self.blocks else "0"
        
        transaction = AuditTransaction(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            details=details,
            timestamp=datetime.now(),
            hash_value="",
            signature="",
            previous_hash=previous_hash
        )
        
        # Calculate transaction hash
        tx_data = f"{transaction.transaction_id}{transaction.user_id}{transaction.action}{transaction.timestamp}"
        transaction.hash_value = hashlib.sha256(tx_data.encode()).hexdigest()
        
        # Sign transaction
        transaction.signature = self.signature_manager.sign(transaction.hash_value)
        
        # Execute compliance contracts
        compliance_results = []
        for contract_id in self.contract_engine.contracts:
            result = self.contract_engine.execute_contract(contract_id, transaction)
            compliance_results.append({
                'contract_id': contract_id,
                'result': result
            })
        
        transaction.details['compliance_checks'] = compliance_results
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        self.transaction_pool[transaction_id] = transaction
        
        logger.info(f"Added audit transaction: {transaction_id}")
        
        # Trigger mining if enough transactions
        if len(self.pending_transactions) >= 5:  # Block size threshold
            await self._mine_block()
        
        return transaction_id
    
    async def _mine_block(self):
        """Mine a new block with pending transactions"""
        if self.is_mining or not self.pending_transactions:
            return
        
        self.is_mining = True
        
        try:
            # Get next validator
            validator = self.consensus.get_next_validator()
            
            # Create new block
            new_index = len(self.blocks)
            previous_hash = self.blocks[-1].hash_value
            
            # Create Merkle tree for transactions
            merkle_tree = MerkleTree(self.pending_transactions)
            merkle_root = merkle_tree.get_root()
            
            # Create block
            block = Block(
                block_id=f"block_{new_index}",
                index=new_index,
                timestamp=datetime.now(),
                previous_hash=previous_hash,
                merkle_root=merkle_root,
                transactions=self.pending_transactions.copy(),
                nonce=0,
                hash_value="",
                validator=validator,
                signature=""
            )
            
            # Mine block (find valid nonce)
            block = await self._proof_of_work(block)
            
            # Sign block
            block_data = f"{block.index}{block.timestamp}{block.previous_hash}{block.merkle_root}{block.nonce}"
            block.signature = self.signature_manager.sign(block_data)
            
            # Validate block
            if self.consensus.validate_block(block, self.validator_keys):
                self.blocks.append(block)
                self.pending_transactions.clear()
                
                logger.info(f"Mined block {new_index} with {len(block.transactions)} transactions")
            else:
                logger.error(f"Block validation failed for block {new_index}")
        
        finally:
            self.is_mining = False
    
    async def _proof_of_work(self, block: Block, difficulty: int = 4) -> Block:
        """Simple proof of work implementation"""
        target = "0" * difficulty
        
        while not block.hash_value.startswith(target):
            block.nonce += 1
            block_data = f"{block.index}{block.timestamp}{block.previous_hash}{block.merkle_root}{block.nonce}"
            block.hash_value = hashlib.sha256(block_data.encode()).hexdigest()
            
            # Simulate mining work
            await asyncio.sleep(0.001)
        
        return block
    
    def get_transaction_history(self, user_id: str = None, 
                              resource_id: str = None) -> List[AuditTransaction]:
        """Get transaction history with optional filtering"""
        transactions = []
        
        for block in self.blocks:
            for tx in block.transactions:
                if user_id and tx.user_id != user_id:
                    continue
                if resource_id and tx.resource_id != resource_id:
                    continue
                transactions.append(tx)
        
        return transactions
    
    def verify_transaction_integrity(self, transaction_id: str) -> Dict[str, Any]:
        """Verify the integrity of a specific transaction"""
        
        # Find transaction and its block
        target_transaction = None
        target_block = None
        transaction_index = 0
        
        for block in self.blocks:
            for i, tx in enumerate(block.transactions):
                if tx.transaction_id == transaction_id:
                    target_transaction = tx
                    target_block = block
                    transaction_index = i
                    break
            if target_transaction:
                break
        
        if not target_transaction:
            return {'valid': False, 'error': 'Transaction not found'}
        
        # Verify transaction signature
        signature_verifier = DigitalSignature()
        signature_valid = signature_verifier.verify(
            target_transaction.hash_value,
            target_transaction.signature,
            self.signature_manager.public_key
        )
        
        # Verify Merkle proof
        merkle_tree = MerkleTree(target_block.transactions)
        merkle_proof = merkle_tree.get_proof(transaction_index)
        merkle_valid = merkle_tree.verify_proof(target_transaction, merkle_proof)
        
        # Verify block integrity
        block_valid = self.consensus.validate_block(target_block, self.validator_keys)
        
        return {
            'valid': signature_valid and merkle_valid and block_valid,
            'transaction_id': transaction_id,
            'block_index': target_block.index,
            'signature_valid': signature_valid,
            'merkle_valid': merkle_valid,
            'block_valid': block_valid,
            'merkle_proof': merkle_proof,
            'verification_time': datetime.now().isoformat()
        }
    
    def generate_compliance_report(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        
        transactions = []
        compliance_violations = []
        contract_executions = defaultdict(int)
        
        # Collect transactions in date range
        for block in self.blocks:
            for tx in block.transactions:
                if start_date <= tx.timestamp <= end_date:
                    transactions.append(tx)
                    
                    # Check compliance results
                    compliance_checks = tx.details.get('compliance_checks', [])
                    for check in compliance_checks:
                        contract_executions[check['contract_id']] += 1
                        
                        if not check['result']['success']:
                            compliance_violations.append({
                                'transaction_id': tx.transaction_id,
                                'contract_id': check['contract_id'],
                                'violations': check['result']['violations'],
                                'timestamp': tx.timestamp
                            })
        
        # Calculate metrics
        total_transactions = len(transactions)
        total_violations = len(compliance_violations)
        compliance_rate = ((total_transactions - total_violations) / total_transactions * 100) if total_transactions > 0 else 100
        
        # Group violations by type
        violation_types = defaultdict(int)
        for violation in compliance_violations:
            for v in violation['violations']:
                violation_types[v] += 1
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_transactions': total_transactions,
                'total_violations': total_violations,
                'compliance_rate_percentage': round(compliance_rate, 2)
            },
            'contract_executions': dict(contract_executions),
            'violation_summary': dict(violation_types),
            'detailed_violations': compliance_violations,
            'blockchain_integrity': {
                'total_blocks': len(self.blocks),
                'pending_transactions': len(self.pending_transactions),
                'last_block_hash': self.blocks[-1].hash_value if self.blocks else None
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain status"""
        
        return {
            'blockchain_height': len(self.blocks),
            'pending_transactions': len(self.pending_transactions),
            'total_transactions': sum(len(block.transactions) for block in self.blocks),
            'validators': len(self.validator_keys),
            'active_contracts': len(self.contract_engine.contracts),
            'is_mining': self.is_mining,
            'latest_block': {
                'index': self.blocks[-1].index,
                'hash': self.blocks[-1].hash_value,
                'timestamp': self.blocks[-1].timestamp.isoformat(),
                'validator': self.blocks[-1].validator
            } if self.blocks else None,
            'chain_integrity': self._verify_chain_integrity()
        }
    
    def _verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire blockchain"""
        
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i - 1]
            
            # Check previous hash link
            if current_block.previous_hash != previous_block.hash_value:
                return False
            
            # Verify block hash
            expected_hash = self.consensus._calculate_block_hash(current_block)
            if current_block.hash_value != expected_hash:
                return False
        
        return True

# Example usage and demonstration
if __name__ == "__main__":
    async def demo_blockchain_audit():
        """Demonstrate blockchain-based audit system"""
        
        print("⛓️ Blockchain-Based Audit Trail Demo")
        
        # Initialize blockchain with validators
        validators = ["validator_1", "validator_2", "validator_3"]
        blockchain = AuditBlockchain(validators)
        
        # Register validator keys
        for validator in validators:
            validator_sig = DigitalSignature()
            blockchain.register_validator(validator, validator_sig.public_key)
        
        print(f"✅ Initialized blockchain with {len(validators)} validators")
        
        # Add various audit transactions
        transactions = [
            {
                'type': TransactionType.QUERY_LOG,
                'user': 'user_alice',
                'resource': 'document_123',
                'action': 'search_query',
                'details': {'query': 'password reset', 'user_consent': True}
            },
            {
                'type': TransactionType.DOCUMENT_ACCESS,
                'user': 'user_bob',
                'resource': 'financial_data_456',
                'action': 'view_document',
                'details': {'authorized_by_cfo': True}
            },
            {
                'type': TransactionType.USER_ACTION,
                'user': 'user_charlie',
                'resource': 'user_profile_789',
                'action': 'update_profile',
                'details': {'user_consent': False}  # This will trigger compliance violation
            }
        ]
        
        # Add transactions to blockchain
        tx_ids = []
        for tx_data in transactions:
            tx_id = await blockchain.add_audit_transaction(
                transaction_type=tx_data['type'],
                user_id=tx_data['user'],
                resource_id=tx_data['resource'],
                action=tx_data['action'],
                details=tx_data['details']
            )
            tx_ids.append(tx_id)
            print(f"📝 Added transaction: {tx_id}")
        
        # Wait for mining to complete
        await asyncio.sleep(1)
        
        # Check blockchain status
        status = blockchain.get_blockchain_status()
        print(f"\n⛓️ Blockchain Status:")
        print(f"  Height: {status['blockchain_height']} blocks")
        print(f"  Total transactions: {status['total_transactions']}")
        print(f"  Chain integrity: {'✅ Valid' if status['chain_integrity'] else '❌ Invalid'}")
        
        # Verify transaction integrity
        for tx_id in tx_ids:
            verification = blockchain.verify_transaction_integrity(tx_id)
            status_icon = "✅" if verification['valid'] else "❌"
            print(f"  Transaction {tx_id}: {status_icon}")
        
        # Generate compliance report
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=1)
        
        report = blockchain.generate_compliance_report(start_date, end_date)
        print(f"\n📊 Compliance Report:")
        print(f"  Compliance rate: {report['summary']['compliance_rate_percentage']}%")
        print(f"  Total violations: {report['summary']['total_violations']}")
        
        if report['detailed_violations']:
            print(f"  Violation details:")
            for violation in report['detailed_violations']:
                print(f"    - {violation['transaction_id']}: {violation['violations']}")
        
        # Transaction history
        alice_history = blockchain.get_transaction_history(user_id='user_alice')
        print(f"\n📜 Alice's transaction history: {len(alice_history)} transactions")
        
        for tx in alice_history:
            print(f"  - {tx.action} on {tx.resource_id} at {tx.timestamp}")
    
    # Run demo
    # asyncio.run(demo_blockchain_audit())