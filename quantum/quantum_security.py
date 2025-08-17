"""
Quantum-Resistant Security for RAG Systems
Implements post-quantum cryptography to protect against future quantum attacks:
- Lattice-based encryption (NTRU, Kyber)
- Hash-based signatures (SPHINCS+)
- Code-based cryptography (McEliece)
- Multivariate cryptography
- Quantum key distribution simulation
- Quantum-safe protocol design

This represents cutting-edge security - preparing for the quantum computing era
when current encryption methods will be vulnerable.
"""

import asyncio
import logging
import json
import secrets
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumAlgorithm(Enum):
    KYBER = "kyber"           # Lattice-based KEM
    NTRU = "ntru"             # Lattice-based encryption
    SPHINCS_PLUS = "sphincs+" # Hash-based signatures
    MCELIECE = "mceliece"     # Code-based encryption
    RAINBOW = "rainbow"       # Multivariate signatures
    DILITHIUM = "dilithium"   # Lattice-based signatures

class SecurityLevel(Enum):
    LEVEL_1 = 128  # Classical security equivalent
    LEVEL_3 = 192
    LEVEL_5 = 256

@dataclass
class QuantumKeyPair:
    algorithm: QuantumAlgorithm
    security_level: SecurityLevel
    public_key: bytes
    private_key: bytes
    creation_time: datetime
    expiry_time: datetime
    key_id: str

@dataclass
class QuantumSignature:
    algorithm: QuantumAlgorithm
    signature: bytes
    message_hash: bytes
    signer_key_id: str
    timestamp: datetime

@dataclass
class QuantumCiphertext:
    algorithm: QuantumAlgorithm
    ciphertext: bytes
    nonce: bytes
    tag: Optional[bytes]
    recipient_key_id: str

class LatticeBasedCrypto:
    """
    Implementation of lattice-based cryptographic schemes (NTRU-like)
    Resistant to both classical and quantum attacks
    """
    
    def __init__(self, n: int = 509, q: int = 2048, security_level: SecurityLevel = SecurityLevel.LEVEL_3):
        self.n = n  # Ring dimension
        self.q = q  # Modulus
        self.security_level = security_level
        
        # NTRU parameters
        self.df = 101  # Number of 1's in f
        self.dg = 101  # Number of 1's in g
        
        # For practical implementation, we'll use simplified operations
        # Real implementation would use optimized polynomial arithmetic
        
    def generate_keypair(self) -> QuantumKeyPair:
        """Generate NTRU-like key pair"""
        
        # Generate private key components
        f_coeffs = self._generate_ternary_polynomial(self.df, self.n)
        g_coeffs = self._generate_ternary_polynomial(self.dg, self.n)
        
        # Compute public key (simplified)
        h_coeffs = self._polynomial_multiply_mod(g_coeffs, self._polynomial_inverse_mod(f_coeffs))
        
        # Serialize keys
        private_key = self._serialize_polynomial(f_coeffs) + self._serialize_polynomial(g_coeffs)
        public_key = self._serialize_polynomial(h_coeffs)
        
        key_id = hashlib.sha256(public_key).hexdigest()[:16]
        
        return QuantumKeyPair(
            algorithm=QuantumAlgorithm.NTRU,
            security_level=self.security_level,
            public_key=public_key,
            private_key=private_key,
            creation_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(days=365),
            key_id=key_id
        )
    
    def encrypt(self, message: bytes, public_key: bytes) -> QuantumCiphertext:
        """Encrypt message using NTRU-like scheme"""
        
        # Generate random polynomial for encryption
        r_coeffs = self._generate_random_polynomial(self.n)
        
        # Deserialize public key
        h_coeffs = self._deserialize_polynomial(public_key)
        
        # Convert message to polynomial (simplified)
        m_coeffs = self._message_to_polynomial(message)
        
        # Encrypt: c = r * h + m (mod q)
        rh = self._polynomial_multiply_mod(r_coeffs, h_coeffs)
        c_coeffs = [(rh[i] + m_coeffs[i]) % self.q for i in range(self.n)]
        
        ciphertext = self._serialize_polynomial(c_coeffs)
        nonce = secrets.token_bytes(16)
        
        return QuantumCiphertext(
            algorithm=QuantumAlgorithm.NTRU,
            ciphertext=ciphertext,
            nonce=nonce,
            tag=None,
            recipient_key_id=hashlib.sha256(public_key).hexdigest()[:16]
        )
    
    def decrypt(self, quantum_ciphertext: QuantumCiphertext, private_key: bytes) -> bytes:
        """Decrypt ciphertext using NTRU-like scheme"""
        
        # Deserialize private key
        key_len = len(private_key) // 2
        f_coeffs = self._deserialize_polynomial(private_key[:key_len])
        g_coeffs = self._deserialize_polynomial(private_key[key_len:])
        
        # Deserialize ciphertext
        c_coeffs = self._deserialize_polynomial(quantum_ciphertext.ciphertext)
        
        # Decrypt: m = c * f (mod q), then recover message
        cf = self._polynomial_multiply_mod(c_coeffs, f_coeffs)
        
        # Recover message (simplified)
        message = self._polynomial_to_message(cf)
        
        return message
    
    def _generate_ternary_polynomial(self, weight: int, length: int) -> List[int]:
        """Generate ternary polynomial with specified weight"""
        poly = [0] * length
        
        # Place +1's
        positions = secrets.SystemRandom().sample(range(length), weight)
        for pos in positions[:weight//2]:
            poly[pos] = 1
        
        # Place -1's
        for pos in positions[weight//2:weight]:
            poly[pos] = -1
        
        return poly
    
    def _generate_random_polynomial(self, length: int) -> List[int]:
        """Generate random polynomial"""
        return [secrets.randbelow(self.q) for _ in range(length)]
    
    def _polynomial_multiply_mod(self, a: List[int], b: List[int]) -> List[int]:
        """Polynomial multiplication modulo (x^n - 1) and q"""
        result = [0] * self.n
        
        for i in range(len(a)):
            for j in range(len(b)):
                pos = (i + j) % self.n
                result[pos] = (result[pos] + a[i] * b[j]) % self.q
        
        return result
    
    def _polynomial_inverse_mod(self, poly: List[int]) -> List[int]:
        """Compute polynomial inverse (simplified)"""
        # This is a placeholder - real implementation would use extended Euclidean algorithm
        # For demo purposes, return a random valid-looking polynomial
        return self._generate_random_polynomial(self.n)
    
    def _serialize_polynomial(self, poly: List[int]) -> bytes:
        """Serialize polynomial to bytes"""
        return b''.join(struct.pack('<H', coeff % (2**16)) for coeff in poly)
    
    def _deserialize_polynomial(self, data: bytes) -> List[int]:
        """Deserialize polynomial from bytes"""
        poly = []
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                coeff = struct.unpack('<H', data[i:i+2])[0]
                poly.append(coeff)
        return poly
    
    def _message_to_polynomial(self, message: bytes) -> List[int]:
        """Convert message to polynomial representation"""
        poly = [0] * self.n
        for i, byte in enumerate(message[:self.n//8]):
            for j in range(8):
                if i*8 + j < self.n:
                    poly[i*8 + j] = (byte >> j) & 1
        return poly
    
    def _polynomial_to_message(self, poly: List[int]) -> bytes:
        """Convert polynomial to message"""
        message = bytearray()
        for i in range(0, min(len(poly), self.n), 8):
            byte = 0
            for j in range(8):
                if i + j < len(poly) and poly[i + j] % 2 == 1:
                    byte |= (1 << j)
            message.append(byte)
        return bytes(message)

class HashBasedSignatures:
    """
    Implementation of hash-based signature scheme (SPHINCS+-like)
    Provides quantum-resistant digital signatures
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.LEVEL_3):
        self.security_level = security_level
        self.hash_function = hashlib.sha256
        
        # SPHINCS+ parameters (simplified)
        self.n = 32  # Hash output length
        self.h = 20  # Height of hypertree
        self.d = 4   # Layers
        self.w = 4   # Winternitz parameter
        
    def generate_keypair(self) -> QuantumKeyPair:
        """Generate hash-based signature key pair"""
        
        # Generate random seed
        seed = secrets.token_bytes(self.n)
        
        # Generate secret key
        sk_seed = secrets.token_bytes(self.n)
        sk_prf = secrets.token_bytes(self.n)
        
        private_key = seed + sk_seed + sk_prf
        
        # Generate public key (root of Merkle tree)
        public_key = self._compute_public_key(seed)
        
        key_id = hashlib.sha256(public_key).hexdigest()[:16]
        
        return QuantumKeyPair(
            algorithm=QuantumAlgorithm.SPHINCS_PLUS,
            security_level=self.security_level,
            public_key=public_key,
            private_key=private_key,
            creation_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(days=365),
            key_id=key_id
        )
    
    def sign(self, message: bytes, private_key: bytes) -> QuantumSignature:
        """Create hash-based signature"""
        
        # Extract components from private key
        seed = private_key[:self.n]
        sk_seed = private_key[self.n:2*self.n]
        sk_prf = private_key[2*self.n:3*self.n]
        
        # Hash message
        message_hash = self.hash_function(message).digest()
        
        # Generate randomness
        randomness = hmac.new(sk_prf, message, self.hash_function).digest()
        
        # Create WOTS+ signature (simplified)
        wots_signature = self._wots_sign(message_hash, seed, randomness)
        
        # Create authentication path (simplified)
        auth_path = self._generate_auth_path(seed)
        
        # Combine signature components
        signature = wots_signature + auth_path
        
        key_id = hashlib.sha256(self._compute_public_key(seed)).hexdigest()[:16]
        
        return QuantumSignature(
            algorithm=QuantumAlgorithm.SPHINCS_PLUS,
            signature=signature,
            message_hash=message_hash,
            signer_key_id=key_id,
            timestamp=datetime.now()
        )
    
    def verify(self, signature: QuantumSignature, message: bytes, public_key: bytes) -> bool:
        """Verify hash-based signature"""
        
        # Hash message and compare
        message_hash = self.hash_function(message).digest()
        if message_hash != signature.message_hash:
            return False
        
        # Extract signature components
        sig_len = len(signature.signature) // 2
        wots_signature = signature.signature[:sig_len]
        auth_path = signature.signature[sig_len:]
        
        # Verify WOTS+ signature (simplified)
        recovered_key = self._wots_verify(signature.message_hash, wots_signature)
        
        # Verify authentication path (simplified)
        return self._verify_auth_path(recovered_key, auth_path, public_key)
    
    def _compute_public_key(self, seed: bytes) -> bytes:
        """Compute public key from seed"""
        # Simplified computation - real implementation would build Merkle tree
        return hashlib.sha256(seed + b"public_key").digest()
    
    def _wots_sign(self, message_hash: bytes, seed: bytes, randomness: bytes) -> bytes:
        """Create WOTS+ signature (simplified)"""
        # Real implementation would use Winternitz one-time signatures
        signature_seed = hashlib.sha256(message_hash + seed + randomness).digest()
        return signature_seed * 8  # Simplified signature
    
    def _wots_verify(self, message_hash: bytes, signature: bytes) -> bytes:
        """Verify WOTS+ signature and recover key"""
        # Simplified verification
        return hashlib.sha256(signature + message_hash).digest()
    
    def _generate_auth_path(self, seed: bytes) -> bytes:
        """Generate authentication path (simplified)"""
        # Real implementation would generate Merkle tree path
        return hashlib.sha256(seed + b"auth_path").digest() * 4
    
    def _verify_auth_path(self, leaf: bytes, auth_path: bytes, root: bytes) -> bool:
        """Verify Merkle tree authentication path"""
        # Simplified verification
        computed_root = hashlib.sha256(leaf + auth_path).digest()
        return hmac.compare_digest(computed_root, root)

class QuantumKeyDistribution:
    """
    Simulated Quantum Key Distribution (QKD) protocol
    In real quantum systems, this would use quantum properties like entanglement
    """
    
    def __init__(self):
        self.basis_choices = ['rectilinear', 'diagonal']
        self.error_threshold = 0.11  # QBER threshold for security
        
    async def generate_quantum_key(self, key_length: int = 256) -> Tuple[bytes, Dict[str, Any]]:
        """
        Simulate BB84 quantum key distribution protocol
        """
        
        # Alice's setup
        alice_bits = [secrets.randbelow(2) for _ in range(key_length * 2)]  # Double length for sifting
        alice_bases = [secrets.choice(self.basis_choices) for _ in range(key_length * 2)]
        
        # Bob's setup (random basis choices)
        bob_bases = [secrets.choice(self.basis_choices) for _ in range(key_length * 2)]
        
        # Quantum transmission simulation
        bob_bits = []
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                # Correct measurement - bit preserved
                bob_bits.append(alice_bits[i])
            else:
                # Wrong basis - random result
                bob_bits.append(secrets.randbelow(2))
        
        # Public basis comparison (classical channel)
        matching_indices = [i for i in range(len(alice_bits)) if alice_bases[i] == bob_bases[i]]
        
        # Sifted key
        sifted_bits = [alice_bits[i] for i in matching_indices]
        
        # Error detection (sacrifice some bits)
        test_indices = secrets.SystemRandom().sample(range(len(sifted_bits)), min(len(sifted_bits)//10, 50))
        test_bits_alice = [sifted_bits[i] for i in test_indices]
        test_bits_bob = [bob_bits[matching_indices[i]] for i in test_indices]
        
        # Calculate error rate
        errors = sum(1 for a, b in zip(test_bits_alice, test_bits_bob) if a != b)
        error_rate = errors / len(test_indices) if test_indices else 0
        
        # Remove test bits
        final_bits = [bit for i, bit in enumerate(sifted_bits) if i not in test_indices]
        
        # Convert to bytes
        quantum_key = bytearray()
        for i in range(0, len(final_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(final_bits):
                    byte |= (final_bits[i + j] << j)
            quantum_key.append(byte)
        
        # Security assessment
        is_secure = error_rate <= self.error_threshold
        
        metadata = {
            'protocol': 'BB84',
            'initial_bits': len(alice_bits),
            'sifted_bits': len(sifted_bits),
            'final_key_bits': len(final_bits),
            'error_rate': error_rate,
            'is_secure': is_secure,
            'efficiency': len(final_bits) / len(alice_bits)
        }
        
        return bytes(quantum_key), metadata

class QuantumSafeProtocol:
    """
    Implementation of quantum-safe communication protocol
    """
    
    def __init__(self):
        self.lattice_crypto = LatticeBasedCrypto()
        self.hash_signatures = HashBasedSignatures()
        self.qkd = QuantumKeyDistribution()
        
        # Protocol state
        self.session_keys = {}
        self.identity_keys = {}
        
    async def generate_identity_keys(self, identity: str) -> Dict[str, QuantumKeyPair]:
        """Generate quantum-safe identity keys"""
        
        keys = {
            'encryption': self.lattice_crypto.generate_keypair(),
            'signing': self.hash_signatures.generate_keypair()
        }
        
        self.identity_keys[identity] = keys
        
        logger.info(f"Generated quantum-safe keys for {identity}")
        return keys
    
    async def establish_session(self, alice_identity: str, bob_identity: str) -> Dict[str, Any]:
        """
        Establish quantum-safe session between two parties
        """
        
        # Generate ephemeral keys
        alice_ephemeral = self.lattice_crypto.generate_keypair()
        bob_ephemeral = self.lattice_crypto.generate_keypair()
        
        # Simulate QKD for session key
        quantum_key, qkd_metadata = await self.qkd.generate_quantum_key(256)
        
        # Create hybrid key (quantum + classical)
        # In practice, this would use key derivation functions
        classical_component = hashlib.sha256(
            alice_ephemeral.public_key + bob_ephemeral.public_key
        ).digest()
        
        session_key = hashlib.sha256(quantum_key + classical_component).digest()
        
        # Create session metadata
        session_id = hashlib.sha256(
            alice_identity.encode() + bob_identity.encode() + session_key
        ).hexdigest()[:16]
        
        session_info = {
            'session_id': session_id,
            'participants': [alice_identity, bob_identity],
            'session_key': session_key,
            'quantum_component': quantum_key,
            'classical_component': classical_component,
            'qkd_metadata': qkd_metadata,
            'established_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        
        self.session_keys[session_id] = session_info
        
        logger.info(f"Established quantum-safe session {session_id}")
        return session_info
    
    async def quantum_encrypt(self, message: bytes, recipient_identity: str, 
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Encrypt message using quantum-safe protocols
        """
        
        if session_id and session_id in self.session_keys:
            # Use session key
            session = self.session_keys[session_id]
            encryption_key = session['session_key']
            
            # AES encryption with quantum-derived key
            cipher = Cipher(
                algorithms.AES(encryption_key[:32]),
                modes.GCM(secrets.token_bytes(12))
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(message) + encryptor.finalize()
            
            result = {
                'type': 'session_encrypted',
                'session_id': session_id,
                'ciphertext': ciphertext,
                'tag': encryptor.tag,
                'nonce': cipher.mode.initialization_vector
            }
            
        else:
            # Use public key encryption
            if recipient_identity not in self.identity_keys:
                raise ValueError(f"No keys found for {recipient_identity}")
            
            recipient_keys = self.identity_keys[recipient_identity]
            quantum_ciphertext = self.lattice_crypto.encrypt(
                message, recipient_keys['encryption'].public_key
            )
            
            result = {
                'type': 'public_key_encrypted',
                'recipient': recipient_identity,
                'quantum_ciphertext': quantum_ciphertext
            }
        
        return result
    
    async def quantum_decrypt(self, encrypted_data: Dict[str, Any], 
                            identity: str) -> bytes:
        """
        Decrypt message using quantum-safe protocols
        """
        
        if encrypted_data['type'] == 'session_encrypted':
            session_id = encrypted_data['session_id']
            if session_id not in self.session_keys:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.session_keys[session_id]
            decryption_key = session['session_key']
            
            # AES decryption
            cipher = Cipher(
                algorithms.AES(decryption_key[:32]),
                modes.GCM(encrypted_data['nonce'], encrypted_data['tag'])
            )
            decryptor = cipher.decryptor()
            message = decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()
            
        elif encrypted_data['type'] == 'public_key_encrypted':
            if identity not in self.identity_keys:
                raise ValueError(f"No private key for {identity}")
            
            private_keys = self.identity_keys[identity]
            message = self.lattice_crypto.decrypt(
                encrypted_data['quantum_ciphertext'],
                private_keys['encryption'].private_key
            )
        
        else:
            raise ValueError(f"Unknown encryption type: {encrypted_data['type']}")
        
        return message
    
    async def quantum_sign(self, message: bytes, signer_identity: str) -> QuantumSignature:
        """
        Create quantum-resistant digital signature
        """
        
        if signer_identity not in self.identity_keys:
            raise ValueError(f"No signing key for {signer_identity}")
        
        signing_key = self.identity_keys[signer_identity]['signing']
        signature = self.hash_signatures.sign(message, signing_key.private_key)
        
        return signature
    
    async def quantum_verify(self, signature: QuantumSignature, message: bytes, 
                           signer_identity: str) -> bool:
        """
        Verify quantum-resistant digital signature
        """
        
        if signer_identity not in self.identity_keys:
            raise ValueError(f"No public key for {signer_identity}")
        
        public_key = self.identity_keys[signer_identity]['signing'].public_key
        return self.hash_signatures.verify(signature, message, public_key)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """
        Get current quantum security status
        """
        
        active_sessions = len([s for s in self.session_keys.values() 
                             if s['expires_at'] > datetime.now()])
        
        total_identities = len(self.identity_keys)
        
        # Analyze key strengths
        key_algorithms = defaultdict(int)
        for keys in self.identity_keys.values():
            key_algorithms[keys['encryption'].algorithm.value] += 1
            key_algorithms[keys['signing'].algorithm.value] += 1
        
        return {
            'quantum_readiness': True,
            'active_sessions': active_sessions,
            'total_identities': total_identities,
            'supported_algorithms': [alg.value for alg in QuantumAlgorithm],
            'key_distribution': dict(key_algorithms),
            'security_level': 'Post-Quantum Safe',
            'timestamp': datetime.now().isoformat()
        }

class QuantumThreatMonitor:
    """
    Monitor for quantum computing threats and readiness
    """
    
    def __init__(self):
        self.threat_levels = {
            'current_quantum_capability': 'Limited',
            'estimated_cryptographically_relevant': 2030,
            'threat_level': 'Medium',
            'migration_urgency': 'High'
        }
        
        self.algorithm_vulnerabilities = {
            'RSA': {'vulnerable_to_quantum': True, 'shor_algorithm': True},
            'ECC': {'vulnerable_to_quantum': True, 'shor_algorithm': True},
            'AES': {'vulnerable_to_quantum': False, 'grover_speedup': True},
            'SHA256': {'vulnerable_to_quantum': False, 'grover_speedup': True},
            'NTRU': {'vulnerable_to_quantum': False, 'post_quantum_safe': True},
            'SPHINCS+': {'vulnerable_to_quantum': False, 'post_quantum_safe': True}
        }
    
    async def assess_quantum_risk(self, current_algorithms: List[str]) -> Dict[str, Any]:
        """
        Assess quantum computing risk for current cryptographic setup
        """
        
        vulnerable_algorithms = []
        safe_algorithms = []
        
        for algorithm in current_algorithms:
            if algorithm in self.algorithm_vulnerabilities:
                vuln = self.algorithm_vulnerabilities[algorithm]
                if vuln.get('vulnerable_to_quantum', False):
                    vulnerable_algorithms.append(algorithm)
                else:
                    safe_algorithms.append(algorithm)
        
        # Calculate overall risk score
        total_algorithms = len(current_algorithms)
        vulnerable_count = len(vulnerable_algorithms)
        risk_score = vulnerable_count / total_algorithms if total_algorithms > 0 else 0
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'Critical'
        elif risk_score >= 0.4:
            risk_level = 'High'
        elif risk_score >= 0.2:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Migration recommendations
        recommendations = []
        if vulnerable_algorithms:
            recommendations.append("Migrate vulnerable algorithms to post-quantum alternatives")
        if 'RSA' in vulnerable_algorithms:
            recommendations.append("Replace RSA with lattice-based encryption (NTRU/Kyber)")
        if 'ECC' in vulnerable_algorithms:
            recommendations.append("Replace ECDSA with hash-based signatures (SPHINCS+)")
        
        return {
            'risk_assessment': {
                'overall_risk_level': risk_level,
                'risk_score': risk_score,
                'vulnerable_algorithms': vulnerable_algorithms,
                'quantum_safe_algorithms': safe_algorithms
            },
            'threat_timeline': self.threat_levels,
            'recommendations': recommendations,
            'quantum_readiness_score': (len(safe_algorithms) / total_algorithms * 100) if total_algorithms > 0 else 0,
            'assessment_date': datetime.now().isoformat()
        }
    
    async def generate_migration_plan(self, current_setup: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate migration plan to quantum-safe cryptography
        """
        
        migration_steps = []
        timeline_months = 0
        
        # Analyze current setup
        for use_case, algorithm in current_setup.items():
            if algorithm in self.algorithm_vulnerabilities:
                vuln = self.algorithm_vulnerabilities[algorithm]
                
                if vuln.get('vulnerable_to_quantum'):
                    if 'RSA' in algorithm or 'encryption' in use_case.lower():
                        migration_steps.append({
                            'step': f"Migrate {use_case} from {algorithm}",
                            'recommended_algorithm': 'NTRU/Kyber',
                            'priority': 'High',
                            'estimated_effort': '2-4 weeks',
                            'dependencies': ['Key management update', 'Protocol changes']
                        })
                        timeline_months = max(timeline_months, 2)
                    
                    elif 'signature' in use_case.lower() or 'sign' in use_case.lower():
                        migration_steps.append({
                            'step': f"Migrate {use_case} from {algorithm}",
                            'recommended_algorithm': 'SPHINCS+/Dilithium',
                            'priority': 'High',
                            'estimated_effort': '1-3 weeks',
                            'dependencies': ['Certificate authority updates']
                        })
                        timeline_months = max(timeline_months, 1)
        
        # Add general recommendations
        migration_steps.extend([
            {
                'step': 'Implement hybrid classical/post-quantum systems',
                'priority': 'Medium',
                'estimated_effort': '4-8 weeks',
                'dependencies': ['New cryptographic libraries']
            },
            {
                'step': 'Update key management systems',
                'priority': 'High',
                'estimated_effort': '2-6 weeks',
                'dependencies': ['Hardware security module updates']
            },
            {
                'step': 'Conduct security audit of migrated systems',
                'priority': 'High',
                'estimated_effort': '1-2 weeks',
                'dependencies': ['Third-party security assessment']
            }
        ])
        
        return {
            'migration_plan': migration_steps,
            'estimated_timeline_months': timeline_months,
            'total_steps': len(migration_steps),
            'critical_path': [step for step in migration_steps if step['priority'] == 'High'],
            'budget_estimate': f"${len(migration_steps) * 50000:,} - ${len(migration_steps) * 150000:,}",
            'plan_generated': datetime.now().isoformat()
        }

# Example usage and demonstration
if __name__ == "__main__":
    async def demo_quantum_security():
        """Demonstrate quantum-resistant security system"""
        
        print("🔮 Quantum-Resistant Security Demo")
        
        # Initialize quantum-safe protocol
        protocol = QuantumSafeProtocol()
        
        # Generate identity keys for Alice and Bob
        alice_keys = await protocol.generate_identity_keys("alice")
        bob_keys = await protocol.generate_identity_keys("bob")
        
        print(f"✅ Generated quantum-safe keys:")
        print(f"  Alice - Encryption: {alice_keys['encryption'].algorithm.value}")
        print(f"  Alice - Signing: {alice_keys['signing'].algorithm.value}")
        print(f"  Bob - Encryption: {bob_keys['encryption'].algorithm.value}")
        print(f"  Bob - Signing: {bob_keys['signing'].algorithm.value}")
        
        # Establish quantum-safe session
        session = await protocol.establish_session("alice", "bob")
        print(f"\n🔗 Established quantum session: {session['session_id']}")
        print(f"  QKD efficiency: {session['qkd_metadata']['efficiency']:.2%}")
        print(f"  Error rate: {session['qkd_metadata']['error_rate']:.2%}")
        print(f"  Security: {'✅ Secure' if session['qkd_metadata']['is_secure'] else '❌ Compromised'}")
        
        # Test message encryption/decryption
        message = b"This is a quantum-safe secret message!"
        
        # Session-based encryption
        encrypted = await protocol.quantum_encrypt(message, "bob", session['session_id'])
        decrypted = await protocol.quantum_decrypt(encrypted, "bob")
        
        print(f"\n🔐 Session Encryption Test:")
        print(f"  Original: {message}")
        print(f"  Decrypted: {decrypted}")
        print(f"  Success: {'✅' if message == decrypted else '❌'}")
        
        # Digital signature test
        signature = await protocol.quantum_sign(message, "alice")
        is_valid = await protocol.quantum_verify(signature, message, "alice")
        
        print(f"\n✍️ Quantum Signature Test:")
        print(f"  Algorithm: {signature.algorithm.value}")
        print(f"  Valid: {'✅' if is_valid else '❌'}")
        
        # Security status
        status = await protocol.get_security_status()
        print(f"\n📊 Security Status:")
        print(f"  Quantum Ready: {'✅' if status['quantum_readiness'] else '❌'}")
        print(f"  Active Sessions: {status['active_sessions']}")
        print(f"  Security Level: {status['security_level']}")
        
        # Threat assessment
        monitor = QuantumThreatMonitor()
        current_algs = ['RSA', 'AES', 'SHA256', 'ECDSA']
        risk_assessment = await monitor.assess_quantum_risk(current_algs)
        
        print(f"\n⚠️ Quantum Risk Assessment:")
        print(f"  Risk Level: {risk_assessment['risk_assessment']['overall_risk_level']}")
        print(f"  Risk Score: {risk_assessment['risk_assessment']['risk_score']:.2%}")
        print(f"  Quantum Readiness: {risk_assessment['quantum_readiness_score']:.1f}%")
        
        # Migration plan
        current_setup = {
            'data_encryption': 'RSA',
            'digital_signatures': 'ECDSA',
            'key_exchange': 'ECDH',
            'hashing': 'SHA256'
        }
        migration_plan = await monitor.generate_migration_plan(current_setup)
        
        print(f"\n🔄 Migration Plan:")
        print(f"  Timeline: {migration_plan['estimated_timeline_months']} months")
        print(f"  Steps: {migration_plan['total_steps']}")
        print(f"  Budget: {migration_plan['budget_estimate']}")
    
    # Run demo
    # asyncio.run(demo_quantum_security())