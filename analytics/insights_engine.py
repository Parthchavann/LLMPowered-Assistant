"""
Advanced Analytics and Insights Engine

Comprehensive analytics system providing:
- User behavior analysis
- Query pattern detection
- Content gap identification
- Performance trend analysis
- Predictive analytics
- Business intelligence dashboards
- Automated insights generation
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    queries: List[str]
    responses: List[str]
    ratings: List[int]
    sources_used: List[str]
    response_times: List[float]
    metadata: Dict[str, Any]

@dataclass
class Insight:
    type: str
    category: str
    title: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    confidence: float
    metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class QueryAnalyzer:
    """Analyze query patterns and extract insights"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns in user queries"""
        if not queries:
            return {'error': 'No queries provided'}
        
        # Basic statistics
        query_lengths = [len(q.split()) for q in queries]
        unique_queries = len(set(queries))
        
        # Most common words
        all_words = ' '.join(queries).lower().split()
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(20)
        
        # Query categorization
        categories = self._categorize_queries(queries)
        
        # Topic modeling
        topics = self._extract_topics(queries)
        
        # Temporal patterns (if timestamps available)
        temporal_patterns = self._analyze_temporal_patterns(queries)
        
        return {
            'total_queries': len(queries),
            'unique_queries': unique_queries,
            'repetition_rate': 1 - (unique_queries / len(queries)),
            'avg_query_length': np.mean(query_lengths),
            'query_length_std': np.std(query_lengths),
            'common_words': common_words,
            'categories': categories,
            'topics': topics,
            'temporal_patterns': temporal_patterns
        }
    
    def _categorize_queries(self, queries: List[str]) -> Dict[str, int]:
        """Categorize queries by intent"""
        categories = {
            'how_to': 0,
            'what_is': 0,
            'troubleshooting': 0,
            'account': 0,
            'billing': 0,
            'feature_request': 0,
            'complaint': 0,
            'general': 0
        }
        
        patterns = {
            'how_to': ['how to', 'how do i', 'how can i', 'steps to'],
            'what_is': ['what is', 'what are', 'what does', 'explain'],
            'troubleshooting': ['error', 'problem', 'issue', 'not working', 'broken', 'fix'],
            'account': ['account', 'login', 'password', 'username', 'profile'],
            'billing': ['bill', 'payment', 'charge', 'refund', 'subscription', 'cost'],
            'feature_request': ['feature', 'add', 'new', 'suggestion', 'request'],
            'complaint': ['bad', 'terrible', 'awful', 'disappointed', 'frustrated']
        }
        
        for query in queries:
            query_lower = query.lower()
            categorized = False
            
            for category, keywords in patterns.items():
                if any(keyword in query_lower for keyword in keywords):
                    categories[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                categories['general'] += 1
        
        return categories
    
    def _extract_topics(self, queries: List[str]) -> Dict[str, Any]:
        """Extract topics using LDA"""
        try:
            if len(queries) < 5:
                return {'error': 'Insufficient queries for topic modeling'}
            
            # Vectorize queries
            tfidf_matrix = self.vectorizer.fit_transform(queries)
            
            # Fit LDA model
            self.lda_model.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'keywords': top_words,
                    'weight': float(topic.max())
                })
            
            return {'topics': topics, 'total_topics': len(topics)}
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_temporal_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze temporal patterns in queries"""
        # This is a simplified version - in production, you'd use actual timestamps
        current_time = datetime.now()
        
        # Simulate hourly distribution
        hourly_dist = np.random.dirichlet(np.ones(24)) * len(queries)
        
        # Simulate daily distribution
        daily_dist = np.random.dirichlet(np.ones(7)) * len(queries)
        
        return {
            'hourly_distribution': {
                str(hour): int(count) for hour, count in enumerate(hourly_dist)
            },
            'daily_distribution': {
                day: int(count) for day, count in zip(
                    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                    daily_dist
                )
            },
            'peak_hour': int(np.argmax(hourly_dist)),
            'peak_day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][np.argmax(daily_dist)]
        }

class UserBehaviorAnalyzer:
    """Analyze user behavior patterns"""
    
    def analyze_user_sessions(self, sessions: List[UserSession]) -> Dict[str, Any]:
        """Analyze user session patterns"""
        if not sessions:
            return {'error': 'No sessions provided'}
        
        # Session duration analysis
        session_durations = []
        for session in sessions:
            if session.end_time:
                duration = (session.end_time - session.start_time).total_seconds() / 60
                session_durations.append(duration)
        
        # Query per session analysis
        queries_per_session = [len(session.queries) for session in sessions]
        
        # User engagement metrics
        user_metrics = self._calculate_user_engagement(sessions)
        
        # Satisfaction analysis
        satisfaction_metrics = self._analyze_satisfaction(sessions)
        
        # Churn prediction indicators
        churn_indicators = self._analyze_churn_indicators(sessions)
        
        return {
            'total_sessions': len(sessions),
            'unique_users': len(set(s.user_id for s in sessions)),
            'avg_session_duration_minutes': np.mean(session_durations) if session_durations else 0,
            'avg_queries_per_session': np.mean(queries_per_session),
            'user_engagement': user_metrics,
            'satisfaction': satisfaction_metrics,
            'churn_indicators': churn_indicators
        }
    
    def _calculate_user_engagement(self, sessions: List[UserSession]) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        user_sessions = defaultdict(list)
        for session in sessions:
            user_sessions[session.user_id].append(session)
        
        # Calculate metrics per user
        engagement_scores = []
        return_rates = []
        
        for user_id, user_session_list in user_sessions.items():
            # Engagement score based on queries and ratings
            total_queries = sum(len(s.queries) for s in user_session_list)
            total_ratings = sum(len(s.ratings) for s in user_session_list)
            avg_rating = np.mean([r for s in user_session_list for r in s.ratings]) if total_ratings > 0 else 0
            
            engagement_score = (total_queries * 0.3 + total_ratings * 0.7) * (avg_rating / 5.0)
            engagement_scores.append(engagement_score)
            
            # Return rate (multiple sessions)
            return_rates.append(len(user_session_list) > 1)
        
        return {
            'avg_engagement_score': np.mean(engagement_scores),
            'return_rate': np.mean(return_rates),
            'high_engagement_users': len([s for s in engagement_scores if s > np.percentile(engagement_scores, 75)]),
            'low_engagement_users': len([s for s in engagement_scores if s < np.percentile(engagement_scores, 25)])
        }
    
    def _analyze_satisfaction(self, sessions: List[UserSession]) -> Dict[str, Any]:
        """Analyze user satisfaction metrics"""
        all_ratings = [r for session in sessions for r in session.ratings]
        
        if not all_ratings:
            return {'error': 'No ratings available'}
        
        satisfaction_distribution = Counter(all_ratings)
        
        return {
            'avg_rating': np.mean(all_ratings),
            'rating_distribution': dict(satisfaction_distribution),
            'satisfaction_rate': len([r for r in all_ratings if r >= 4]) / len(all_ratings),
            'dissatisfaction_rate': len([r for r in all_ratings if r <= 2]) / len(all_ratings),
            'total_ratings': len(all_ratings)
        }
    
    def _analyze_churn_indicators(self, sessions: List[UserSession]) -> Dict[str, Any]:
        """Analyze indicators of user churn"""
        # Group sessions by user
        user_sessions = defaultdict(list)
        for session in sessions:
            user_sessions[session.user_id].append(session)
        
        churn_indicators = {
            'declining_usage': 0,
            'low_satisfaction': 0,
            'short_sessions': 0,
            'no_return': 0
        }
        
        total_users = len(user_sessions)
        
        for user_id, user_session_list in user_sessions.items():
            # Sort sessions by time
            user_session_list.sort(key=lambda s: s.start_time)
            
            # Check for declining usage
            if len(user_session_list) >= 3:
                recent_sessions = user_session_list[-2:]
                older_sessions = user_session_list[:-2]
                
                recent_queries = sum(len(s.queries) for s in recent_sessions)
                older_avg_queries = np.mean([len(s.queries) for s in older_sessions])
                
                if recent_queries < older_avg_queries * 0.5:
                    churn_indicators['declining_usage'] += 1
            
            # Check for low satisfaction
            all_user_ratings = [r for s in user_session_list for r in s.ratings]
            if all_user_ratings and np.mean(all_user_ratings) < 3.0:
                churn_indicators['low_satisfaction'] += 1
            
            # Check for consistently short sessions
            session_durations = []
            for session in user_session_list:
                if session.end_time:
                    duration = (session.end_time - session.start_time).total_seconds() / 60
                    session_durations.append(duration)
            
            if session_durations and np.mean(session_durations) < 2.0:  # Less than 2 minutes
                churn_indicators['short_sessions'] += 1
            
            # Check for no return (single session users)
            if len(user_session_list) == 1:
                churn_indicators['no_return'] += 1
        
        # Convert to percentages
        for key in churn_indicators:
            churn_indicators[key] = (churn_indicators[key] / total_users) * 100
        
        return churn_indicators

class ContentGapAnalyzer:
    """Identify gaps in content coverage"""
    
    def identify_content_gaps(
        self, 
        queries: List[str], 
        successful_queries: List[str],
        document_contents: List[str]
    ) -> Dict[str, Any]:
        """Identify content gaps based on query analysis"""
        
        # Failed queries (queries not in successful ones)
        failed_queries = [q for q in queries if q not in successful_queries]
        
        # Analyze failed queries for patterns
        failed_patterns = self._analyze_failed_queries(failed_queries)
        
        # Content coverage analysis
        coverage_analysis = self._analyze_content_coverage(queries, document_contents)
        
        # Topic gaps
        topic_gaps = self._identify_topic_gaps(queries, document_contents)
        
        return {
            'failed_queries_count': len(failed_queries),
            'failure_rate': len(failed_queries) / len(queries) if queries else 0,
            'failed_patterns': failed_patterns,
            'content_coverage': coverage_analysis,
            'topic_gaps': topic_gaps,
            'recommendations': self._generate_content_recommendations(failed_patterns, topic_gaps)
        }
    
    def _analyze_failed_queries(self, failed_queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns in failed queries"""
        if not failed_queries:
            return {'common_themes': [], 'keywords': []}
        
        # Extract common keywords
        all_words = ' '.join(failed_queries).lower().split()
        keyword_counts = Counter(all_words)
        
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with'}
        filtered_keywords = [(word, count) for word, count in keyword_counts.items() 
                           if word not in stop_words and len(word) > 2]
        
        # Identify common themes
        themes = []
        theme_patterns = {
            'integration': ['api', 'integrate', 'connect', 'webhook'],
            'advanced_features': ['advanced', 'custom', 'enterprise', 'professional'],
            'mobile': ['mobile', 'ios', 'android', 'app'],
            'reporting': ['report', 'analytics', 'dashboard', 'metrics'],
            'security': ['security', 'permission', 'access', 'auth']
        }
        
        for theme, keywords in theme_patterns.items():
            matches = sum(1 for query in failed_queries 
                         if any(keyword in query.lower() for keyword in keywords))
            if matches > 0:
                themes.append({'theme': theme, 'frequency': matches})
        
        return {
            'common_keywords': filtered_keywords[:10],
            'themes': themes,
            'total_failed': len(failed_queries)
        }
    
    def _analyze_content_coverage(self, queries: List[str], documents: List[str]) -> Dict[str, Any]:
        """Analyze how well documents cover query topics"""
        if not queries or not documents:
            return {'coverage_score': 0, 'gaps': []}
        
        try:
            # Vectorize queries and documents
            all_text = queries + documents
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_text)
            
            # Separate query and document vectors
            query_vectors = tfidf_matrix[:len(queries)]
            doc_vectors = tfidf_matrix[len(queries):]
            
            # Calculate similarity between queries and documents
            similarities = cosine_similarity(query_vectors, doc_vectors)
            
            # Find queries with low similarity to any document
            max_similarities = np.max(similarities, axis=1)
            low_coverage_threshold = 0.1
            
            gaps = []
            for i, similarity in enumerate(max_similarities):
                if similarity < low_coverage_threshold:
                    gaps.append({
                        'query': queries[i],
                        'max_similarity': float(similarity),
                        'gap_severity': 'high' if similarity < 0.05 else 'medium'
                    })
            
            coverage_score = np.mean(max_similarities)
            
            return {
                'coverage_score': float(coverage_score),
                'gaps': gaps[:20],  # Top 20 gaps
                'well_covered_queries': int(np.sum(max_similarities > 0.3)),
                'poorly_covered_queries': int(np.sum(max_similarities < 0.1))
            }
            
        except Exception as e:
            logger.error(f"Content coverage analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _identify_topic_gaps(self, queries: List[str], documents: List[str]) -> List[Dict[str, Any]]:
        """Identify topic gaps between queries and documents"""
        gaps = []
        
        # Simple keyword-based gap detection
        query_keywords = set()
        for query in queries:
            query_keywords.update(query.lower().split())
        
        doc_keywords = set()
        for doc in documents:
            doc_keywords.update(doc.lower().split())
        
        # Find keywords in queries but not in documents
        missing_keywords = query_keywords - doc_keywords
        
        # Filter out common words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'how', 'do', 'i'}
        significant_missing = [word for word in missing_keywords 
                             if len(word) > 3 and word not in stop_words]
        
        for keyword in significant_missing[:10]:  # Top 10
            # Count frequency in queries
            frequency = sum(1 for query in queries if keyword in query.lower())
            gaps.append({
                'missing_topic': keyword,
                'query_frequency': frequency,
                'gap_type': 'keyword_missing'
            })
        
        return gaps
    
    def _generate_content_recommendations(
        self, 
        failed_patterns: Dict[str, Any], 
        topic_gaps: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate content creation recommendations"""
        recommendations = []
        
        # Recommendations based on failed patterns
        for theme_info in failed_patterns.get('themes', []):
            if theme_info['frequency'] > 5:
                recommendations.append(
                    f"Create documentation for {theme_info['theme']} "
                    f"({theme_info['frequency']} related failed queries)"
                )
        
        # Recommendations based on topic gaps
        high_frequency_gaps = [gap for gap in topic_gaps if gap['query_frequency'] > 3]
        for gap in high_frequency_gaps[:3]:
            recommendations.append(
                f"Add content covering '{gap['missing_topic']}' "
                f"(mentioned in {gap['query_frequency']} queries)"
            )
        
        # General recommendations
        if failed_patterns.get('total_failed', 0) > len(failed_patterns.get('common_keywords', [])) * 2:
            recommendations.append("Consider expanding FAQ section with more specific examples")
        
        if not recommendations:
            recommendations.append("Content coverage appears adequate based on current analysis")
        
        return recommendations

class InsightsEngine:
    """Main insights generation engine"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.content_analyzer = ContentGapAnalyzer()
        self.insights_history = []
    
    def generate_comprehensive_insights(
        self, 
        queries: List[str],
        sessions: List[UserSession],
        successful_queries: List[str],
        document_contents: List[str],
        response_times: List[float],
        satisfaction_ratings: List[int]
    ) -> Dict[str, Any]:
        """Generate comprehensive insights from all available data"""
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_queries': len(queries),
                'total_sessions': len(sessions),
                'unique_users': len(set(s.user_id for s in sessions)),
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'avg_satisfaction': np.mean(satisfaction_ratings) if satisfaction_ratings else 0
            }
        }
        
        # Query pattern analysis
        insights['query_patterns'] = self.query_analyzer.analyze_query_patterns(queries)
        
        # User behavior analysis
        insights['user_behavior'] = self.behavior_analyzer.analyze_user_sessions(sessions)
        
        # Content gap analysis
        insights['content_gaps'] = self.content_analyzer.identify_content_gaps(
            queries, successful_queries, document_contents
        )
        
        # Performance insights
        insights['performance'] = self._analyze_performance_trends(
            response_times, satisfaction_ratings
        )
        
        # Generate actionable insights
        insights['actionable_insights'] = self._generate_actionable_insights(insights)
        
        # Store insights for historical analysis
        self.insights_history.append(insights)
        
        return insights
    
    def _analyze_performance_trends(
        self, 
        response_times: List[float], 
        satisfaction_ratings: List[int]
    ) -> Dict[str, Any]:
        """Analyze performance trends"""
        
        if not response_times or not satisfaction_ratings:
            return {'error': 'Insufficient performance data'}
        
        # Response time analysis
        rt_percentiles = {
            'p50': np.percentile(response_times, 50),
            'p90': np.percentile(response_times, 90),
            'p95': np.percentile(response_times, 95),
            'p99': np.percentile(response_times, 99)
        }
        
        # Satisfaction analysis
        satisfaction_dist = Counter(satisfaction_ratings)
        
        # Correlation between response time and satisfaction
        correlation = np.corrcoef(response_times[:len(satisfaction_ratings)], satisfaction_ratings)[0, 1]
        
        return {
            'response_time_percentiles': rt_percentiles,
            'avg_response_time': np.mean(response_times),
            'response_time_std': np.std(response_times),
            'satisfaction_distribution': dict(satisfaction_dist),
            'rt_satisfaction_correlation': float(correlation) if not np.isnan(correlation) else 0,
            'performance_score': self._calculate_performance_score(rt_percentiles, satisfaction_ratings)
        }
    
    def _calculate_performance_score(
        self, 
        rt_percentiles: Dict[str, float], 
        satisfaction_ratings: List[int]
    ) -> float:
        """Calculate overall performance score (0-100)"""
        
        # Response time score (lower is better)
        rt_score = max(0, 100 - (rt_percentiles['p95'] / 1000) * 10)  # Penalize high response times
        
        # Satisfaction score
        avg_satisfaction = np.mean(satisfaction_ratings)
        satisfaction_score = (avg_satisfaction / 5.0) * 100
        
        # Combined score (weighted average)
        performance_score = (rt_score * 0.3) + (satisfaction_score * 0.7)
        
        return round(performance_score, 2)
    
    def _generate_actionable_insights(self, insights: Dict[str, Any]) -> List[Insight]:
        """Generate actionable insights from analysis"""
        actionable_insights = []
        
        # Query pattern insights
        query_patterns = insights.get('query_patterns', {})
        if query_patterns.get('repetition_rate', 0) > 0.3:
            actionable_insights.append(Insight(
                type='optimization',
                category='user_experience',
                title='High Query Repetition Detected',
                description=f"Users are repeating {query_patterns['repetition_rate']:.1%} of queries, indicating potential UX issues.",
                impact='medium',
                confidence=0.8,
                metrics={'repetition_rate': query_patterns['repetition_rate']},
                recommendations=[
                    'Improve search result relevance',
                    'Add suggested questions feature',
                    'Enhance query auto-completion'
                ],
                timestamp=datetime.now()
            ))
        
        # Content gap insights
        content_gaps = insights.get('content_gaps', {})
        if content_gaps.get('failure_rate', 0) > 0.2:
            actionable_insights.append(Insight(
                type='content',
                category='knowledge_base',
                title='Content Gaps Identified',
                description=f"Query failure rate of {content_gaps['failure_rate']:.1%} indicates missing content.",
                impact='high',
                confidence=0.9,
                metrics={'failure_rate': content_gaps['failure_rate']},
                recommendations=content_gaps.get('recommendations', []),
                timestamp=datetime.now()
            ))
        
        # Performance insights
        performance = insights.get('performance', {})
        if performance.get('performance_score', 100) < 70:
            actionable_insights.append(Insight(
                type='performance',
                category='system',
                title='Performance Issues Detected',
                description=f"Overall performance score of {performance['performance_score']} indicates system issues.",
                impact='high',
                confidence=0.85,
                metrics={'performance_score': performance['performance_score']},
                recommendations=[
                    'Optimize response generation pipeline',
                    'Implement caching for common queries',
                    'Scale infrastructure resources'
                ],
                timestamp=datetime.now()
            ))
        
        # User behavior insights
        user_behavior = insights.get('user_behavior', {})
        satisfaction = user_behavior.get('satisfaction', {})
        if satisfaction.get('dissatisfaction_rate', 0) > 0.15:
            actionable_insights.append(Insight(
                type='quality',
                category='user_satisfaction',
                title='High Dissatisfaction Rate',
                description=f"Dissatisfaction rate of {satisfaction['dissatisfaction_rate']:.1%} requires attention.",
                impact='high',
                confidence=0.9,
                metrics={'dissatisfaction_rate': satisfaction['dissatisfaction_rate']},
                recommendations=[
                    'Review and improve low-rated responses',
                    'Implement human escalation for complex queries',
                    'Enhance response quality validation'
                ],
                timestamp=datetime.now()
            ))
        
        return actionable_insights
    
    def get_insights_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for insights dashboard"""
        if not self.insights_history:
            return {'error': 'No insights history available'}
        
        latest_insights = self.insights_history[-1]
        
        # Key metrics
        key_metrics = {
            'total_queries': latest_insights['data_summary']['total_queries'],
            'satisfaction_score': latest_insights['data_summary']['avg_satisfaction'],
            'response_time': latest_insights['data_summary']['avg_response_time'],
            'content_coverage': latest_insights['content_gaps'].get('content_coverage', {}).get('coverage_score', 0)
        }
        
        # Trends (if multiple insights available)
        trends = {}
        if len(self.insights_history) > 1:
            prev_insights = self.insights_history[-2]
            
            for metric in key_metrics:
                current = key_metrics[metric]
                previous = prev_insights['data_summary'].get(metric, current)
                
                if previous != 0:
                    change = ((current - previous) / previous) * 100
                    trends[metric] = {
                        'current': current,
                        'previous': previous,
                        'change_percent': round(change, 2),
                        'direction': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                    }
        
        # Top insights
        actionable_insights = latest_insights.get('actionable_insights', [])
        top_insights = sorted(
            actionable_insights, 
            key=lambda x: (x.impact == 'high', x.confidence), 
            reverse=True
        )[:5]
        
        return {
            'key_metrics': key_metrics,
            'trends': trends,
            'top_insights': [asdict(insight) for insight in top_insights],
            'query_categories': latest_insights['query_patterns'].get('categories', {}),
            'user_engagement': latest_insights['user_behavior'].get('user_engagement', {}),
            'last_updated': latest_insights['timestamp']
        }