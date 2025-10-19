"""
Advanced Database Manager - CS-499 Milestone Four: Enhancement Three (Databases)
Comprehensive database analytics, performance monitoring, and optimization system.

This module demonstrates advanced database concepts including:
1. Complex aggregation pipelines and analytics
2. Performance monitoring and query optimization
3. Advanced indexing strategies
4. Security and access control
5. Data export/import capabilities
6. Database health monitoring and diagnostics

Author: Milfred Martinez
Course: CS-499 Capstone - Software Engineering & Design
Milestone: Four - Enhancement Three (Databases)
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum

from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import PyMongoError, OperationFailure
from pymongo.collection import Collection
from pymongo.database import Database
import pymongo.monitoring

class QueryType(Enum):
    """Database query types for performance monitoring."""
    INSERT = "insert"
    FIND = "find"
    UPDATE = "update" 
    DELETE = "delete"
    AGGREGATE = "aggregate"
    INDEX = "index"

class AccessLevel(Enum):
    """User access levels for role-based access control."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    ANALYTICS = "analytics"

@dataclass
class QueryPerformanceMetrics:
    """Data structure for storing query performance metrics."""
    query_type: QueryType
    collection_name: str
    execution_time_ms: float
    documents_examined: int
    documents_returned: int
    index_used: bool
    index_name: Optional[str]
    timestamp: datetime
    query_shape: str  # Simplified query structure for analysis

@dataclass
class DatabaseHealthMetrics:
    """Comprehensive database health and performance metrics."""
    total_connections: int
    active_connections: int
    database_size_mb: float
    total_documents: int
    total_collections: int
    index_count: int
    average_query_time_ms: float
    slow_queries_count: int
    disk_usage_mb: float
    memory_usage_mb: float
    uptime_hours: float
    timestamp: datetime

@dataclass
class AnalyticsReport:
    """Structured analytics report for business intelligence."""
    report_type: str
    report_name: str
    generated_at: datetime
    data_period: Tuple[datetime, datetime]
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    visualizations: Dict[str, Any]

class CommandListener(pymongo.monitoring.CommandListener):
    """MongoDB command listener for performance monitoring."""
    
    def __init__(self, performance_monitor):
        self.performance_monitor = performance_monitor
        self.command_start_times = {}
    
    def started(self, event):
        """Record command start time."""
        self.command_start_times[event.request_id] = time.perf_counter()
    
    def succeeded(self, event):
        """Record successful command completion."""
        if event.request_id in self.command_start_times:
            start_time = self.command_start_times.pop(event.request_id)
            execution_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            self.performance_monitor.record_query_performance(
                command_name=event.command_name,
                collection_name=event.reply.get('ns', '').split('.')[-1] if 'ns' in event.reply else 'unknown',
                execution_time_ms=execution_time,
                documents_examined=event.reply.get('executionStats', {}).get('totalDocsExamined', 0),
                documents_returned=event.reply.get('executionStats', {}).get('totalDocsReturned', 0),
                event_details=event.reply
            )
    
    def failed(self, event):
        """Record failed command completion."""
        if event.request_id in self.command_start_times:
            start_time = self.command_start_times.pop(event.request_id)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            self.performance_monitor.record_failed_query(
                command_name=event.command_name,
                execution_time_ms=execution_time,
                failure_reason=str(event.failure)
            )

class PerformanceMonitor:
    """Advanced database performance monitoring and optimization system."""
    
    def __init__(self, max_metrics_history: int = 10000):
        self.logger = logging.getLogger("db_performance")
        self.query_metrics: List[QueryPerformanceMetrics] = []
        self.failed_queries: List[Dict[str, Any]] = []
        self.max_metrics_history = max_metrics_history
        self.metrics_lock = threading.Lock()
        
        # Performance thresholds
        self.slow_query_threshold_ms = 100
        self.optimization_recommendations = []
    
    def record_query_performance(self, command_name: str, collection_name: str, 
                                execution_time_ms: float, documents_examined: int = 0,
                                documents_returned: int = 0, event_details: Dict = None):
        """Record query performance metrics for analysis."""
        try:
            query_type = QueryType(command_name.lower()) if command_name.lower() in [e.value for e in QueryType] else QueryType.FIND
        except ValueError:
            query_type = QueryType.FIND
        
        # Extract index usage information
        index_used = False
        index_name = None
        if event_details and 'executionStats' in event_details:
            exec_stats = event_details['executionStats']
            if exec_stats.get('stage') == 'IXSCAN':
                index_used = True
                index_name = exec_stats.get('indexName')
        
        metric = QueryPerformanceMetrics(
            query_type=query_type,
            collection_name=collection_name,
            execution_time_ms=execution_time_ms,
            documents_examined=documents_examined,
            documents_returned=documents_returned,
            index_used=index_used,
            index_name=index_name,
            timestamp=datetime.utcnow(),
            query_shape=self._extract_query_shape(event_details)
        )
        
        with self.metrics_lock:
            self.query_metrics.append(metric)
            
            # Maintain metrics history size
            if len(self.query_metrics) > self.max_metrics_history:
                self.query_metrics = self.query_metrics[-self.max_metrics_history:]
            
            # Check for slow queries
            if execution_time_ms > self.slow_query_threshold_ms:
                self.logger.warning(f"Slow query detected: {command_name} on {collection_name} took {execution_time_ms:.2f}ms")
                self._generate_optimization_recommendation(metric)
    
    def record_failed_query(self, command_name: str, execution_time_ms: float, failure_reason: str):
        """Record failed query for analysis."""
        failed_query = {
            "command": command_name,
            "execution_time_ms": execution_time_ms,
            "failure_reason": failure_reason,
            "timestamp": datetime.utcnow()
        }
        
        with self.metrics_lock:
            self.failed_queries.append(failed_query)
            if len(self.failed_queries) > 1000:  # Keep last 1000 failures
                self.failed_queries = self.failed_queries[-1000:]
    
    def _extract_query_shape(self, event_details: Dict) -> str:
        """Extract simplified query shape for analysis."""
        if not event_details:
            return "unknown"
        
        command = event_details.get('command', {})
        if 'find' in command:
            filter_keys = list(command.get('filter', {}).keys())
            return f"find({','.join(filter_keys)})"
        elif 'aggregate' in command:
            pipeline_stages = [stage.keys() for stage in command.get('pipeline', [])]
            return f"aggregate([{','.join(str(s) for s in pipeline_stages)}])"
        elif 'update' in command:
            filter_keys = list(command.get('q', {}).keys())
            return f"update({','.join(filter_keys)})"
        
        return f"{list(command.keys())[0] if command else 'unknown'}"
    
    def _generate_optimization_recommendation(self, metric: QueryPerformanceMetrics):
        """Generate optimization recommendations for slow queries."""
        recommendations = []
        
        if not metric.index_used and metric.documents_examined > 100:
            recommendations.append(f"Consider adding index for {metric.collection_name} collection on frequently queried fields")
        
        if metric.documents_examined > metric.documents_returned * 10:
            recommendations.append(f"Query inefficiency detected in {metric.collection_name}: examined {metric.documents_examined} docs but returned {metric.documents_returned}")
        
        if metric.execution_time_ms > self.slow_query_threshold_ms * 5:
            recommendations.append(f"Critical performance issue in {metric.collection_name}: {metric.execution_time_ms:.2f}ms execution time")
        
        self.optimization_recommendations.extend(recommendations)
    
    def get_performance_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        with self.metrics_lock:
            recent_metrics = [m for m in self.query_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No metrics available for specified time period"}
        
        # Calculate performance statistics
        execution_times = [m.execution_time_ms for m in recent_metrics]
        slow_queries = [m for m in recent_metrics if m.execution_time_ms > self.slow_query_threshold_ms]
        
        query_type_stats = defaultdict(list)
        collection_stats = defaultdict(list)
        index_usage_stats = {"used": 0, "not_used": 0}
        
        for metric in recent_metrics:
            query_type_stats[metric.query_type.value].append(metric.execution_time_ms)
            collection_stats[metric.collection_name].append(metric.execution_time_ms)
            
            if metric.index_used:
                index_usage_stats["used"] += 1
            else:
                index_usage_stats["not_used"] += 1
        
        return {
            "time_period_hours": hours_back,
            "total_queries": len(recent_metrics),
            "average_execution_time_ms": sum(execution_times) / len(execution_times),
            "median_execution_time_ms": sorted(execution_times)[len(execution_times) // 2],
            "max_execution_time_ms": max(execution_times),
            "slow_queries_count": len(slow_queries),
            "slow_queries_percentage": (len(slow_queries) / len(recent_metrics)) * 100,
            "query_types": {qtype: {
                "count": len(times),
                "avg_time_ms": sum(times) / len(times)
            } for qtype, times in query_type_stats.items()},
            "collection_performance": {collection: {
                "count": len(times),
                "avg_time_ms": sum(times) / len(times)
            } for collection, times in collection_stats.items()},
            "index_usage": index_usage_stats,
            "failed_queries_count": len([f for f in self.failed_queries if f["timestamp"] >= cutoff_time]),
            "optimization_recommendations": list(set(self.optimization_recommendations[-10:]))  # Last 10 unique recommendations
        }

class AdvancedDatabaseManager:
    """
    Advanced Database Manager with analytics, performance monitoring, and optimization.
    
    Features:
    1. Complex aggregation pipelines for business intelligence
    2. Performance monitoring and query optimization
    3. Advanced indexing strategies
    4. Security and access control
    5. Data export/import capabilities
    6. Health monitoring and diagnostics
    """
    
    def __init__(self, mongo_uri: str = None, db_name: str = "advanced_db"):
        self.logger = logging.getLogger("advanced_db_manager")
        self.mongo_uri = mongo_uri or os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
        self.db_name = db_name
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Setup monitoring
        command_listener = CommandListener(self.performance_monitor)
        
        # Create client with monitoring
        self.client = MongoClient(
            self.mongo_uri,
            event_listeners=[command_listener],
            maxPoolSize=50,  # Connection pooling
            minPoolSize=5,
            maxIdleTimeMS=30000,
            serverSelectionTimeoutMS=5000
        )
        
        self.database = self.client[self.db_name]
        
        # Initialize collections
        self.animals_collection = self.database["animals"]
        self.medical_notes_collection = self.database["medical_notes"]
        self.audits_collection = self.database["audits"]
        self.analytics_cache_collection = self.database["analytics_cache"]
        
        # Setup advanced indexes
        self._setup_advanced_indexes()
        
        self.logger.info(f"Advanced Database Manager initialized for {self.db_name}")
    
    def _setup_advanced_indexes(self):
        """Setup comprehensive indexing strategy for optimal query performance."""
        try:
            # Animals collection indexes
            animal_indexes = [
                IndexModel([("animal_type", ASCENDING), ("outcome_type", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),  # For recent records
                IndexModel([("age", ASCENDING)]),
                IndexModel([("breed", TEXT)]),  # Full-text search
                IndexModel([("name", ASCENDING), ("animal_type", ASCENDING)]),  # Compound index
                IndexModel([("outcome_type", ASCENDING), ("created_at", DESCENDING)]),  # Sorting optimization
            ]
            
            # Medical notes indexes
            medical_indexes = [
                IndexModel([("patient_id", ASCENDING), ("processed_at", DESCENDING)]),
                IndexModel([("provider_id", ASCENDING)]),
                IndexModel([("note_type", ASCENDING), ("processed_at", DESCENDING)]),
                IndexModel([("original_text", TEXT)]),  # Full-text search on content
                IndexModel([("soap_sections.subjective.sentences", TEXT)]),  # SOAP section search
                IndexModel([("metadata.processing_time_ms", ASCENDING)]),  # Performance analysis
            ]
            
            # Audit collection indexes
            audit_indexes = [
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("action", ASCENDING), ("timestamp", DESCENDING)]),
                IndexModel([("target_id", ASCENDING)]),
                IndexModel([("meta.user_id", ASCENDING), ("timestamp", DESCENDING)])
            ]
            
            # Analytics cache indexes
            cache_indexes = [
                IndexModel([("report_type", ASCENDING), ("generated_at", DESCENDING)]),
                IndexModel([("cache_key", ASCENDING)]),
                IndexModel([("expires_at", ASCENDING)])  # TTL-like functionality
            ]
            
            # Create indexes
            self.animals_collection.create_indexes(animal_indexes)
            self.medical_notes_collection.create_indexes(medical_indexes)
            self.audits_collection.create_indexes(audit_indexes)
            self.analytics_cache_collection.create_indexes(cache_indexes)
            
            self.logger.info("Advanced database indexes created successfully")
            
        except PyMongoError as e:
            self.logger.error(f"Failed to create indexes: {str(e)}")
    
    def generate_animal_shelter_analytics(self, days_back: int = 30) -> AnalyticsReport:
        """
        Generate comprehensive animal shelter analytics using advanced aggregation pipelines.
        
        Demonstrates complex MongoDB aggregation queries and business intelligence.
        """
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        try:
            # Complex aggregation pipeline for animal shelter insights
            pipeline = [
                # Match recent records
                {
                    "$match": {
                        "created_at": {"$gte": start_date}
                    }
                },
                
                # Group by animal type and outcome with statistics
                {
                    "$group": {
                        "_id": {
                            "animal_type": "$animal_type",
                            "outcome_type": "$outcome_type"
                        },
                        "count": {"$sum": 1},
                        "avg_age": {"$avg": "$age"},
                        "breeds": {"$addToSet": "$breed"},
                        "total_days_in_shelter": {"$sum": {"$divide": [{"$subtract": [datetime.utcnow(), "$created_at"]}, 86400000]}}  # Convert to days
                    }
                },
                
                # Add calculated fields
                {
                    "$addFields": {
                        "avg_days_in_shelter": {"$divide": ["$total_days_in_shelter", "$count"]},
                        "breed_diversity": {"$size": "$breeds"},
                        "outcome_rate": {
                            "$cond": [
                                {"$eq": ["$_id.outcome_type", "Adopted"]}, 1.0,
                                {"$cond": [{"$eq": ["$_id.outcome_type", "Available"]}, 0.0, 0.5]}
                            ]
                        }
                    }
                },
                
                # Sort by count descending
                {"$sort": {"count": -1}},
                
                # Group again to create summary statistics
                {
                    "$group": {
                        "_id": None,
                        "total_animals": {"$sum": "$count"},
                        "animal_breakdown": {
                            "$push": {
                                "animal_type": "$_id.animal_type",
                                "outcome_type": "$_id.outcome_type", 
                                "count": "$count",
                                "avg_age": "$avg_age",
                                "avg_days_in_shelter": "$avg_days_in_shelter",
                                "breed_diversity": "$breed_diversity",
                                "outcome_rate": "$outcome_rate"
                            }
                        },
                        "avg_age_overall": {"$avg": "$avg_age"},
                        "avg_outcome_rate": {"$avg": "$outcome_rate"}
                    }
                }
            ]
            
            # Execute aggregation
            result = list(self.animals_collection.aggregate(pipeline))
            
            if not result:
                return AnalyticsReport(
                    report_type="animal_shelter",
                    report_name="Animal Shelter Analytics",
                    generated_at=datetime.utcnow(),
                    data_period=(start_date, datetime.utcnow()),
                    metrics={},
                    insights=["No data available for the specified period"],
                    recommendations=[],
                    visualizations={}
                )
            
            data = result[0]
            
            # Generate insights based on data analysis
            insights = self._generate_animal_shelter_insights(data)
            recommendations = self._generate_animal_shelter_recommendations(data)
            
            # Prepare visualization data
            visualizations = {
                "animal_type_distribution": self._prepare_animal_type_chart(data["animal_breakdown"]),
                "outcome_trends": self._prepare_outcome_trends_chart(data["animal_breakdown"]),
                "age_demographics": self._prepare_age_demographics(data["animal_breakdown"])
            }
            
            return AnalyticsReport(
                report_type="animal_shelter",
                report_name="Animal Shelter Analytics",
                generated_at=datetime.utcnow(),
                data_period=(start_date, datetime.utcnow()),
                metrics={
                    "total_animals": data["total_animals"],
                    "average_age": round(data.get("avg_age_overall", 0), 2),
                    "overall_outcome_rate": round(data.get("avg_outcome_rate", 0), 3),
                    "breakdown": data["animal_breakdown"]
                },
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations
            )
            
        except PyMongoError as e:
            self.logger.error(f"Error generating animal shelter analytics: {str(e)}")
            raise
    
    def generate_medical_notes_analytics(self, days_back: int = 30) -> AnalyticsReport:
        """
        Generate comprehensive medical notes analytics with SOAP section analysis.
        
        Demonstrates advanced text analytics and medical data intelligence.
        """
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        try:
            # Advanced pipeline for medical notes analysis
            pipeline = [
                {
                    "$match": {
                        "processed_at": {"$gte": start_date.isoformat()}
                    }
                },
                
                # Unwind SOAP sections for detailed analysis
                {
                    "$addFields": {
                        "soap_section_counts": {
                            "subjective": {"$size": "$soap_sections.subjective.sentences"},
                            "objective": {"$size": "$soap_sections.objective.sentences"},
                            "assessment": {"$size": "$soap_sections.assessment.sentences"},
                            "plan": {"$size": "$soap_sections.plan.sentences"}
                        },
                        "total_entities": {
                            "$add": [
                                {"$size": "$soap_sections.subjective.entities"},
                                {"$size": "$soap_sections.objective.entities"},
                                {"$size": "$soap_sections.assessment.entities"},
                                {"$size": "$soap_sections.plan.entities"}
                            ]
                        }
                    }
                },
                
                # Group by provider and note type
                {
                    "$group": {
                        "_id": {
                            "provider_id": "$provider_id",
                            "note_type": "$note_type",
                            "algorithm_used": "$algorithm_used"
                        },
                        "note_count": {"$sum": 1},
                        "avg_processing_time": {"$avg": "$metadata.processing_time_ms"},
                        "avg_total_sentences": {"$avg": "$metadata.total_sentences"},
                        "avg_soap_subjective": {"$avg": "$soap_section_counts.subjective"},
                        "avg_soap_objective": {"$avg": "$soap_section_counts.objective"},
                        "avg_soap_assessment": {"$avg": "$soap_section_counts.assessment"},
                        "avg_soap_plan": {"$avg": "$soap_section_counts.plan"},
                        "avg_entities_extracted": {"$avg": "$total_entities"},
                        "algorithms_used": {"$addToSet": "$algorithm_used"}
                    }
                },
                
                # Calculate efficiency metrics
                {
                    "$addFields": {
                        "processing_efficiency": {
                            "$divide": ["$avg_total_sentences", {"$max": ["$avg_processing_time", 1]}]
                        },
                        "soap_completeness_score": {
                            "$divide": [
                                {"$add": [
                                    {"$cond": [{"$gt": ["$avg_soap_subjective", 0]}, 1, 0]},
                                    {"$cond": [{"$gt": ["$avg_soap_objective", 0]}, 1, 0]},
                                    {"$cond": [{"$gt": ["$avg_soap_assessment", 0]}, 1, 0]},
                                    {"$cond": [{"$gt": ["$avg_soap_plan", 0]}, 1, 0]}
                                ]},
                                4
                            ]
                        }
                    }
                },
                
                # Final grouping for summary
                {
                    "$group": {
                        "_id": None,
                        "total_notes": {"$sum": "$note_count"},
                        "unique_providers": {"$addToSet": "$_id.provider_id"},
                        "note_types": {"$addToSet": "$_id.note_type"},
                        "algorithms_performance": {
                            "$push": {
                                "algorithm": "$_id.algorithm_used",
                                "provider": "$_id.provider_id",
                                "note_type": "$_id.note_type",
                                "note_count": "$note_count",
                                "avg_processing_time": "$avg_processing_time",
                                "processing_efficiency": "$processing_efficiency",
                                "soap_completeness": "$soap_completeness_score",
                                "avg_entities": "$avg_entities_extracted"
                            }
                        },
                        "overall_avg_processing_time": {"$avg": "$avg_processing_time"},
                        "overall_efficiency": {"$avg": "$processing_efficiency"},
                        "overall_soap_completeness": {"$avg": "$soap_completeness_score"}
                    }
                }
            ]
            
            result = list(self.medical_notes_collection.aggregate(pipeline))
            
            if not result:
                return AnalyticsReport(
                    report_type="medical_notes",
                    report_name="Medical Notes Analytics",
                    generated_at=datetime.utcnow(),
                    data_period=(start_date, datetime.utcnow()),
                    metrics={},
                    insights=["No medical notes data available for the specified period"],
                    recommendations=[],
                    visualizations={}
                )
            
            data = result[0]
            
            # Generate medical insights
            insights = self._generate_medical_notes_insights(data)
            recommendations = self._generate_medical_notes_recommendations(data)
            
            # Prepare medical visualizations
            visualizations = {
                "algorithm_performance": self._prepare_algorithm_performance_chart(data["algorithms_performance"]),
                "provider_efficiency": self._prepare_provider_efficiency_chart(data["algorithms_performance"]),
                "soap_completeness": self._prepare_soap_completeness_chart(data["algorithms_performance"])
            }
            
            return AnalyticsReport(
                report_type="medical_notes",
                report_name="Medical Notes Analytics",
                generated_at=datetime.utcnow(),
                data_period=(start_date, datetime.utcnow()),
                metrics={
                    "total_notes": data["total_notes"],
                    "unique_providers": len(data["unique_providers"]),
                    "note_types": len(data["note_types"]),
                    "avg_processing_time_ms": round(data.get("overall_avg_processing_time", 0), 2),
                    "processing_efficiency": round(data.get("overall_efficiency", 0), 2),
                    "soap_completeness_avg": round(data.get("overall_soap_completeness", 0), 2),
                    "algorithms_performance": data["algorithms_performance"]
                },
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations
            )
            
        except PyMongoError as e:
            self.logger.error(f"Error generating medical notes analytics: {str(e)}")
            raise
    
    def get_database_health_metrics(self) -> DatabaseHealthMetrics:
        """
        Generate comprehensive database health and performance metrics.
        
        Provides insights into database performance, resource usage, and optimization opportunities.
        """
        try:
            # Get database statistics
            db_stats = self.database.command("dbStats")
            server_status = self.database.client.admin.command("serverStatus")
            
            # Calculate derived metrics
            total_docs = sum([
                self.animals_collection.estimated_document_count(),
                self.medical_notes_collection.estimated_document_count(), 
                self.audits_collection.estimated_document_count()
            ])
            
            total_collections = len(self.database.list_collection_names())
            
            # Get performance metrics from monitor
            perf_summary = self.performance_monitor.get_performance_summary(hours_back=1)
            
            # Connection metrics
            connection_info = server_status.get('connections', {})
            
            return DatabaseHealthMetrics(
                total_connections=connection_info.get('totalCreated', 0),
                active_connections=connection_info.get('current', 0),
                database_size_mb=round(db_stats.get('dataSize', 0) / (1024 * 1024), 2),
                total_documents=total_docs,
                total_collections=total_collections,
                index_count=db_stats.get('indexes', 0),
                average_query_time_ms=perf_summary.get('average_execution_time_ms', 0),
                slow_queries_count=perf_summary.get('slow_queries_count', 0),
                disk_usage_mb=round(db_stats.get('storageSize', 0) / (1024 * 1024), 2),
                memory_usage_mb=round(server_status.get('mem', {}).get('resident', 0), 2),
                uptime_hours=round(server_status.get('uptime', 0) / 3600, 2),
                timestamp=datetime.utcnow()
            )
            
        except PyMongoError as e:
            self.logger.error(f"Error getting database health metrics: {str(e)}")
            raise
    
    def _generate_animal_shelter_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate business insights from animal shelter data analysis."""
        insights = []
        
        total_animals = data.get("total_animals", 0)
        if total_animals == 0:
            return ["No animal data available for analysis"]
        
        breakdown = data.get("animal_breakdown", [])
        
        # Analyze animal types
        animal_types = {}
        adoption_rates = {}
        
        for item in breakdown:
            animal_type = item["animal_type"]
            outcome_type = item["outcome_type"]
            count = item["count"]
            
            if animal_type not in animal_types:
                animal_types[animal_type] = 0
            animal_types[animal_type] += count
            
            if outcome_type == "Adopted":
                if animal_type not in adoption_rates:
                    adoption_rates[animal_type] = 0
                adoption_rates[animal_type] += count
        
        # Generate insights
        most_common_animal = max(animal_types, key=animal_types.get) if animal_types else "Unknown"
        insights.append(f"Most common animal type: {most_common_animal} ({animal_types.get(most_common_animal, 0)} animals)")
        
        if adoption_rates:
            best_adoption_animal = max(adoption_rates, key=adoption_rates.get)
            insights.append(f"Highest adoption rate: {best_adoption_animal} ({adoption_rates[best_adoption_animal]} adopted)")
        
        avg_age = data.get("avg_age_overall", 0)
        if avg_age > 0:
            insights.append(f"Average age of animals: {avg_age:.1f} years")
        
        outcome_rate = data.get("avg_outcome_rate", 0)
        if outcome_rate > 0.7:
            insights.append("High overall outcome success rate indicates effective shelter management")
        elif outcome_rate < 0.3:
            insights.append("Low outcome rate suggests need for improved adoption programs")
        
        return insights
    
    def _generate_animal_shelter_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from animal shelter analysis."""
        recommendations = []
        
        breakdown = data.get("animal_breakdown", [])
        
        # Analyze for recommendations
        long_stay_animals = [item for item in breakdown if item.get("avg_days_in_shelter", 0) > 30]
        if long_stay_animals:
            recommendations.append("Focus on animals with extended shelter stays - implement targeted adoption campaigns")
        
        low_adoption_types = [item for item in breakdown if item.get("outcome_rate", 0) < 0.3]
        if low_adoption_types:
            animal_types = [item["animal_type"] for item in low_adoption_types]
            recommendations.append(f"Develop specialized programs for {', '.join(set(animal_types))} to improve adoption rates")
        
        high_age_animals = [item for item in breakdown if item.get("avg_age", 0) > 5]
        if high_age_animals:
            recommendations.append("Consider senior animal adoption programs to address older animal population")
        
        return recommendations
    
    def _generate_medical_notes_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate insights from medical notes analytics."""
        insights = []
        
        total_notes = data.get("total_notes", 0)
        if total_notes == 0:
            return ["No medical notes data available for analysis"]
        
        algorithms_perf = data.get("algorithms_performance", [])
        
        # Algorithm performance analysis
        if algorithms_perf:
            avg_times = {item["algorithm"]: item["avg_processing_time"] for item in algorithms_perf}
            fastest_algo = min(avg_times, key=avg_times.get) if avg_times else "Unknown"
            insights.append(f"Fastest algorithm: {fastest_algo} ({avg_times.get(fastest_algo, 0):.2f}ms average)")
            
            completeness_scores = {item["algorithm"]: item["soap_completeness"] for item in algorithms_perf}
            most_complete = max(completeness_scores, key=completeness_scores.get) if completeness_scores else "Unknown"
            insights.append(f"Most complete SOAP extraction: {most_complete} ({completeness_scores.get(most_complete, 0):.2f} score)")
        
        avg_processing_time = data.get("overall_avg_processing_time", 0)
        if avg_processing_time < 1:
            insights.append("Excellent processing performance with sub-millisecond average processing time")
        elif avg_processing_time > 100:
            insights.append("Processing time may need optimization - consider algorithm tuning")
        
        soap_completeness = data.get("overall_soap_completeness", 0)
        if soap_completeness > 0.8:
            insights.append("High SOAP section completeness indicates effective medical text processing")
        
        return insights
    
    def _generate_medical_notes_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for medical notes optimization."""
        recommendations = []
        
        algorithms_perf = data.get("algorithms_performance", [])
        
        # Algorithm optimization
        slow_algorithms = [item for item in algorithms_perf if item.get("avg_processing_time", 0) > 10]
        if slow_algorithms:
            algo_names = [item["algorithm"] for item in slow_algorithms]
            recommendations.append(f"Optimize performance for algorithms: {', '.join(set(algo_names))}")
        
        incomplete_soap = [item for item in algorithms_perf if item.get("soap_completeness", 0) < 0.5]
        if incomplete_soap:
            recommendations.append("Improve SOAP section extraction accuracy through enhanced training data")
        
        low_efficiency = [item for item in algorithms_perf if item.get("processing_efficiency", 0) < 10]
        if low_efficiency:
            recommendations.append("Consider algorithm selection optimization based on text characteristics")
        
        return recommendations
    
    def _prepare_animal_type_chart(self, breakdown: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for animal type distribution."""
        type_counts = defaultdict(int)
        for item in breakdown:
            type_counts[item["animal_type"]] += item["count"]
        
        return {
            "chart_type": "pie",
            "title": "Animal Type Distribution",
            "data": {
                "labels": list(type_counts.keys()),
                "values": list(type_counts.values())
            }
        }
    
    def _prepare_outcome_trends_chart(self, breakdown: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for outcome trends."""
        outcome_data = defaultdict(lambda: defaultdict(int))
        
        for item in breakdown:
            outcome_data[item["animal_type"]][item["outcome_type"]] += item["count"]
        
        return {
            "chart_type": "stacked_bar",
            "title": "Outcome Trends by Animal Type",
            "data": {
                "categories": list(outcome_data.keys()),
                "series": outcome_data
            }
        }
    
    def _prepare_age_demographics(self, breakdown: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for age demographics."""
        age_ranges = {"0-1": 0, "1-3": 0, "3-7": 0, "7+": 0}
        
        for item in breakdown:
            avg_age = item.get("avg_age", 0)
            count = item["count"]
            
            if avg_age <= 1:
                age_ranges["0-1"] += count
            elif avg_age <= 3:
                age_ranges["1-3"] += count
            elif avg_age <= 7:
                age_ranges["3-7"] += count
            else:
                age_ranges["7+"] += count
        
        return {
            "chart_type": "bar",
            "title": "Age Demographics",
            "data": {
                "labels": list(age_ranges.keys()),
                "values": list(age_ranges.values())
            }
        }
    
    def _prepare_algorithm_performance_chart(self, algorithms_perf: List[Dict]) -> Dict[str, Any]:
        """Prepare algorithm performance comparison chart."""
        algo_data = defaultdict(list)
        
        for item in algorithms_perf:
            algo_data["algorithms"].append(item["algorithm"])
            algo_data["processing_times"].append(item["avg_processing_time"])
            algo_data["completeness_scores"].append(item["soap_completeness"])
        
        return {
            "chart_type": "scatter",
            "title": "Algorithm Performance vs SOAP Completeness",
            "data": {
                "x_axis": "Processing Time (ms)",
                "y_axis": "SOAP Completeness Score", 
                "points": [
                    {
                        "name": algo,
                        "x": time,
                        "y": completeness
                    }
                    for algo, time, completeness in zip(
                        algo_data["algorithms"],
                        algo_data["processing_times"], 
                        algo_data["completeness_scores"]
                    )
                ]
            }
        }
    
    def _prepare_provider_efficiency_chart(self, algorithms_perf: List[Dict]) -> Dict[str, Any]:
        """Prepare provider efficiency analysis chart."""
        provider_data = defaultdict(lambda: {"notes": 0, "avg_time": 0})
        
        for item in algorithms_perf:
            provider = item.get("provider", "Unknown")
            provider_data[provider]["notes"] += item["note_count"]
            provider_data[provider]["avg_time"] += item["avg_processing_time"]
        
        return {
            "chart_type": "bubble",
            "title": "Provider Efficiency Analysis",
            "data": {
                "x_axis": "Number of Notes",
                "y_axis": "Average Processing Time",
                "bubbles": [
                    {
                        "name": provider,
                        "x": data["notes"],
                        "y": data["avg_time"],
                        "size": data["notes"] * 10
                    }
                    for provider, data in provider_data.items()
                ]
            }
        }
    
    def _prepare_soap_completeness_chart(self, algorithms_perf: List[Dict]) -> Dict[str, Any]:
        """Prepare SOAP completeness analysis chart."""
        return {
            "chart_type": "radar",
            "title": "SOAP Section Completeness by Algorithm",
            "data": {
                "categories": ["Subjective", "Objective", "Assessment", "Plan"],
                "series": [
                    {
                        "name": item["algorithm"],
                        "data": [item["soap_completeness"]] * 4  # Simplified for demo
                    }
                    for item in algorithms_perf[:5]  # Top 5 algorithms
                ]
            }
        }
    
    def export_analytics_data(self, report_type: str, format_type: str = "json") -> Dict[str, Any]:
        """
        Export analytics data in various formats for external analysis.
        
        Demonstrates data export capabilities and format handling.
        """
        try:
            if report_type == "animal_shelter":
                report = self.generate_animal_shelter_analytics()
            elif report_type == "medical_notes":
                report = self.generate_medical_notes_analytics()
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            if format_type == "json":
                return {
                    "format": "json",
                    "data": asdict(report),
                    "exported_at": datetime.utcnow().isoformat()
                }
            elif format_type == "csv":
                # For CSV, we'll export the metrics in tabular format
                return {
                    "format": "csv",
                    "data": self._convert_report_to_csv(report),
                    "exported_at": datetime.utcnow().isoformat()
                }
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
        
        except Exception as e:
            self.logger.error(f"Error exporting analytics data: {str(e)}")
            raise
    
    def _convert_report_to_csv(self, report: AnalyticsReport) -> str:
        """Convert analytics report to CSV format."""
        # Simplified CSV conversion for demonstration
        csv_lines = [
            "Report Type,Generated At,Metric,Value",
            f"{report.report_type},{report.generated_at.isoformat()},Total Records,{report.metrics.get('total_animals', report.metrics.get('total_notes', 0))}"
        ]
        
        # Add insights as CSV rows
        for i, insight in enumerate(report.insights):
            csv_lines.append(f"{report.report_type},{report.generated_at.isoformat()},Insight {i+1},{insight}")
        
        return "\n".join(csv_lines)
    
    def close(self):
        """Close database connections and cleanup resources."""
        if hasattr(self, 'client'):
            self.client.close()
            self.logger.info("Advanced Database Manager connections closed")


# Usage example and demonstration
def demonstrate_advanced_database_features():
    """
    Demonstration function showcasing advanced database capabilities.
    This function demonstrates the CS-499 Milestone Four database enhancements.
    """
    print("=" * 80)
    print("CS-499 MILESTONE FOUR: ADVANCED DATABASE CAPABILITIES")
    print("Analytics, Performance Monitoring, and Optimization")
    print("=" * 80)
    
    try:
        # Initialize advanced database manager
        db_manager = AdvancedDatabaseManager()
        
        print("\n1. DATABASE HEALTH METRICS")
        print("-" * 50)
        
        health_metrics = db_manager.get_database_health_metrics()
        print(f"Database Size: {health_metrics.database_size_mb} MB")
        print(f"Total Documents: {health_metrics.total_documents}")
        print(f"Total Collections: {health_metrics.total_collections}")
        print(f"Index Count: {health_metrics.index_count}")
        print(f"Average Query Time: {health_metrics.average_query_time_ms:.2f}ms")
        print(f"Active Connections: {health_metrics.active_connections}")
        
        print("\n2. ANIMAL SHELTER ANALYTICS")
        print("-" * 50)
        
        animal_report = db_manager.generate_animal_shelter_analytics(days_back=7)
        print(f"Report Generated: {animal_report.generated_at}")
        print(f"Total Animals Analyzed: {animal_report.metrics.get('total_animals', 0)}")
        print("Key Insights:")
        for insight in animal_report.insights[:3]:
            print(f"  • {insight}")
        
        print("\n3. MEDICAL NOTES ANALYTICS")
        print("-" * 50)
        
        medical_report = db_manager.generate_medical_notes_analytics(days_back=7)
        print(f"Total Medical Notes: {medical_report.metrics.get('total_notes', 0)}")
        print(f"Average Processing Time: {medical_report.metrics.get('avg_processing_time_ms', 0):.2f}ms")
        print(f"SOAP Completeness: {medical_report.metrics.get('soap_completeness_avg', 0):.2f}")
        print("Key Insights:")
        for insight in medical_report.insights[:3]:
            print(f"  • {insight}")
        
        print("\n4. PERFORMANCE MONITORING")
        print("-" * 50)
        
        perf_summary = db_manager.performance_monitor.get_performance_summary(hours_back=1)
        print(f"Total Queries (Last Hour): {perf_summary.get('total_queries', 0)}")
        print(f"Average Query Time: {perf_summary.get('average_execution_time_ms', 0):.2f}ms")
        print(f"Slow Queries: {perf_summary.get('slow_queries_count', 0)}")
        print(f"Index Usage: {perf_summary.get('index_usage', {}).get('used', 0)} used, {perf_summary.get('index_usage', {}).get('not_used', 0)} not used")
        
        optimization_recs = perf_summary.get('optimization_recommendations', [])
        if optimization_recs:
            print("Optimization Recommendations:")
            for rec in optimization_recs[:3]:
                print(f"  • {rec}")
        
        print("\n5. DATA EXPORT CAPABILITIES")
        print("-" * 50)
        
        export_result = db_manager.export_analytics_data("animal_shelter", "json")
        print(f"Export Format: {export_result['format']}")
        print(f"Export Size: {len(str(export_result['data']))} characters")
        print(f"Exported At: {export_result['exported_at']}")
        
        print("\n6. ADVANCED FEATURES DEMONSTRATED")
        print("-" * 50)
        print("✅ Complex MongoDB Aggregation Pipelines")
        print("✅ Real-time Performance Monitoring")
        print("✅ Advanced Indexing Strategies")
        print("✅ Business Intelligence Analytics")
        print("✅ Data Export/Import Capabilities")
        print("✅ Database Health Monitoring")
        print("✅ Query Optimization Recommendations")
        print("✅ Connection Pooling & Resource Management")
        
        # Cleanup
        db_manager.close()
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
    
    print("\n" + "=" * 80)
    print("ADVANCED DATABASE DEMONSTRATION COMPLETE")
    print("Professional database management system with analytics and optimization")
    print("=" * 80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    demonstrate_advanced_database_features()