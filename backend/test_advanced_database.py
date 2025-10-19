"""
Unit Tests for Advanced Database Manager - CS-499 Milestone Four
Comprehensive testing for database analytics, performance monitoring, and optimization.

This test suite validates:
1. Complex aggregation pipeline functionality
2. Performance monitoring and metrics collection
3. Database health monitoring
4. Analytics report generation
5. Data export/import capabilities
6. Query optimization recommendations
7. Index management and performance
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from AdvancedDatabaseManager import (
    AdvancedDatabaseManager, PerformanceMonitor, QueryPerformanceMetrics,
    DatabaseHealthMetrics, AnalyticsReport, QueryType, AccessLevel,
    demonstrate_advanced_database_features
)

class TestAdvancedDatabaseManager:
    """Test suite for Advanced Database Manager functionality."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Mock MongoDB client to avoid actual database connections in tests."""
        with patch('AdvancedDatabaseManager.MongoClient') as mock_client:
            mock_db = MagicMock()
            mock_collection = MagicMock()
            
            # Setup mock client structure
            mock_client.return_value = mock_client
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_db.list_collection_names.return_value = ['animals', 'medical_notes', 'audits']
            mock_db.command.return_value = {
                'dataSize': 1048576,  # 1MB
                'storageSize': 2097152,  # 2MB
                'indexes': 10
            }
            
            # Mock server status
            mock_client.admin.command.return_value = {
                'connections': {'current': 5, 'totalCreated': 100},
                'mem': {'resident': 256},
                'uptime': 86400  # 24 hours
            }
            
            # Mock collection methods
            mock_collection.estimated_document_count.return_value = 1000
            mock_collection.create_indexes.return_value = None
            mock_collection.aggregate.return_value = []
            
            yield mock_client, mock_db, mock_collection

    @pytest.fixture
    def db_manager(self, mock_mongo_client):
        """Create database manager instance with mocked MongoDB."""
        mock_client, mock_db, mock_collection = mock_mongo_client
        manager = AdvancedDatabaseManager("mongodb://test", "test_db")
        return manager, mock_collection

    def test_initialization(self, db_manager):
        """Test database manager initialization."""
        manager, mock_collection = db_manager
        
        assert manager.db_name == "test_db"
        assert hasattr(manager, 'performance_monitor')
        assert hasattr(manager, 'animals_collection')
        assert hasattr(manager, 'medical_notes_collection')
        assert hasattr(manager, 'audits_collection')

    def test_database_health_metrics(self, db_manager):
        """Test database health metrics collection."""
        manager, _ = db_manager
        
        health_metrics = manager.get_database_health_metrics()
        
        assert isinstance(health_metrics, DatabaseHealthMetrics)
        assert health_metrics.database_size_mb == 1.0  # 1MB from mock
        assert health_metrics.disk_usage_mb == 2.0    # 2MB from mock
        assert health_metrics.total_collections == 3
        assert health_metrics.index_count == 10
        assert health_metrics.active_connections == 5
        assert health_metrics.memory_usage_mb == 256
        assert health_metrics.uptime_hours == 24.0

    def test_animal_shelter_analytics_empty_data(self, db_manager):
        """Test animal shelter analytics with empty dataset."""
        manager, mock_collection = db_manager
        
        # Mock empty aggregation result
        manager.animals_collection.aggregate.return_value = []
        
        report = manager.generate_animal_shelter_analytics(days_back=7)
        
        assert isinstance(report, AnalyticsReport)
        assert report.report_type == "animal_shelter"
        assert report.report_name == "Animal Shelter Analytics"
        assert len(report.insights) > 0
        assert "No data available" in report.insights[0]

    def test_animal_shelter_analytics_with_data(self, db_manager):
        """Test animal shelter analytics with sample data."""
        manager, _ = db_manager
        
        # Mock aggregation result with sample data
        sample_result = [{
            "total_animals": 150,
            "animal_breakdown": [
                {
                    "animal_type": "dog",
                    "outcome_type": "Adopted",
                    "count": 45,
                    "avg_age": 3.2,
                    "avg_days_in_shelter": 12.5,
                    "breed_diversity": 8,
                    "outcome_rate": 1.0
                },
                {
                    "animal_type": "cat", 
                    "outcome_type": "Available",
                    "count": 30,
                    "avg_age": 2.8,
                    "avg_days_in_shelter": 18.3,
                    "breed_diversity": 5,
                    "outcome_rate": 0.0
                }
            ],
            "avg_age_overall": 3.0,
            "avg_outcome_rate": 0.6
        }]
        
        manager.animals_collection.aggregate.return_value = sample_result
        
        report = manager.generate_animal_shelter_analytics(days_back=7)
        
        assert report.metrics["total_animals"] == 150
        assert report.metrics["average_age"] == 3.0
        assert report.metrics["overall_outcome_rate"] == 0.6
        assert len(report.insights) > 0
        assert "dog" in report.insights[0].lower()
        assert len(report.recommendations) >= 0
        assert "animal_type_distribution" in report.visualizations

    def test_medical_notes_analytics_with_data(self, db_manager):
        """Test medical notes analytics with sample data."""
        manager, _ = db_manager
        
        # Mock aggregation result for medical notes
        sample_result = [{
            "total_notes": 75,
            "unique_providers": ["provider_1", "provider_2", "provider_3"],
            "note_types": ["progress_note", "consultation"],
            "algorithms_performance": [
                {
                    "algorithm": "hybrid",
                    "provider": "provider_1",
                    "note_type": "progress_note",
                    "note_count": 25,
                    "avg_processing_time": 0.85,
                    "processing_efficiency": 15.2,
                    "soap_completeness": 0.92,
                    "avg_entities": 8.3
                },
                {
                    "algorithm": "trie_based",
                    "provider": "provider_2", 
                    "note_type": "consultation",
                    "note_count": 20,
                    "avg_processing_time": 1.15,
                    "processing_efficiency": 12.8,
                    "soap_completeness": 0.87,
                    "avg_entities": 6.9
                }
            ],
            "overall_avg_processing_time": 1.0,
            "overall_efficiency": 14.0,
            "overall_soap_completeness": 0.895
        }]
        
        manager.medical_notes_collection.aggregate.return_value = sample_result
        
        report = manager.generate_medical_notes_analytics(days_back=7)
        
        assert report.metrics["total_notes"] == 75
        assert report.metrics["unique_providers"] == 3
        assert report.metrics["note_types"] == 2
        assert report.metrics["avg_processing_time_ms"] == 1.0
        assert report.metrics["soap_completeness_avg"] == 0.9
        assert len(report.insights) > 0
        assert "hybrid" in report.insights[0].lower()

    def test_export_analytics_data_json(self, db_manager):
        """Test analytics data export in JSON format."""
        manager, _ = db_manager
        
        # Mock empty data for simplicity
        manager.animals_collection.aggregate.return_value = []
        
        export_result = manager.export_analytics_data("animal_shelter", "json")
        
        assert export_result["format"] == "json"
        assert "data" in export_result
        assert "exported_at" in export_result
        assert isinstance(export_result["data"], dict)

    def test_export_analytics_data_csv(self, db_manager):
        """Test analytics data export in CSV format.""" 
        manager, _ = db_manager
        
        # Mock empty data
        manager.animals_collection.aggregate.return_value = []
        
        export_result = manager.export_analytics_data("animal_shelter", "csv")
        
        assert export_result["format"] == "csv"
        assert isinstance(export_result["data"], str)
        assert "Report Type,Generated At,Metric,Value" in export_result["data"]

    def test_export_unsupported_report_type(self, db_manager):
        """Test export with unsupported report type."""
        manager, _ = db_manager
        
        with pytest.raises(ValueError, match="Unsupported report type"):
            manager.export_analytics_data("unsupported_type", "json")

    def test_export_unsupported_format(self, db_manager):
        """Test export with unsupported format type."""
        manager, _ = db_manager
        
        manager.animals_collection.aggregate.return_value = []
        
        with pytest.raises(ValueError, match="Unsupported format type"):
            manager.export_analytics_data("animal_shelter", "xml")

class TestPerformanceMonitor:
    """Test suite for Performance Monitor functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance."""
        return PerformanceMonitor(max_metrics_history=100)

    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert len(performance_monitor.query_metrics) == 0
        assert len(performance_monitor.failed_queries) == 0
        assert performance_monitor.max_metrics_history == 100
        assert performance_monitor.slow_query_threshold_ms == 100

    def test_record_query_performance(self, performance_monitor):
        """Test recording query performance metrics."""
        performance_monitor.record_query_performance(
            command_name="find",
            collection_name="test_collection",
            execution_time_ms=50.5,
            documents_examined=100,
            documents_returned=25
        )
        
        assert len(performance_monitor.query_metrics) == 1
        
        metric = performance_monitor.query_metrics[0]
        assert metric.query_type == QueryType.FIND
        assert metric.collection_name == "test_collection"
        assert metric.execution_time_ms == 50.5
        assert metric.documents_examined == 100
        assert metric.documents_returned == 25

    def test_record_slow_query(self, performance_monitor):
        """Test slow query detection and recommendation generation."""
        # Record a slow query
        performance_monitor.record_query_performance(
            command_name="find",
            collection_name="slow_collection",
            execution_time_ms=250.0,  # Above 100ms threshold
            documents_examined=1000,
            documents_returned=10
        )
        
        assert len(performance_monitor.query_metrics) == 1
        assert len(performance_monitor.optimization_recommendations) > 0
        
        # Check if recommendation was generated for inefficient query
        recommendations = performance_monitor.optimization_recommendations
        assert any("inefficiency detected" in rec for rec in recommendations)

    def test_record_failed_query(self, performance_monitor):
        """Test recording failed queries."""
        performance_monitor.record_failed_query(
            command_name="update",
            execution_time_ms=75.0,
            failure_reason="Index not found"
        )
        
        assert len(performance_monitor.failed_queries) == 1
        
        failed_query = performance_monitor.failed_queries[0]
        assert failed_query["command"] == "update"
        assert failed_query["execution_time_ms"] == 75.0
        assert failed_query["failure_reason"] == "Index not found"

    def test_performance_summary_empty(self, performance_monitor):
        """Test performance summary with no metrics."""
        summary = performance_monitor.get_performance_summary(hours_back=1)
        
        assert "No metrics available" in summary["message"]

    def test_performance_summary_with_metrics(self, performance_monitor):
        """Test performance summary with sample metrics."""
        # Add some sample metrics
        test_metrics = [
            (QueryType.FIND, "collection1", 25.0, 50, 10),
            (QueryType.UPDATE, "collection1", 45.0, 30, 5),
            (QueryType.FIND, "collection2", 120.0, 200, 20),  # Slow query
            (QueryType.INSERT, "collection2", 15.0, 0, 1),
        ]
        
        for query_type, collection, time_ms, examined, returned in test_metrics:
            performance_monitor.record_query_performance(
                command_name=query_type.value,
                collection_name=collection,
                execution_time_ms=time_ms,
                documents_examined=examined,
                documents_returned=returned
            )
        
        summary = performance_monitor.get_performance_summary(hours_back=1)
        
        assert summary["total_queries"] == 4
        assert summary["average_execution_time_ms"] == 51.25  # (25+45+120+15)/4
        assert summary["slow_queries_count"] == 1  # Only the 120ms query
        assert summary["slow_queries_percentage"] == 25.0  # 1/4 * 100
        
        # Check query type breakdown
        assert "find" in summary["query_types"]
        assert summary["query_types"]["find"]["count"] == 2
        
        # Check collection breakdown
        assert "collection1" in summary["collection_performance"]
        assert "collection2" in summary["collection_performance"]

    def test_metrics_history_limit(self, performance_monitor):
        """Test that metrics history is properly limited."""
        performance_monitor.max_metrics_history = 5
        
        # Add more metrics than the limit
        for i in range(10):
            performance_monitor.record_query_performance(
                command_name="find",
                collection_name=f"collection_{i}",
                execution_time_ms=10.0 + i
            )
        
        # Should only keep the last 5 metrics
        assert len(performance_monitor.query_metrics) == 5
        
        # Check that it kept the most recent ones
        collection_names = [m.collection_name for m in performance_monitor.query_metrics]
        assert "collection_5" in collection_names
        assert "collection_9" in collection_names
        assert "collection_0" not in collection_names

    def test_concurrent_metrics_recording(self, performance_monitor):
        """Test thread safety of performance metrics recording."""
        def record_metrics(thread_id):
            for i in range(10):
                performance_monitor.record_query_performance(
                    command_name="find",
                    collection_name=f"thread_{thread_id}_collection_{i}",
                    execution_time_ms=10.0 + i
                )
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 50 total metrics (5 threads × 10 metrics each)
        assert len(performance_monitor.query_metrics) == 50
        
        # Verify no data corruption
        collection_names = [m.collection_name for m in performance_monitor.query_metrics]
        assert len(set(collection_names)) == 50  # All should be unique

class TestDataStructures:
    """Test suite for data structures and models."""
    
    def test_query_performance_metrics_creation(self):
        """Test QueryPerformanceMetrics data structure."""
        timestamp = datetime.utcnow()
        
        metric = QueryPerformanceMetrics(
            query_type=QueryType.AGGREGATE,
            collection_name="test_collection",
            execution_time_ms=75.5,
            documents_examined=500,
            documents_returned=50,
            index_used=True,
            index_name="test_index",
            timestamp=timestamp,
            query_shape="aggregate([{$match: {...}}])"
        )
        
        assert metric.query_type == QueryType.AGGREGATE
        assert metric.collection_name == "test_collection"
        assert metric.execution_time_ms == 75.5
        assert metric.documents_examined == 500
        assert metric.documents_returned == 50
        assert metric.index_used is True
        assert metric.index_name == "test_index"
        assert metric.timestamp == timestamp

    def test_database_health_metrics_creation(self):
        """Test DatabaseHealthMetrics data structure."""
        timestamp = datetime.utcnow()
        
        health_metrics = DatabaseHealthMetrics(
            total_connections=100,
            active_connections=15,
            database_size_mb=256.5,
            total_documents=10000,
            total_collections=8,
            index_count=25,
            average_query_time_ms=45.3,
            slow_queries_count=3,
            disk_usage_mb=512.8,
            memory_usage_mb=128.0,
            uptime_hours=72.5,
            timestamp=timestamp
        )
        
        assert health_metrics.total_connections == 100
        assert health_metrics.active_connections == 15
        assert health_metrics.database_size_mb == 256.5
        assert health_metrics.total_documents == 10000
        assert health_metrics.uptime_hours == 72.5

    def test_analytics_report_creation(self):
        """Test AnalyticsReport data structure."""
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        
        report = AnalyticsReport(
            report_type="test_report",
            report_name="Test Analytics Report",
            generated_at=end_time,
            data_period=(start_time, end_time),
            metrics={"total_records": 1000, "avg_score": 85.5},
            insights=["High performance observed", "Data quality is good"],
            recommendations=["Continue current approach", "Monitor trends"],
            visualizations={"chart1": {"type": "bar", "data": [1, 2, 3]}}
        )
        
        assert report.report_type == "test_report"
        assert report.report_name == "Test Analytics Report"
        assert report.metrics["total_records"] == 1000
        assert len(report.insights) == 2
        assert len(report.recommendations) == 2
        assert "chart1" in report.visualizations

class TestIntegration:
    """Integration tests for advanced database features."""
    
    def test_complete_analytics_workflow(self):
        """Test complete analytics workflow integration."""
        with patch('AdvancedDatabaseManager.MongoClient') as mock_client:
            # Setup comprehensive mock
            mock_db = MagicMock()
            mock_client.return_value = mock_client
            mock_client.__getitem__.return_value = mock_db
            
            # Mock database stats
            mock_db.command.return_value = {'dataSize': 1048576, 'storageSize': 2097152, 'indexes': 15}
            mock_client.admin.command.return_value = {
                'connections': {'current': 8, 'totalCreated': 150},
                'mem': {'resident': 384},
                'uptime': 172800  # 48 hours
            }
            
            # Mock collections
            mock_animals = MagicMock()
            mock_medical = MagicMock()
            mock_db.__getitem__.side_effect = lambda name: {
                'animals': mock_animals,
                'medical_notes': mock_medical,
                'audits': MagicMock(),
                'analytics_cache': MagicMock()
            }.get(name, MagicMock())
            
            mock_animals.estimated_document_count.return_value = 500
            mock_medical.estimated_document_count.return_value = 300
            mock_db.list_collection_names.return_value = ['animals', 'medical_notes', 'audits', 'analytics_cache']
            
            # Initialize manager
            manager = AdvancedDatabaseManager("mongodb://test", "integration_test")
            
            # Test health metrics
            health = manager.get_database_health_metrics()
            assert health.database_size_mb == 1.0
            assert health.total_documents >= 800  # 500 + 300 from collections
            
            # Test analytics with mock data
            mock_animals.aggregate.return_value = [{
                "total_animals": 100,
                "animal_breakdown": [],
                "avg_age_overall": 3.5,
                "avg_outcome_rate": 0.75
            }]
            
            animal_report = manager.generate_animal_shelter_analytics()
            assert animal_report.report_type == "animal_shelter"
            assert animal_report.metrics["total_animals"] == 100
            
            # Test export functionality
            export_result = manager.export_analytics_data("animal_shelter", "json")
            assert export_result["format"] == "json"
            assert "data" in export_result
            
            manager.close()

def test_demonstration_function():
    """Test the demonstration function runs without errors."""
    with patch('AdvancedDatabaseManager.AdvancedDatabaseManager') as mock_manager_class:
        # Setup mock manager
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Mock all required methods
        mock_manager.get_database_health_metrics.return_value = MagicMock(
            database_size_mb=10.5,
            total_documents=1500,
            total_collections=4,
            index_count=20,
            average_query_time_ms=25.3,
            active_connections=12
        )
        
        mock_manager.generate_animal_shelter_analytics.return_value = MagicMock(
            generated_at=datetime.utcnow(),
            metrics={"total_animals": 250},
            insights=["Test insight 1", "Test insight 2"]
        )
        
        mock_manager.generate_medical_notes_analytics.return_value = MagicMock(
            metrics={"total_notes": 150, "avg_processing_time_ms": 1.2, "soap_completeness_avg": 0.89},
            insights=["Medical insight 1", "Medical insight 2"]
        )
        
        mock_manager.performance_monitor.get_performance_summary.return_value = {
            "total_queries": 1000,
            "average_execution_time_ms": 15.7,
            "slow_queries_count": 5,
            "index_usage": {"used": 950, "not_used": 50},
            "optimization_recommendations": ["Test recommendation"]
        }
        
        mock_manager.export_analytics_data.return_value = {
            "format": "json",
            "data": {"test": "data"},
            "exported_at": datetime.utcnow().isoformat()
        }
        
        # Test the demonstration runs without errors
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            demonstrate_advanced_database_features()
            
            output = buffer.getvalue()
            
            # Verify key elements are present in output
            assert "CS-499 MILESTONE FOUR" in output
            assert "DATABASE HEALTH METRICS" in output
            assert "ANIMAL SHELTER ANALYTICS" in output
            assert "MEDICAL NOTES ANALYTICS" in output
            assert "PERFORMANCE MONITORING" in output
            assert "DATA EXPORT CAPABILITIES" in output
            assert "ADVANCED FEATURES DEMONSTRATED" in output
            assert "Complex MongoDB Aggregation Pipelines" in output
            
        finally:
            sys.stdout = old_stdout

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])