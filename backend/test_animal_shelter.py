"""
Unit Tests for AnimalShelter Enhanced Repository
Demonstrates validation, guarded deletes, and edge case handling.
"""

import pytest
import os
import logging
from datetime import datetime
from unittest.mock import patch, MagicMock
from AnimalShelter_enhanced import AnimalShelterRepository

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestAnimalShelterRepository:
    """Test suite for enhanced Animal Shelter repository."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Mock MongoDB client to avoid actual database connections in tests."""
        with patch('AnimalShelter_enhanced.MongoClient') as mock_client:
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_client.return_value = mock_client
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            yield mock_client, mock_db, mock_collection

    @pytest.fixture
    def repository(self, mock_mongo_client):
        """Create repository instance with mocked MongoDB."""
        mock_client, mock_db, mock_collection = mock_mongo_client
        repo = AnimalShelterRepository(mongo_uri="mongodb://test", db_name="test_db")
        repo.collection = mock_collection
        return repo, mock_collection

    def test_input_validation_require_dict(self, repository):
        """Test input validation for dictionary requirement."""
        repo, _ = repository
        
        # Valid input should pass
        valid_data = {"name": "Buddy", "animal_type": "dog"}
        result = repo._require_dict(valid_data, "test_data")
        assert result == valid_data
        
        # Invalid inputs should raise ValueError
        with pytest.raises(ValueError, match="test_data must be a dictionary"):
            repo._require_dict("not a dict", "test_data")
        
        with pytest.raises(ValueError, match="test_data must be a dictionary"):
            repo._require_dict(123, "test_data")

    def test_input_validation_required_fields(self, repository):
        """Test validation of required fields."""
        repo, _ = repository
        
        # Valid data with all required fields
        valid_data = {"name": "Buddy", "animal_type": "dog", "breed": "Golden Retriever"}
        repo._validate_required_fields(valid_data, ["name", "animal_type"])
        # Should not raise exception
        
        # Missing required fields should raise ValueError
        invalid_data = {"breed": "Golden Retriever"}
        with pytest.raises(ValueError, match="Missing required fields: \\['name', 'animal_type'\\]"):
            repo._validate_required_fields(invalid_data, ["name", "animal_type"])

    def test_create_happy_path(self, repository):
        """Test successful animal record creation."""
        repo, mock_collection = repository
        
        # Mock successful insertion
        mock_result = MagicMock()
        mock_result.inserted_id = "mock_object_id"
        mock_collection.insert_one.return_value = mock_result
        mock_collection.find_one.return_value = {
            "_id": "mock_object_id",
            "name": "Buddy",
            "animal_type": "dog",
            "created_at": datetime.utcnow()
        }
        
        # Test creation
        animal_data = {"name": "Buddy", "animal_type": "dog"}
        result = repo.create(animal_data, user_id="test_user")
        
        assert result["ok"] is True
        assert result["error"] is None
        assert result["data"]["name"] == "Buddy"
        assert result["data"]["animal_type"] == "dog"
        assert "_id" in result["data"]

    def test_create_validation_error(self, repository):
        """Test create with validation errors."""
        repo, _ = repository
        
        # Test with invalid input type
        result = repo.create("not a dict")
        assert result["ok"] is False
        assert "must be a dictionary" in result["error"]
        
        # Test with missing required fields
        result = repo.create({"breed": "Golden Retriever"})
        assert result["ok"] is False
        assert "Missing required fields" in result["error"]

    def test_safe_delete_guard(self, repository):
        """Test safe delete functionality to prevent accidental mass deletion."""
        repo, mock_collection = repository
        
        # Test 1: Empty query without confirmation should fail
        result = repo.delete({})
        assert result["ok"] is False
        assert "Empty filter requires confirm_empty_filter=True" in result["error"]
        
        # Test 2: Empty query with confirmation should work
        mock_collection.find.return_value = [{"_id": "test_id"}]
        mock_result = MagicMock()
        mock_result.deleted_count = 5
        mock_collection.delete_many.return_value = mock_result
        
        result = repo.delete({}, confirm_empty_filter=True, user_id="admin")
        assert result["ok"] is True
        assert result["data"]["deleted_count"] == 5
        
        # Test 3: Non-empty query should work without confirmation
        query = {"animal_type": "cat"}
        mock_collection.find.return_value = [{"_id": "cat_id"}]
        mock_result.deleted_count = 3
        
        result = repo.delete(query, user_id="test_user")
        assert result["ok"] is True
        assert result["data"]["deleted_count"] == 3

    def test_safe_delete_validation(self, repository):
        """Test delete method input validation."""
        repo, _ = repository
        
        # Test with invalid query type
        result = repo.delete("not a dict")
        assert result["ok"] is False
        assert "must be a dictionary" in result["error"]

    def test_update_validation(self, repository):
        """Test update method validation."""
        repo, _ = repository
        
        # Test with empty query
        result = repo.update({}, {"name": "Updated"})
        assert result["ok"] is False
        assert "Query cannot be empty for update" in result["error"]
        
        # Test with invalid input types
        result = repo.update("not a dict", {"name": "Updated"})
        assert result["ok"] is False
        assert "must be a dictionary" in result["error"]

    def test_read_happy_path(self, repository):
        """Test successful read operation."""
        repo, mock_collection = repository
        
        # Mock database response
        mock_collection.find.return_value.limit.return_value = [
            {"_id": "id1", "name": "Buddy", "animal_type": "dog"},
            {"_id": "id2", "name": "Fluffy", "animal_type": "cat"}
        ]
        
        result = repo.read({"animal_type": "dog"})
        assert result["ok"] is True
        assert len(result["data"]) == 2
        assert result["data"][0]["name"] == "Buddy"

    def test_audit_trail_functionality(self, repository):
        """Test audit trail logging."""
        repo, _ = repository
        
        # Mock audits collection
        mock_audits_collection = MagicMock()
        repo.database = MagicMock()
        repo.database.__getitem__.return_value = mock_audits_collection
        
        # Test audit logging
        repo._audit("create", "test_id", {"user_id": "test_user"})
        
        # Verify audit entry was created
        mock_audits_collection.insert_one.assert_called_once()
        audit_call = mock_audits_collection.insert_one.call_args[0][0]
        assert audit_call["action"] == "create"
        assert audit_call["target_id"] == "test_id"
        assert audit_call["meta"]["user_id"] == "test_user"

    def test_get_audit_trail(self, repository):
        """Test audit trail retrieval."""
        repo, _ = repository
        
        # Mock audits collection and response
        mock_audits_collection = MagicMock()
        repo.database = {"audits": mock_audits_collection}
        
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [
            {
                "_id": "audit_id",
                "action": "create",
                "target_id": "test_id",
                "timestamp": datetime.utcnow(),
                "meta": {"user_id": "test_user"}
            }
        ]
        mock_audits_collection.find.return_value = mock_cursor
        
        result = repo.get_audit_trail(target_id="test_id")
        assert result["ok"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["action"] == "create"

    def test_consistent_return_envelopes(self, repository):
        """Test that all methods return consistent envelope format."""
        repo, mock_collection = repository
        
        # Mock database errors
        mock_collection.insert_one.side_effect = Exception("Database error")
        mock_collection.find.side_effect = Exception("Database error")
        mock_collection.update_many.side_effect = Exception("Database error")
        mock_collection.delete_many.side_effect = Exception("Database error")
        
        # Test all methods return consistent error envelopes
        methods_and_args = [
            (repo.create, [{"name": "Test", "animal_type": "dog"}]),
            (repo.read, [{"name": "Test"}]),
            (repo.update, [{"name": "Test"}, {"breed": "Updated"}]),
            (repo.delete, [{"name": "Test"}])
        ]
        
        for method, args in methods_and_args:
            result = method(*args)
            assert isinstance(result, dict)
            assert "ok" in result
            assert "data" in result
            assert "error" in result
            assert result["ok"] is False
            assert result["data"] is None
            assert result["error"] is not None

    def test_environment_configuration(self):
        """Test environment-based configuration."""
        # Test with environment variable
        with patch.dict(os.environ, {'MONGO_URI': 'mongodb://env-test:27017/'}):
            with patch('AnimalShelter_enhanced.MongoClient') as mock_client:
                repo = AnimalShelterRepository()
                mock_client.assert_called_with('mongodb://env-test:27017/')
        
        # Test with explicit parameter
        with patch('AnimalShelter_enhanced.MongoClient') as mock_client:
            repo = AnimalShelterRepository(mongo_uri='mongodb://explicit:27017/')
            mock_client.assert_called_with('mongodb://explicit:27017/')


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets and pagination."""
        with patch('AnimalShelter_enhanced.MongoClient') as mock_client:
            repo = AnimalShelterRepository()
            mock_collection = MagicMock()
            repo.collection = mock_collection
            
            # Test default limit is applied
            repo.read()
            mock_collection.find.return_value.limit.assert_called_with(100)
            
            # Test custom limit
            repo.read(limit=50)
            mock_collection.find.return_value.limit.assert_called_with(50)

    def test_connection_failure_handling(self):
        """Test graceful handling of connection failures."""
        with patch('AnimalShelter_enhanced.MongoClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                AnimalShelterRepository()

    def test_id_normalization(self):
        """Test ObjectId to string conversion."""
        with patch('AnimalShelter_enhanced.MongoClient'):
            repo = AnimalShelterRepository()
            
            # Test with ObjectId
            test_data = {"_id": "mock_object_id", "name": "Test"}
            normalized = repo._normalize_id(test_data)
            assert normalized["_id"] == "mock_object_id"
            
            # Test with None
            assert repo._normalize_id(None) is None
            
            # Test without _id
            test_data = {"name": "Test"}
            normalized = repo._normalize_id(test_data)
            assert normalized == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])