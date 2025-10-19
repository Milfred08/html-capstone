"""
Enhanced AnimalShelter Repository Layer (CS-499)
Professional repository pattern with validation, logging, and safety features.
Demonstrates software engineering best practices and clean architecture.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pymongo import MongoClient, IndexModel
from pymongo.errors import PyMongoError, DuplicateKeyError
import uuid

class AnimalShelterRepository:
    """
    Enhanced repository layer for Animal Shelter data management.
    
    Features:
    - Environment-based configuration
    - Input validation and normalized returns
    - Safe delete operations with confirmation
    - Structured logging and audit trail
    - Index management helpers
    - Consistent error handling
    """
    
    def __init__(self, mongo_uri: Optional[str] = None, db_name: str = "shelter", collection_name: str = "animals"):
        """
        Initialize the repository with environment-based configuration.
        
        Args:
            mongo_uri: MongoDB connection string (defaults to MONGO_URI env var)
            db_name: Database name
            collection_name: Collection name
        """
        self.logger = logging.getLogger("animalshelter")
        
        # Environment-based configuration - no secrets in code
        self.mongo_uri = mongo_uri or os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
        self.db_name = db_name
        self.collection_name = collection_name
        
        try:
            self.client = MongoClient(self.mongo_uri)
            self.database = self.client[self.db_name]
            self.collection = self.database[self.collection_name]
            self._ensure_indexes()
            self.logger.info(f"Connected to MongoDB: {self.db_name}.{self.collection_name}")
        except PyMongoError as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def _require_dict(self, input_arg: Any, arg_name: str) -> Dict[str, Any]:
        """Validate that input is a dictionary."""
        if not isinstance(input_arg, dict):
            raise ValueError(f"{arg_name} must be a dictionary, got {type(input_arg)}")
        return input_arg

    def _validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """Validate that required fields are present in data."""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    def _normalize_id(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MongoDB ObjectId to string for consistent JSON serialization."""
        if data and '_id' in data:
            data['_id'] = str(data['_id'])
        
        # Also handle datetime serialization
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data

    def _audit(self, action: str, target_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Audit trail helper for tracking operations.
        
        Args:
            action: Type of operation (create, read, update, delete)
            target_id: ID of the affected document
            meta: Additional metadata (user, timestamp, etc.)
        """
        audit_entry = {
            "action": action,
            "target_id": target_id,
            "timestamp": datetime.utcnow(),
            "meta": meta or {}
        }
        
        try:
            # Store in audits collection
            audits_collection = self.database["audits"]
            audits_collection.insert_one(audit_entry)
            self.logger.info(f"Audit logged: {action} on {target_id}")
        except PyMongoError as e:
            self.logger.error(f"Failed to log audit entry: {str(e)}")

    def _ensure_indexes(self) -> None:
        """Create helpful indexes for performance."""
        try:
            # Common query indexes
            indexes = [
                IndexModel([("animal_type", 1)]),
                IndexModel([("breed", 1)]),
                IndexModel([("outcome_type", 1)]),
                IndexModel([("date_of_birth", 1)]),
                IndexModel([("created_at", -1)])  # For recent records
            ]
            
            # Create indexes if they don't exist
            self.collection.create_indexes(indexes)
            self.logger.info("Database indexes ensured")
        except PyMongoError as e:
            self.logger.warning(f"Failed to create indexes: {str(e)}")

    def create(self, data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new animal record.
        
        Args:
            data: Animal data dictionary
            user_id: ID of user performing the operation
            
        Returns:
            {"ok": bool, "data": created_record, "error": str|None}
        """
        try:
            # Input validation
            data = self._require_dict(data, "data")
            self._validate_required_fields(data, ["name", "animal_type"])
            
            # Add metadata
            data["created_at"] = datetime.utcnow()
            data["updated_at"] = datetime.utcnow()
            if not data.get("id"):
                data["id"] = str(uuid.uuid4())
            
            # Insert record
            result = self.collection.insert_one(data.copy())
            created_record = self.collection.find_one({"_id": result.inserted_id})
            created_record = self._normalize_id(created_record)
            
            # Audit log
            self._audit("create", str(result.inserted_id), {"user_id": user_id})
            
            self.logger.info(f"Created animal record: {result.inserted_id}")
            return {"ok": True, "data": created_record, "error": None}
            
        except ValueError as e:
            self.logger.error(f"Validation error in create: {str(e)}")
            return {"ok": False, "data": None, "error": str(e)}
        except DuplicateKeyError as e:
            self.logger.error(f"Duplicate key error in create: {str(e)}")
            return {"ok": False, "data": None, "error": "Record already exists"}
        except PyMongoError as e:
            self.logger.error(f"Database error in create: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def read(self, query: Optional[Dict[str, Any]] = None, limit: int = 100) -> Dict[str, Any]:
        """
        Read animal records matching the query.
        
        Args:
            query: MongoDB query filter (empty dict for all records)
            limit: Maximum number of records to return
            
        Returns:
            {"ok": bool, "data": [records], "error": str|None}
        """
        try:
            query = query or {}
            if query:
                query = self._require_dict(query, "query")
            
            cursor = self.collection.find(query).limit(limit)
            records = [self._normalize_id(doc) for doc in cursor]
            
            self.logger.info(f"Read {len(records)} animal records")
            return {"ok": True, "data": records, "error": None}
            
        except ValueError as e:
            self.logger.error(f"Validation error in read: {str(e)}")
            return {"ok": False, "data": None, "error": str(e)}
        except PyMongoError as e:
            self.logger.error(f"Database error in read: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def update(self, query: Dict[str, Any], new_values: Dict[str, Any], 
               user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update animal records matching the query.
        
        Args:
            query: MongoDB query filter
            new_values: Fields to update
            user_id: ID of user performing the operation
            
        Returns:
            {"ok": bool, "data": {"modified_count": int}, "error": str|None}
        """
        try:
            # Input validation
            query = self._require_dict(query, "query")
            new_values = self._require_dict(new_values, "new_values")
            
            if not query:
                return {"ok": False, "data": None, "error": "Query cannot be empty for update"}
            
            # Add update metadata
            new_values["updated_at"] = datetime.utcnow()
            
            # Update records
            result = self.collection.update_many(query, {"$set": new_values})
            
            # Audit log for each updated record
            if result.modified_count > 0:
                updated_records = self.collection.find(query, {"_id": 1})
                for record in updated_records:
                    self._audit("update", str(record["_id"]), {"user_id": user_id})
            
            self.logger.info(f"Updated {result.modified_count} animal records")
            return {
                "ok": True, 
                "data": {"modified_count": result.modified_count}, 
                "error": None
            }
            
        except ValueError as e:
            self.logger.error(f"Validation error in update: {str(e)}")
            return {"ok": False, "data": None, "error": str(e)}
        except PyMongoError as e:
            self.logger.error(f"Database error in update: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def delete(self, query: Dict[str, Any], confirm_empty_filter: bool = False, 
               user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete animal records matching the query.
        
        Args:
            query: MongoDB query filter
            confirm_empty_filter: Must be True to allow empty filter (delete all)
            user_id: ID of user performing the operation
            
        Returns:
            {"ok": bool, "data": {"deleted_count": int}, "error": str|None}
        """
        try:
            # Input validation
            query = self._require_dict(query, "query")
            
            # Safe delete guard - prevent accidental mass deletion
            if not query and not confirm_empty_filter:
                return {
                    "ok": False, 
                    "data": None, 
                    "error": "Empty filter requires confirm_empty_filter=True"
                }
            
            # Get IDs of records to be deleted for audit
            to_delete = list(self.collection.find(query, {"_id": 1}))
            
            # Perform deletion
            result = self.collection.delete_many(query)
            
            # Audit log for each deleted record
            for record in to_delete:
                self._audit("delete", str(record["_id"]), {"user_id": user_id})
            
            self.logger.info(f"Deleted {result.deleted_count} animal records")
            return {
                "ok": True, 
                "data": {"deleted_count": result.deleted_count}, 
                "error": None
            }
            
        except ValueError as e:
            self.logger.error(f"Validation error in delete: {str(e)}")
            return {"ok": False, "data": None, "error": str(e)}
        except PyMongoError as e:
            self.logger.error(f"Database error in delete: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def get_audit_trail(self, target_id: Optional[str] = None, 
                       action: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve audit trail entries.
        
        Args:
            target_id: Filter by specific record ID
            action: Filter by action type (create, read, update, delete)
            
        Returns:
            {"ok": bool, "data": [audit_entries], "error": str|None}
        """
        try:
            query = {}
            if target_id:
                query["target_id"] = target_id
            if action:
                query["action"] = action
            
            audits_collection = self.database["audits"]
            cursor = audits_collection.find(query).sort("timestamp", -1).limit(100)
            
            audit_entries = []
            for entry in cursor:
                entry = self._normalize_id(entry)
                # Convert datetime to string for JSON serialization
                if isinstance(entry.get("timestamp"), datetime):
                    entry["timestamp"] = entry["timestamp"].isoformat()
                audit_entries.append(entry)
            
            return {"ok": True, "data": audit_entries, "error": None}
            
        except PyMongoError as e:
            self.logger.error(f"Database error in get_audit_trail: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'client'):
            self.client.close()
            self.logger.info("Database connection closed")