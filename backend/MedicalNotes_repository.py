"""
Medical Notes Repository Layer
Specialized repository for VoiceNote MD application with medical data handling.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pymongo import MongoClient, IndexModel
from pymongo.errors import PyMongoError, DuplicateKeyError
import uuid

class MedicalNotesRepository:
    """
    Specialized repository for medical notes and SOAP data.
    """
    
    def __init__(self, mongo_uri: Optional[str] = None, db_name: str = "voicenote_md", collection_name: str = "medical_notes"):
        """Initialize the medical notes repository."""
        self.logger = logging.getLogger("medicalnotes")
        
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

    def _validate_medical_note_fields(self, data: Dict[str, Any]) -> None:
        """Validate medical note specific fields."""
        required_fields = ["note_id", "original_text"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required medical note fields: {missing_fields}")

    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data for JSON serialization."""
        if data and '_id' in data:
            data['_id'] = str(data['_id'])
        
        # Handle datetime serialization
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data

    def _audit(self, action: str, target_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Audit trail for medical operations."""
        audit_entry = {
            "action": action,
            "target_id": target_id,
            "timestamp": datetime.utcnow(),
            "collection": self.collection_name,
            "meta": meta or {}
        }
        
        try:
            audits_collection = self.database["audits"]
            audits_collection.insert_one(audit_entry)
            self.logger.info(f"Medical note audit logged: {action} on {target_id}")
        except PyMongoError as e:
            self.logger.error(f"Failed to log medical note audit: {str(e)}")

    def _ensure_indexes(self) -> None:
        """Create indexes for medical notes."""
        try:
            indexes = [
                IndexModel([("note_id", 1)]),
                IndexModel([("patient_id", 1)]),
                IndexModel([("provider_id", 1)]),
                IndexModel([("note_type", 1)]),
                IndexModel([("processed_at", -1)]),
                IndexModel([("created_by", 1)])
            ]
            
            self.collection.create_indexes(indexes)
            self.logger.info("Medical notes indexes ensured")
        except PyMongoError as e:
            self.logger.warning(f"Failed to create medical notes indexes: {str(e)}")

    def create(self, data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new medical note record."""
        try:
            data = self._require_dict(data, "data")
            self._validate_medical_note_fields(data)
            
            # Add metadata
            data["stored_at"] = datetime.utcnow()
            if not data.get("created_by"):
                data["created_by"] = user_id
            
            # Insert record
            result = self.collection.insert_one(data.copy())
            created_record = self.collection.find_one({"_id": result.inserted_id})
            created_record = self._normalize_data(created_record)
            
            # Audit log
            self._audit("create", data.get("note_id", str(result.inserted_id)), {"user_id": user_id})
            
            self.logger.info(f"Created medical note: {result.inserted_id}")
            return {"ok": True, "data": created_record, "error": None}
            
        except ValueError as e:
            self.logger.error(f"Validation error in create: {str(e)}")
            return {"ok": False, "data": None, "error": str(e)}
        except DuplicateKeyError as e:
            self.logger.error(f"Duplicate key error in create: {str(e)}")
            return {"ok": False, "data": None, "error": "Medical note already exists"}
        except PyMongoError as e:
            self.logger.error(f"Database error in create: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def read(self, query: Optional[Dict[str, Any]] = None, limit: int = 100) -> Dict[str, Any]:
        """Read medical notes matching the query."""
        try:
            query = query or {}
            if query:
                query = self._require_dict(query, "query")
            
            cursor = self.collection.find(query).limit(limit).sort("processed_at", -1)
            records = [self._normalize_data(doc) for doc in cursor]
            
            self.logger.info(f"Read {len(records)} medical note records")
            return {"ok": True, "data": records, "error": None}
            
        except ValueError as e:
            self.logger.error(f"Validation error in read: {str(e)}")
            return {"ok": False, "data": None, "error": str(e)}
        except PyMongoError as e:
            self.logger.error(f"Database error in read: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def get_audit_trail(self, target_id: Optional[str] = None, action: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve audit trail for medical notes."""
        try:
            query = {"collection": self.collection_name}
            if target_id:
                query["target_id"] = target_id
            if action:
                query["action"] = action
            
            audits_collection = self.database["audits"]
            cursor = audits_collection.find(query).sort("timestamp", -1).limit(100)
            
            audit_entries = []
            for entry in cursor:
                entry = self._normalize_data(entry)
                audit_entries.append(entry)
            
            return {"ok": True, "data": audit_entries, "error": None}
            
        except PyMongoError as e:
            self.logger.error(f"Database error in get_audit_trail: {str(e)}")
            return {"ok": False, "data": None, "error": "Database operation failed"}

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'client'):
            self.client.close()
            self.logger.info("Medical notes database connection closed")