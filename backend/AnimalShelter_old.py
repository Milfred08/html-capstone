"""
Original AnimalShelter CRUD Module (CS-340 style)
Basic PyMongo operations with minimal error handling and validation.
This represents the "before" state for comparison purposes.
"""

import pymongo
from pymongo import MongoClient

class AnimalShelter:
    def __init__(self):
        # Hardcoded connection - poor practice
        self.client = MongoClient('mongodb://localhost:27017/')
        self.database = self.client['shelter']
        self.collection = self.database['animals']

    def create(self, data):
        # No validation, inconsistent return values
        try:
            result = self.collection.insert_one(data)
            return result.inserted_id
        except:
            return False

    def read(self, query):
        # Returns raw cursor, inconsistent with other methods
        try:
            if query:
                return self.collection.find(query)
            else:
                return self.collection.find({})
        except:
            return None

    def update(self, query, new_values):
        # No validation of inputs, poor error handling
        try:
            result = self.collection.update_many(query, {"$set": new_values})
            return result.modified_count > 0
        except:
            return False

    def delete(self, query):
        # Dangerous - no protection against empty query
        try:
            result = self.collection.delete_many(query)
            return result.deleted_count > 0
        except:
            return False