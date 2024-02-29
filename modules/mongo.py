# !pip install pymongo

from pymongo import MongoClient
from bson import ObjectId

class MongoDB:
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string)
        
    def get(self, database, collection):
        data = list(self.client[database][collection].find({}))
        for doc in data:
            doc['_id'] = str(doc['_id'])
        return data

    def get_one(self, database, collection, id):
        doc = self.client[database][collection].find_one({"_id": ObjectId(id)})
        if doc:
            doc['_id'] = str(doc['_id'])
            return doc
        return None

    def create(self, database, collection, record_data):
        if isinstance(record_data, list):
            self.client[database][collection].insert_many(record_data)
        elif isinstance(record_data, dict):
            self.client[database][collection].insert_one(record_data)

    def update(self, database, collection, id, updated_data):
        return self.client[database][collection].update_one({"_id": ObjectId(id)}, {"$set": updated_data})

    def delete(self, database, collection, id):
        return self.client[database][collection].delete_one({"_id": ObjectId(id)})
