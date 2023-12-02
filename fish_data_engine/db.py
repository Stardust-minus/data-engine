from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from loguru import logger
from hashlib import sha384
from uuid import uuid4
import pymongo

load_dotenv()

DB_SESSION = None
# class User(Document):
#     name = StringField(required=True, unique=True, index=True)
#     password = StringField(required=True)
#     created_at = StringField(required=True)

#     meta = {'collection': 'users'}


# class Sample(Document):
#     id = StringField(required=True, unique=True, index=True)
#     sample_type = StringField(required=True, index=True)
#     s3_url = StringField(required=True)
#     annotation = DictField()
#     annotator = StringField(index=True, sparse=True)
#     created_at = StringField(required=True, index=True)
#     updated_at = StringField(required=True, index=True)

#     meta = {'collection': 'samples'}


def init_database():
    global DB_SESSION

    client = pymongo.MongoClient(host=os.getenv("MONGODB_URI"))
    db = client.get_database()

    db.users.create_index([("name", pymongo.ASCENDING)], unique=True)
    db.samples.create_index([("id", pymongo.ASCENDING)], unique=True)

    db.samples.create_index([("sample_type", pymongo.ASCENDING)])
    db.samples.create_index([("annotator", pymongo.ASCENDING)])
    db.samples.create_index([("status", pymongo.ASCENDING)])
    db.samples.create_index([("created_at", pymongo.DESCENDING)])
    db.samples.create_index([("updated_at", pymongo.DESCENDING)])

    logger.info("Database loaded")

    DB_SESSION = db
    return db


def create_user(name, password):
    user = {
        "_id": str(uuid4()),
        "name": name,
        "password": sha384(password.encode("utf-8")).hexdigest(),
        "created_at": datetime.utcnow(),
    }

    # Will raise DuplicateKeyError if user already exists
    DB_SESSION.users.insert_one(user)
    return user


def login_user(name, password):
    password = sha384(password.encode("utf-8")).hexdigest()
    return DB_SESSION.users.find_one({"name": name, "password": password})


def remove_user(name):
    return DB_SESSION.users.delete_one({"name": name})


def create_sample(sample_type, s3_url):
    sample = {
        "_id": str(uuid4()),
        "sample_type": sample_type,
        "s3_url": s3_url,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "annotation": None,
        "annotator": None,
        "status": "pending",
    }

    # Will raise DuplicateKeyError if sample already exists
    DB_SESSION.samples.insert_one(sample)
    return sample


def update_sample(sample_id, annotation, annotator):
    return DB_SESSION.samples.update_one(
        {"_id": sample_id},
        {
            "$set": {
                "annotation": annotation,
                "updated_at": datetime.utcnow(),
                "annotator": annotator,
                "status": "annotated",
            }
        },
    )


def get_new_sample(sample_type):
    return DB_SESSION.samples.find_one_and_update(
        {
            "$or": [
                {"status": "pending", "sample_type": sample_type},
                {
                    "status": "annotating",
                    "sample_type": sample_type,
                    "updated_at": {"$lt": datetime.utcnow() - timedelta(minutes=10)},
                },
            ],
        },
        {
            "$set": {
                "status": "annotating",
                "updated_at": datetime.utcnow(),
            }
        },
    )
