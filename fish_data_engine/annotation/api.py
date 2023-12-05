from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from loguru import logger
from hashlib import sha384
from uuid import uuid4
import pymongo
import boto3
from pathlib import Path

load_dotenv()

DB_SESSION = None
BOTO_SESSION = boto3.session.Session()
S3_CLIENT = BOTO_SESSION.client(
    service_name="s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = os.getenv("S3_PREFIX")


def init_database():
    global DB_SESSION

    import dns.resolver

    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ["8.8.8.8"]

    client = pymongo.MongoClient(host=os.getenv("MONGODB_URI"))
    db = client.get_database()

    db.users.create_index([("name", pymongo.ASCENDING)], unique=True)

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


def create_sample(sample_type, path):
    name = str(uuid4())
    suffix = Path(path).suffix
    file_path = f"{S3_PREFIX}/{name}{suffix}"
    S3_CLIENT.upload_file(path, S3_BUCKET_NAME, file_path)

    sample = {
        "_id": name,
        "sample_type": sample_type,
        "s3_path": file_path,
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


def get_new_sample(sample_type, uid):
    sample = DB_SESSION.samples.aggregate(
        [
            {
                "$match": {
                    "$or": [
                        {"status": "pending", "sample_type": sample_type},
                        {
                            "status": "annotating",
                            "sample_type": sample_type,
                            "updated_at": {
                                "$lt": datetime.utcnow() - timedelta(minutes=10)
                            },
                        },
                    ],
                }
            },
            {"$sample": {"size": 1}},
        ]
    )

    if sample is None:
        return None

    sample = list(sample)[0]

    DB_SESSION.samples.find_one_and_update(
        {
            "_id": sample["_id"],
        },
        {
            "$set": {
                "status": "annotating",
                "updated_at": datetime.utcnow(),
            }
        },
    )

    if sample is None:
        return None

    suffix = Path(sample["s3_path"]).suffix
    data = S3_CLIENT.get_object(
        Bucket=S3_BUCKET_NAME,
        Key=sample["s3_path"],
    )["Body"].read()

    tmp_path = Path("tmp") / f"{uid}{suffix}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(data)

    sample["data"] = tmp_path

    return sample
