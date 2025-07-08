import os

from dotenv import load_dotenv
from surrealdb import Surreal, RecordID

load_dotenv()


def create_conn():
    db=Surreal('https://aws.datahubweb.com:6000')
    db.signin({'username': "root", 'password': os.getenv("SURREALDB_PASSWORD")})
    db.use("llm_study", "benchmark")
    return db

def add_or_update(id,value,table,db):
    db.upsert(RecordID(table, id), value)