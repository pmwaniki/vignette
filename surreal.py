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
    try:
        db.upsert(RecordID(table, id), value)
    except HTTPError as e:
        # print("raising auth error")
        error=e
        if error.response.content.__contains__(b'token has expired'):
            print("Token has expired: trying to sign in")
            db.signin({'username': "root", 'password': os.getenv("SURREALDB_PASSWORD")})
            db.use("llm_study", "benchmark")
            db.upsert(RecordID(table, id), value)
        else:
            raise e