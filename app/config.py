from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv(override=True)

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file='.env',extra='ignore')
    aws_region:str='us-east-2'
    aws_access_key_id:str=''
    aws_secret_access_key:str=''
    s3_bucket:str=''
    database_url:str=''  # Supabase PostgreSQL External Database URL
    groq_api_key:str=''
    google_api_key:str=os.getenv("GOOGLE_API_KEY",'')
    embedding_provider:str='gemini'
    embedding_model:str='models/gemini-embedding-001'
    embed_dim:int=3072
    groq_chat_model:str='openai/gpt-oss-120b'
    max_upload_bytes:int=10485760
    pinecone_api_key:str=''
    pinecone_index_name:str='atlas-ai-index'
    pinecone_cloud:str='aws'
    pinecone_region:str='us-east-1'
    pinecone_ready_timeout:int=180
    min_relevance:float=0.25
    hitl_enabled:bool=True
    memory_turns:int=6


@lru_cache
def settings() -> Settings:
    return Settings()