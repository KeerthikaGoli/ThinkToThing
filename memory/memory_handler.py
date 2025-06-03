import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
import logging


class MemoryHandler:
    def __init__(self, memory_type: str = "sqlite"):
        self.memory_type = memory_type
        self.db_path = "memory/creative_memory.db"
        self.vector_db_path = "memory/vector_store"
        
        try:
            if memory_type == "sqlite":
                self._init_sqlite()
            elif memory_type == "vector":
                self._init_vector_store()
            else:
                raise ValueError(f"Unsupported memory type: {memory_type}")
        except Exception as e:
            logging.error(f"Error initializing memory handler: {str(e)}")
            raise RuntimeError(f"Failed to initialize memory handler: {str(e)}")
    
    def _init_sqlite(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS creations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    prompt TEXT,
                    image_path TEXT,
                    model_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            logging.info("SQLite database initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing SQLite: {str(e)}")
            raise
    
    def _init_vector_store(self):
        try:
            os.makedirs(self.vector_db_path, exist_ok=True)
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.vector_db_path
            ))
            self.collection = self.chroma_client.create_collection(
                name="creative_memory",
                get_or_create=True
            )
            logging.info("Vector store initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def save_creation(self, 
                     creation_id: str,
                     session_id: str,
                     prompt: str,
                     image_path: Optional[str] = None,
                     model_path: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> None:
        try:
            if self.memory_type == "sqlite":
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO creations 
                    (id, session_id, prompt, image_path, model_path, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    creation_id,
                    session_id,
                    prompt,
                    image_path,
                    model_path,
                    json.dumps(metadata or {}),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                logging.info(f"Creation {creation_id} saved to SQLite")
            
            elif self.memory_type == "vector":
                self.collection.add(
                    documents=[prompt],
                    metadatas=[{
                        "image_path": image_path,
                        "model_path": model_path,
                        **(metadata or {})
                    }],
                    ids=[creation_id]
                )
                logging.info(f"Creation {creation_id} saved to vector store")
        
        except Exception as e:
            logging.error(f"Error saving creation: {str(e)}")
            raise RuntimeError(f"Failed to save creation: {str(e)}")
    
    def get_similar_creations(self, prompt: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            if self.memory_type == "vector":
                results = self.collection.query(
                    query_texts=[prompt],
                    n_results=n_results
                )
                return [
                    {
                        "id": id,
                        "prompt": doc,
                        **metadata
                    }
                    for id, doc, metadata in zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0]
                    )
                ]
            return []
        except Exception as e:
            logging.error(f"Error finding similar creations: {str(e)}")
            return []
    
    def get_creation_by_id(self, creation_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self.memory_type == "sqlite":
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM creations WHERE id = ?",
                    (creation_id,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        "id": row[0],
                        "session_id": row[1],
                        "prompt": row[2],
                        "image_path": row[3],
                        "model_path": row[4],
                        "metadata": json.loads(row[5]),
                        "created_at": row[6]
                    }
            return None
        except Exception as e:
            logging.error(f"Error retrieving creation: {str(e)}")
            return None 