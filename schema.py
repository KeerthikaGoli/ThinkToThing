from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from openfabric_pysdk.utility import SchemaUtil


@dataclass
class InputClass(SchemaUtil):
    prompt: str
    session_id: Optional[str] = None
    reference_id: Optional[str] = None


@dataclass
class OutputClass(SchemaUtil):
    message: str
    image_path: Optional[str] = None
    model_path: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None


@dataclass
class ConfigClass(SchemaUtil):
    app_ids: List[str]
    llm_model: str
    memory_type: str
