import logging
import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from openfabric_pysdk.context import Stub, SchemaUtil

from llm.llm_handler import LLMHandler
from memory.memory_handler import MemoryHandler
from pipeline.pipeline_handler import PipelineHandler
from schema import InputClass, OutputClass, ConfigClass

# Initialize FastAPI app
app = FastAPI(
    title="Creative AI Partner",
    description="A powerful AI-driven creative platform that transforms text prompts into stunning images and 3D models.",
    version="1.0.0"
)

# Initialize configurations
configurations = {
    'super-user': ConfigClass(
        app_ids=[
            "f0997a01-d6d3-a5fe-53d8-561300318557",
            "69543f29-4d41-4afc-7f29-3d51591f11eb"
        ],
        llm_model="deepseek-ai/deepseek-coder-6.7b-base",
        memory_type="vector"
    )
}

@app.post("/execution")
async def execute(model: SchemaUtil) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (SchemaUtil): The model object containing request and response structures.
    """
    try:
        # Retrieve input
        request: InputClass = model.request
        
        # Retrieve user config
        user_config: ConfigClass = configurations.get('super-user', None)
        logging.info(f"Configuration: {configurations}")
        
        if not user_config:
            raise ValueError("User configuration not found")
        
        # Initialize components
        stub = Stub(user_config.app_ids)
        llm_handler = LLMHandler(user_config.llm_model)
        memory_handler = MemoryHandler(user_config.memory_type)
        
        # Initialize pipeline
        pipeline = PipelineHandler(
            stub=stub,
            llm_handler=llm_handler,
            memory_handler=memory_handler,
            config=user_config.__dict__
        )
        
        # Process the creation
        result = pipeline.process_creation(
            prompt=request.prompt,
            session_id=request.session_id,
            reference_id=request.reference_id
        )
        
        # Prepare response
        response: OutputClass = model.response
        response.message = "Creation successful!"
        response.image_path = result["image_path"]
        response.model_path = result["model_path"]
        response.history = [result]
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Create necessary directories on startup
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/models", exist_ok=True)
os.makedirs("memory", exist_ok=True)
