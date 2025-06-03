import os
import uuid
from typing import Dict, Optional, Tuple
from datetime import datetime

from llm.llm_handler import LLMHandler
from memory.memory_handler import MemoryHandler
from openfabric_pysdk.context import Stub


class PipelineHandler:
    def __init__(self, 
                 stub: Stub,
                 llm_handler: LLMHandler,
                 memory_handler: MemoryHandler,
                 config: Dict):
        self.stub = stub
        self.llm_handler = llm_handler
        self.memory_handler = memory_handler
        self.config = config
        
        # Create output directories
        os.makedirs("static/images", exist_ok=True)
        os.makedirs("static/models", exist_ok=True)
    
    def process_creation(self,
                        prompt: str,
                        session_id: Optional[str] = None,
                        reference_id: Optional[str] = None) -> Dict:
        """Process a creation request from prompt to 3D model."""
        
        # Generate unique IDs
        creation_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Step 1: Get reference if provided
        reference_info = None
        if reference_id:
            reference_info = self.memory_handler.get_creation_by_id(reference_id)
        
        # Step 2: Enhance prompt using LLM
        enhanced_prompt = self.llm_handler.enhance_prompt(prompt)
        if reference_info:
            analysis = self.llm_handler.analyze_reference(
                reference_info["prompt"],
                enhanced_prompt
            )
            enhanced_prompt = f"{enhanced_prompt} (Style reference: {analysis['analysis']})"
        
        # Step 3: Generate image
        image_path = f"static/images/{creation_id}.png"
        image_result = self.stub.call(
            'f0997a01-d6d3-a5fe-53d8-561300318557',
            {'prompt': enhanced_prompt},
            'super-user'
        )
        
        # Save the image
        with open(image_path, 'wb') as f:
            f.write(image_result.get('result'))
        
        # Step 4: Generate 3D model
        model_path = f"static/models/{creation_id}.glb"
        model_result = self.stub.call(
            '69543f29-4d41-4afc-7f29-3d51591f11eb',
            {'image': open(image_path, 'rb').read()},
            'super-user'
        )
        
        # Save the 3D model
        with open(model_path, 'wb') as f:
            f.write(model_result.get('result'))
        
        # Step 5: Save to memory
        metadata = {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "reference_id": reference_id,
            "reference_analysis": analysis if reference_info else None,
            "created_at": datetime.now().isoformat()
        }
        
        self.memory_handler.save_creation(
            creation_id=creation_id,
            session_id=session_id,
            prompt=enhanced_prompt,
            image_path=image_path,
            model_path=model_path,
            metadata=metadata
        )
        
        return {
            "creation_id": creation_id,
            "session_id": session_id,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "image_path": image_path,
            "model_path": model_path,
            "metadata": metadata
        }
    
    def find_similar_creations(self, prompt: str, n_results: int = 5) -> Dict:
        """Find similar previous creations based on the prompt."""
        return self.memory_handler.get_similar_creations(prompt, n_results) 