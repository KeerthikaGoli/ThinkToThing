import os
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging


class LLMHandler:
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GGML"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logging.info(f"Loading {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def enhance_prompt(self, prompt: str, style_guide: Optional[Dict] = None) -> str:
        """Enhance the user's prompt with more artistic and detailed descriptions."""
        try:
            system_prompt = """You are a creative AI assistant specialized in enhancing text prompts for image generation.
Your task is to expand the given prompt with rich, artistic details while maintaining the original intent.
Focus on visual elements like lighting, mood, composition, colors, and artistic style.
Keep the enhanced prompt concise but descriptive."""

            if style_guide:
                system_prompt += f"\nStyle preferences: {style_guide}"
            
            input_text = f"{system_prompt}\n\nOriginal prompt: {prompt}\nEnhanced prompt:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            enhanced_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced_prompt = enhanced_prompt.split("Enhanced prompt:")[-1].strip()
            
            if not enhanced_prompt:
                return prompt  # Fallback to original prompt if enhancement fails
                
            return enhanced_prompt
            
        except Exception as e:
            logging.error(f"Error enhancing prompt: {str(e)}")
            return prompt  # Fallback to original prompt
    
    def analyze_reference(self, reference_prompt: str, new_prompt: str) -> Dict:
        """Analyze the relationship between a reference creation and a new prompt."""
        try:
            system_prompt = """Analyze the relationship between a reference creation and a new prompt.
Extract key differences and similarities in terms of:
1. Subject matter
2. Style
3. Mood
4. Composition
Return specific aspects that should be maintained or modified."""

            input_text = f"{system_prompt}\n\nReference: {reference_prompt}\nNew prompt: {new_prompt}\nAnalysis:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = analysis.split("Analysis:")[-1].strip()
            
            if not analysis:
                analysis = "No significant relationships found."
            
            return {
                "analysis": analysis,
                "reference_prompt": reference_prompt,
                "new_prompt": new_prompt
            }
            
        except Exception as e:
            logging.error(f"Error analyzing reference: {str(e)}")
            return {
                "analysis": "Failed to analyze relationship.",
                "reference_prompt": reference_prompt,
                "new_prompt": new_prompt
            } 