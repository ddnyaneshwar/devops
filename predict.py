# Optimized prediction script for Hugging Face Inference Endpoints
# This version uses less memory and is optimized for smaller instances

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cog import BasePredictor, Input
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the DevOps SLM model into memory with optimizations"""
        logger.info("Loading DevOps SLM model with memory optimizations...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            "lakhera2023/devops-slm",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # Memory optimizations
            use_cache=False,  # Disable KV cache to save memory
            attn_implementation="eager"  # Use eager attention (less memory)
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("lakhera2023/devops-slm")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("DevOps SLM model loaded successfully with optimizations!")

    def predict(
        self,
        prompt: str = Input(description="DevOps question or task prompt"),
        max_tokens: int = Input(description="Maximum number of tokens to generate", default=150, ge=1, le=500),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0.1, le=2.0),
        top_p: float = Input(description="Top-p sampling parameter", default=0.9, ge=0.1, le=1.0),
        top_k: int = Input(description="Top-k sampling parameter", default=50, ge=1, le=100),
    ) -> str:
        """Generate DevOps response using the specialized model"""
        try:
            logger.info(f"Generating response for prompt: {prompt[:100]}...")
            
            # Tokenize input with truncation to save memory
            inputs = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256).to(self.device)
            
            # Generate response with memory optimizations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    early_stopping=True,  # Stop early to save computation
                    use_cache=False,  # Don't use KV cache
                    output_attentions=False,  # Don't output attention weights
                    output_hidden_states=False  # Don't output hidden states
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if prompt in full_response:
                response = full_response.split(prompt)[-1].strip()
            else:
                response = full_response.strip()
            
            # Clean up template artifacts
            response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            
            # Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Generated response length: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"