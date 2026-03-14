"""LLM-based zero-shot text classification."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import re


class LLMClassifier:
    """LLM-based zero-shot classifier using prompting."""
    
    def __init__(self, model_name: str, device: str = None):
        """Initialize LLM classifier.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
        """
        print(f"Loading LLM: {model_name}")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        print(f"Model loaded on device: {self.model.device}")
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_prompt(
        self,
        text: str,
        label_texts: Dict[int, List[str]],
        few_shot_examples: List = None
    ) -> str:
        """Create classification prompt.
        
        Args:
            text: Text to classify
            label_texts: Dictionary mapping label IDs to descriptions
            few_shot_examples: Optional few-shot examples
            
        Returns:
            Formatted prompt string
        """
        # Create label list
        labels = []
        label_map = {}
        for label_id, descriptions in label_texts.items():
            # Use first description
            label_desc = descriptions[0]
            labels.append(f"{label_id}: {label_desc}")
            label_map[label_id] = label_desc
        
        # Build prompt
        prompt = "You are a text classification expert. Classify the given text into one of the provided categories.\n\n"
        prompt += "Categories:\n"
        for label in labels:
            prompt += f"- {label}\n"
        prompt += "\n"
        
        # Add few-shot examples if provided
        if few_shot_examples:
            prompt += "Examples:\n"
            for example in few_shot_examples:
                prompt += f"Text: {example['text']}\n"
                prompt += f"Category: {example['label']}\n\n"
        
        # Add the text to classify
        prompt += f"Text: {text}\n"
        prompt += "Category (return only the number):"
        
        return prompt
    
    def predict(
        self,
        texts: List[str],
        label_texts: Dict[int, List[str]],
        batch_size: int = 1,
        max_new_tokens: int = 10
    ) -> List[int]:
        """Predict labels for texts.
        
        Args:
            texts: List of texts to classify
            label_texts: Dictionary mapping label IDs to descriptions
            batch_size: Batch size (LLMs usually work with batch_size=1)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of predicted label IDs
        """
        predictions = []
        confidences = []
        
        total = len(texts)
        for i, text in enumerate(texts):
            # Create prompt
            prompt = self.create_prompt(text, label_texts)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Extract label
            pred_label = self._extract_label(generated_text, label_texts)
            predictions.append(pred_label)
            
            # Simple confidence (can be improved)
            confidences.append(0.9)  # Placeholder
            
            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == total:
                percent = (i + 1) * 100 // total
                print(f"  Classified {i+1}/{total} texts ({percent}%)...")
        
        return predictions, confidences
    
    def _extract_label(self, generated_text: str, label_texts: Dict[int, List[str]]) -> int:
        """Extract label ID from generated text.
        
        Args:
            generated_text: Generated text from LLM
            label_texts: Label descriptions
            
        Returns:
            Predicted label ID
        """
        # Try to find a number in the first few characters
        match = re.search(r'(\d+)', generated_text[:20])
        if match:
            label_id = int(match.group(1))
            if label_id in label_texts:
                return label_id
        
        # Fallback: look for label keywords in text
        text_lower = generated_text.lower()
        for label_id, descriptions in label_texts.items():
            for desc in descriptions:
                if any(word.lower() in text_lower for word in desc.split()[:3]):
                    return label_id
        
        # Default to first label
        return list(label_texts.keys())[0]