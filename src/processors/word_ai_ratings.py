from pathlib import Path
import csv
from datetime import datetime
import time
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from utils.prompt_manager import PromptManager

load_dotenv()

class RatingStrategy(ABC):
    """Abstract base class for rating strategies"""
    
    @abstractmethod
    def initialize_client(self):
        """Initialize the API client"""
        pass
        
    @abstractmethod
    def get_rating(self, prompt: str, scale_range: List[int]) -> Dict[str, Any]:
        """Get rating for a word"""
        pass
        
    @abstractmethod
    def get_output_headers(self) -> List[str]:
        """Get headers for the output CSV file"""
        pass

class LogProbRatingStrategy(RatingStrategy):
    """Strategy for models that support logprobs (e.g., GPT-4)"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.client = None
        
    def initialize_client(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def get_rating(self, prompt: str, scale_range: List[int]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            logprobs=True,
            top_logprobs=5,
            max_tokens=1
        )
        
        logprobs_list = response.choices[0].logprobs.content[0].top_logprobs
        
        probs = {
            int(logprob_obj.token.strip()): np.exp(logprob_obj.logprob)
            for logprob_obj in logprobs_list
            if logprob_obj.token.strip().isdigit() 
            and scale_range[0] <= int(logprob_obj.token.strip()) <= scale_range[1]
        }
        
        if not probs:
            raise ValueError("No valid probability values found")
            
        total_prob = sum(probs.values())
        normalized_probs = {k: v/total_prob for k, v in probs.items()}
        
        return {
            'top_rating': max(normalized_probs.items(), key=lambda x: x[1])[0],
            'weighted_avg': round(sum(k * v for k, v in normalized_probs.items()), 3),
            'probabilities': str(normalized_probs),
            'model': self.model
        }
        
    def get_output_headers(self) -> List[str]:
        return ['top_rating', 'weighted_avg', 'probabilities', 'model', 'word', 'prompt_type']


class DirectRatingStrategy(RatingStrategy):
    """Strategy for models that don't support logprobs (e.g., Claude)"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self.client = None
        
    def initialize_client(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
    def get_rating(self, prompt: str, scale_range: List[int]) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                max_tokens=10  # Increased from 1 to ensure we get the full response
            )
                        
            if not response.content:
                raise ValueError("Empty response from Claude API")
            
            content_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            rating_str = ''.join(filter(str.isdigit, content_text))
            
            if not rating_str:
                raise ValueError(f"No numeric rating found in response: {content_text}")
                
            rating = int(rating_str)
            if not scale_range[0] <= rating <= scale_range[1]:
                raise ValueError(f"Rating {rating} out of valid range {scale_range[0]}-{scale_range[1]}")
            
            return {
                'rating': rating,
                'model': self.model
            }
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Response structure: {response if 'response' in locals() else 'No response'}")
            raise e
        
    def get_output_headers(self) -> List[str]:
        return ['word', 'rating', 'prompt_type', 'model']

class WordAIRatings:
    """A class to handle AI-based word ratings across different dimensions."""
    
    def __init__(self, 
                 dimension: str,
                 rating_strategy: RatingStrategy,
                 prompt_type: str = 'standard',
                 input_file: Optional[str] = None, 
                 output_dir: Optional[str] = None, 
                 batch_size: int = 50):
        """
        Initialize Word AI Ratings processor
        """
        self.rating_strategy = rating_strategy
        self.rating_strategy.initialize_client()
        
        self.prompt_manager = PromptManager()
        self.prompt_type = prompt_type
        self.prompt_key = f"{prompt_type}" 
        
        self._validate_dimension_and_prompt(dimension, self.prompt_key)
        
        self.project_root = Path(__file__).parent.parent.parent
        self.input_file = self._setup_input_path(input_file)
        self.output_dir = self._setup_output_path(output_dir)
        self.dimension = dimension
        self.batch_size = batch_size
        self.output_file = None
        self.processed_words = set()
        
        self.metadata = self._load_metadata()
        self.scale_range = self.metadata.get('scale_range', [1, 7])

    def _validate_dimension_and_prompt(self, dimension: str, prompt_key: str) -> None:
        """Validate the dimension and prompt type against available prompts."""
        try:
            prompts = self.prompt_manager._load_dimension_prompts(dimension)
            if prompt_key not in prompts:
                raise ValueError(f"Invalid prompt type '{self.prompt_type}' for dimension '{dimension}'")
        except FileNotFoundError:
            raise ValueError(f"Invalid dimension: {dimension}")

    def _setup_input_path(self, input_file: Optional[str]) -> Path:
        """Setup and validate input file path."""
        if input_file is None:
            input_file = self.project_root / 'data' / 'input' / self.dimension / 'words.txt'
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        return input_path

    def _setup_output_path(self, output_dir: Optional[str]) -> Path:
        """Setup output directory path."""
        if output_dir is None:
            output_dir = self.project_root / 'data' / 'output' / self.dimension
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _load_metadata(self) -> Dict:
        """Load metadata for the current dimension."""
        prompts = self.prompt_manager._load_dimension_prompts(self.dimension)
        metadata = prompts.get('prompt_metadata', {})
        if not metadata:
            print(f"Warning: No metadata found for dimension {self.dimension}")
        return metadata

    def _create_output_file(self) -> Path:
        """Create a new timestamped output file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{self.dimension}_{self.prompt_type}_{timestamp}.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.rating_strategy.get_output_headers())
            
        return output_file

    def get_rating(self, word: str) -> Optional[Dict[str, Any]]:
        """Get AI rating for a single word."""
        prompt = self.prompt_manager.get_prompt(self.dimension, self.prompt_key, word=word)
                
        try:
            result = self.rating_strategy.get_rating(prompt, self.scale_range)
            result.update({
                'word': word,
                'prompt_type': self.prompt_type
            })
            return result
            
        except Exception as e:
            print(f"Error processing word '{word}': {str(e)}")
            return None

    def process_words(self) -> None:
        """Process words from input file and save results."""
        self.output_file = self._create_output_file()
        print(f"Results will be saved to: {self.output_file}")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise IOError(f"Error reading input file: {str(e)}")
            
        if not words:
            print("No words to process!")
            return
            
        print(f"Processing {len(words)} words for {self.dimension} ratings...")
        
        for i in range(0, len(words), self.batch_size):
            batch = words[i:i + self.batch_size]
            self._process_batch(batch, i)
            
        print(f"Processing complete! Results saved to {self.output_file}")

    def _process_batch(self, batch: List[str], batch_start_idx: int) -> None:
        """Process a batch of words."""
        batch_results = []
        
        for word in batch:
            if word in self.processed_words:
                print(f"Skipping already processed word: {word}")
                continue
                
            result = self.get_rating(word)
            if result:
                batch_results.append(result)
                self.processed_words.add(word)
                print(f"Processed '{word}': rating={result.get('rating') or result.get('top_rating')}")
            time.sleep(1)  
        
        if batch_results:
            df = pd.DataFrame(batch_results)
            df.to_csv(self.output_file, mode='a', header=False, index=False, encoding='utf-8')
        
        current_batch = (batch_start_idx // self.batch_size) + 1
        total_batches = (len(self.processed_words) + self.batch_size - 1) // self.batch_size
        print(f"Processed and saved batch {current_batch}/{total_batches}")