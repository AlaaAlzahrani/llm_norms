from pathlib import Path
import yaml
from typing import Dict, Optional

class PromptManager:
    def __init__(self, prompts_dir: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.prompts_dir = Path(prompts_dir) if prompts_dir else self.project_root / 'prompts'
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_cache: Dict[str, Dict] = {}
        
    def _load_dimension_prompts(self, dimension: str) -> Dict:
        """Load prompts for a specific dimension from YAML file"""
        if dimension in self.prompts_cache:
            return self.prompts_cache[dimension]
            
        prompt_file = self.prompts_dir / f"{dimension}.yaml"
        if not prompt_file.exists():
            raise FileNotFoundError(f"No prompts file found for dimension: {dimension}")
            
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
            self.prompts_cache[dimension] = prompts
            return prompts
    
    def get_prompt(self, dimension: str, prompt_key: str, **kwargs) -> str:
        """Get a specific prompt with optional parameter substitution"""
        prompts = self._load_dimension_prompts(dimension)
        if prompt_key not in prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found in dimension '{dimension}'")
            
        prompt_template = prompts[prompt_key]
        return prompt_template.format(**kwargs) if kwargs else prompt_template
    
    def save_prompt(self, dimension: str, prompt_key: str, prompt_template: str):
        """Save a new prompt or update existing one"""
        prompt_file = self.prompts_dir / f"{dimension}.yaml"
        
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f) or {}
        else:
            prompts = {}
        
        prompts[prompt_key] = prompt_template
        
        with open(prompt_file, 'w', encoding='utf-8') as f:
            yaml.dump(prompts, f, allow_unicode=True, sort_keys=False)
            
        if dimension in self.prompts_cache:
            self.prompts_cache[dimension][prompt_key] = prompt_template