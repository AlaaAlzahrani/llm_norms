import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from processors.word_ai_ratings import (
    WordAIRatings, 
    LogProbRatingStrategy, 
    DirectRatingStrategy
)

MODEL_STRATEGIES = {
    'o1-preview': (DirectRatingStrategy, 'OPENAI_API_KEY'),
    'o1-mini': (DirectRatingStrategy, 'OPENAI_API_KEY'),
    'gpt-4': (LogProbRatingStrategy, 'OPENAI_API_KEY'),
    'gpt-4o': (LogProbRatingStrategy, 'OPENAI_API_KEY'),
    'gpt-3.5-turbo': (LogProbRatingStrategy, 'OPENAI_API_KEY'),
    'claude-3-5-sonnet-20241022': (DirectRatingStrategy, 'ANTHROPIC_API_KEY'),
    'claude-3-opus-20240229': (DirectRatingStrategy, 'ANTHROPIC_API_KEY'),
    'claude-3-haiku-20240307': (DirectRatingStrategy, 'ANTHROPIC_API_KEY'),
}

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process words with AI ratings for different dimensions')
    
    parser.add_argument(
        '--dimensions',
        nargs='+',
        default=['concreteness'],
        help='Dimensions to process (e.g., concreteness familiarity)'
    )
    
    parser.add_argument(
        '--prompt-types',
        nargs='+',
        default=['standard'],
        choices=['detailed', 'standard', 'short'],
        help='Type of prompt to use (detailed, standard, or short)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='claude-3-5-sonnet-20241022',
        choices=list(MODEL_STRATEGIES.keys()),
        help='AI model to use for ratings'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to input file containing words'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of words to process in each batch'
    )
    
    return parser.parse_args()

def validate_environment(model: str):
    """Validate that necessary API keys are present"""
    strategy_class, api_key_name = MODEL_STRATEGIES[model]
    if not os.getenv(api_key_name):
        raise ValueError(f"No {api_key_name} found in environment variables")

def setup_paths(args, project_root: Path):
    """Setup and validate input/output paths"""
    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.is_absolute():
            input_file = project_root / args.input_file
    else:
        input_file = project_root / 'data' / 'input' / 'words.txt'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / args.output_dir
    else:
        output_dir = project_root / 'data' / 'output'
    
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found at either:\n"
            f"1. {input_file}\n"
            f"2. {Path.cwd() / args.input_file}\n"
            "Please provide the correct path to the input file."
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return input_file, output_dir

def main():
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    args = parse_arguments()
    
    load_dotenv()
    
    validate_environment(args.model)
    
    input_file, output_dir = setup_paths(args, project_root)
    print(f"Using input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    strategy = create_strategy(args.model)
    
    for dimension in args.dimensions:
        for prompt_type in args.prompt_types:
            print(f"\nStarting {dimension} ratings using {prompt_type} prompt with {args.model}...")
            
            try:
                rater = WordAIRatings(
                    dimension=dimension,
                    rating_strategy=strategy,
                    prompt_type=prompt_type,
                    input_file=str(input_file),
                    output_dir=str(output_dir),
                    batch_size=args.batch_size
                )
                
                rater.process_words()
                print(f"Completed {dimension} ratings with {prompt_type} prompt successfully!")
                
            except Exception as e:
                print(f"Error processing {dimension} ratings with {prompt_type} prompt: {str(e)}")
                continue

def create_strategy(model: str) -> LogProbRatingStrategy | DirectRatingStrategy:
    """Create and return appropriate strategy for the selected model"""
    strategy_class, _ = MODEL_STRATEGIES[model]
    return strategy_class(model=model)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Application error: {str(e)}")