import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import torch
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch.distributed as dist
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DELTA.Engine.utils import setup_seed
from reason_backend_utils import (
    FINAL_DELTA_BACKEND_CONFIG,
    add_delta_backend_args,
    build_delta_backend_kwargs,
    delta_backend_config_fields,
    validate_delta_backend_args,
)
from tokenizer_utils import load_tokenizer
# from DELTA.Engine.SnapKV.backend import LMBackend
from DELTA.Engine.DELTA.backend import LMBackend
# from DELTA.Engine.DELTA.backend_monitor import LMBackend
from DELTA.Data.Reasoning.gsm8k import GSM8k
from DELTA.Data.Reasoning.math500 import Math500
from DELTA.Data.Reasoning.aime import AIME
from DELTA.Data.Reasoning.gpqa import Gpqa

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'

class DatasetWrapper:
    """Wrapper to make the dataset classes compatible with direct iteration"""
    
    def __init__(self, dataset_class, tokenizer=None, num_samples=None, year_filter=None):
        self.dataset = dataset_class(tokenizer=tokenizer, tot_num_data=num_samples or int(1e6), year_filter=year_filter)
        self.tokenizer = tokenizer
        
        # Limit samples if specified
        if num_samples is not None and num_samples < len(self.dataset):
            self.dataset.data = self.dataset.data.head(num_samples)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.data.iloc[idx]
        prompt = row['prompt']
        ground_truth = row['groundtruth']
        
        if self.tokenizer:
            # Tokenize the prompt with proper truncation
            input_ids = self.tokenizer.encode(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=8192
            ).squeeze(0)
            return input_ids, ground_truth
        else:
            return prompt, ground_truth
    
    def get_answer(self, idx):
        """Get the ground truth answer for a specific index"""
        return self.dataset.data.iloc[idx]['groundtruth']

def extract_answer_from_generation(text: str, dataset_type: str) -> str:
    """Extract the final answer from the generated text based on dataset type"""
    # Use the extract_answer function from utils.py which is more sophisticated
    from DELTA.Data.Reasoning.utils import extract_answer
    
    if dataset_type.lower() == 'gsm8k':
        return extract_answer(text, "gsm8k")
    elif dataset_type.lower() == 'math500':
        return extract_answer(text, "math")
    elif dataset_type.lower() in ['aime', 'aime2024', 'aime2025'] :
        return extract_answer(text, "aime")
    elif dataset_type.lower() == 'gpqa':
        return extract_answer(text, "multiple_choice")
    else:
        # Fallback to simple extraction
        if dataset_type.lower() in ['gsm8k', 'math500']:
            patterns = [
                r"#### *([+-]?\d+(?:\.\d+)?)",  # GSM8K format
                r"[Tt]he answer is ([+-]?\d+(?:\.\d+)?)",
                r"\$([+-]?\d+(?:\.\d+)?)\$",  # LaTeX format
                r"([+-]?\d+(?:\.\d+)?) *$",  # Number at the end
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
            
            # If no pattern matches, try to find the last number in the text
            numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
            if numbers:
                return numbers[-1]
                
        elif dataset_type.lower() in ['aime', 'aime2024', 'aime2025']:
            # AIME answers are integers from 0-999
            patterns = [
                r"[Tt]he answer is (\d{1,3})",
                r"#### *(\d{1,3})",
                r"\$(\d{1,3})\$",
                r"(\d{1,3}) *$",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    answer = int(match.group(1))
                    if 0 <= answer <= 999:
                        return str(answer)
            
            # Find all numbers and return the last valid AIME answer
            numbers = re.findall(r"(\d{1,3})", text)
            for num in reversed(numbers):
                if 0 <= int(num) <= 999:
                    return num
        
        elif dataset_type.lower() == 'gpqa':
            # Extract choice letter from boxed format for GPQA
            patterns = [
                r"\\boxed\{([A-D])\}",  # \boxed{A}
                r"[Tt]he answer is ([A-D])",
                r"([A-D])\s*$",  # Letter at the end
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).upper()
            
            # Look for any single letter A-D in the text
            letters = re.findall(r"\b([A-D])\b", text)
            if letters:
                return letters[-1].upper()
        
        return "N/A"

def calculate_accuracy(predictions: List[str], ground_truths: List[str], dataset_type: str) -> float:
    """Calculate accuracy using the sophisticated math_equal function from utils.py"""
    from DELTA.Data.Reasoning.utils import math_equal
    
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truths):
        pred_answer = extract_answer_from_generation(pred, dataset_type)
        truth_answer = str(truth).strip()
        
        # Use the sophisticated math_equal function for better accuracy
        if pred_answer != "N/A":
            try:
                if dataset_type.lower() == 'gpqa':
                    # For GPQA, do simple string comparison of letters
                    if pred_answer.upper() == truth_answer.upper():
                        correct += 1
                else:
                    if math_equal(pred_answer, truth_answer):
                        correct += 1
            except:
                # Fallback to simple string comparison
                if pred_answer == truth_answer:
                    correct += 1
    
    return correct / total if total > 0 else 0.0


def setup_configuration():
    """Setup command line arguments and configuration"""
    parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
    parser.add_argument('--model', type=Path, default=Path("/scratch/models/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')

    parser.add_argument('--B', type=int, default=16, help='Batch size.')
    parser.add_argument('--max_len', type=int, default=2048, help='Maximum generation length')

    parser.add_argument('--seed', type=int, default=123, help='Random seed.')

    add_delta_backend_args(parser)
    parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
    parser.add_argument('--printoutput', action='store_true', help='Whether to print sample outputs.')

    # Dataset selection arguments
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math500', 'aime', 'gpqa', 'aime2024', 'aime2025'], default='gsm8k',
                       help='Dataset to use for evaluation')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--split', type=str, default='test', 
                       help='Dataset split to use (train/test/validation)')

    # Add selective KV cache arguments
    parser.add_argument('--full_cache_layers', nargs='+', type=int, default=None, 
                       help='Layer indices that use full KV cache (e.g., --full_cache_layers 1 2 10 20)')
    parser.add_argument('--subset_cache_ratio', type=float, default=0.5, 
                       help='Ratio of KV cache to use for subset layers (default: 0.5)')
    parser.add_argument('--compression_ratio', type=int, default=1, help="[compression_ratio * subset_cache_ratio] is the required pages to start selection")
    parser.add_argument('--subset_cache_size', type=int, default=1024,
                        help='Fixed size of KV cache for subset layers (overrides ratio if set)')
    parser.add_argument('--L', type=int, default=8, help='The last L pages always kept.')
    parser.add_argument('--enable_selective_cache', action='store_true', 
                       help='Enable selective KV cache optimization')
    args = parser.parse_args()
    validate_delta_backend_args(args)

    # Adjust max_len based on dataset requirements
    if args.dataset == 'aime':
        args.max_len = max(args.max_len, 4096)  # AIME problems might need more space
    elif args.dataset in ['gsm8k', 'math500']:
        args.max_len = max(args.max_len, 2048)

    assert args.max_len % 128 == 0

    # Configure selective KV cache
    if args.full_cache_layers is not None:
        args.full_cache_layers = sorted(list(set(args.full_cache_layers)))

    return args

args = setup_configuration()

def setup_model_and_engine(config_args):
    """Setup model, engine, and tokenizer"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_tp = len(config_args.rank_group) > 1 if config_args.rank_group else False
    global_group = None
    rank = 0  # Default rank for non-distributed case
    
    if use_tp:
        from DELTA.Engine.tp import init_dist
        rank, global_group = init_dist()
    
    def conditional_print(*print_args, **kwargs):
        if not use_tp or rank == config_args.rank_group[0]:
            print(*print_args, **kwargs)
    
    conditional_print(f"Setting up model and engine...")
    conditional_print(f"Using device={DEVICE}")
    conditional_print("DELTA backend configuration:")
    conditional_print("  delta_backend=final_public_release")
    conditional_print(f"  page_selector_version={FINAL_DELTA_BACKEND_CONFIG['page_selector_version']}")
    conditional_print(f"  cuda_graph_decode={config_args.cuda_graph_decode}")
    conditional_print(f"  cuda_graph_delta_subset_segments={FINAL_DELTA_BACKEND_CONFIG['cuda_graph_delta_subset_segments']}")
    conditional_print(f"  delta_dump_buffer_dtype={FINAL_DELTA_BACKEND_CONFIG['delta_dump_buffer_dtype']}")
    conditional_print(f"  delta_page_score_impl={FINAL_DELTA_BACKEND_CONFIG['delta_page_score_impl']}")
    
    DTYPE = torch.bfloat16
    checkpoint_path = config_args.model
    
    if config_args.full_cache_layers is not None:
        conditional_print(f"Selective KV Cache enabled:")
        conditional_print(f"  Full cache layers: {config_args.full_cache_layers}")
        conditional_print(f"  Subset cache ratio: {config_args.subset_cache_ratio}")
        conditional_print(f"  Compression ratio: {config_args.compression_ratio}")
        conditional_print(f"  Subset cache fixed size: {config_args.subset_cache_size} (overrides ratio if >0)")
        conditional_print(f"  L: {config_args.L}")
        conditional_print(f"  Subset cache layers: {[i for i in range(32) if i not in config_args.full_cache_layers]}")
    else:
        conditional_print("Using standard full KV cache for all layers")
    engine = LMBackend(
        dtype=DTYPE,
        device=DEVICE,
        **build_delta_backend_kwargs(config_args),
    )
    engine.load_model(checkpoint_path, use_tp=use_tp, rank_group=config_args.rank_group if config_args.rank_group else [0], group=global_group)
    engine.model.config.enable_selective_cache = config_args.enable_selective_cache
    engine.model.config.page_selector_version = FINAL_DELTA_BACKEND_CONFIG["page_selector_version"]
    engine.model.config.page_selector_v2 = True
    if config_args.full_cache_layers is not None:
        engine.model.config.full_cache_layers = config_args.full_cache_layers
        engine.model.config.subset_cache_ratio = config_args.subset_cache_ratio
        engine.model.config.compression_ratio = config_args.compression_ratio
        engine.model.config.subset_cache_size = config_args.subset_cache_size
        engine.model.config.L = config_args.L
        conditional_print(f"Applied selective cache config to model: full_cache_layers={config_args.full_cache_layers}")
        conditional_print("  Page selector backend: v2")
        conditional_print("  Using PageSelectorV2 (single-pass topk with exp2 scoring and position bias)")
    
    engine.setup_caches(max_batch_size=config_args.B, max_seq_length=config_args.max_len)
    
    tokenizer = load_tokenizer(
        config_args.model,
        config_args.model_name,
        verbose=(not use_tp or rank == config_args.rank_group[0]),
    )
    conditional_print(
        f"Tokenizer source: {getattr(tokenizer, '_resolved_source', getattr(tokenizer, 'name_or_path', '<unknown>'))}"
    )
    eot_1 = tokenizer.eos_token_id
    if tokenizer.unk_token_id is not None:
        eot_2 = tokenizer.unk_token_id
    else:
        eot_2 = tokenizer.encode("<|eot_id|>")[-1] if "<|eot_id|>" in tokenizer.get_vocab() else eot_1
    conditional_print(f"eot_1: {eot_1}, eot_2: {eot_2}")
    
    engine.set_tokenizer(tokenizer)
    
    return engine, tokenizer, DEVICE, DTYPE, eot_1, eot_2

setup_seed(args.seed)
engine, tokenizer, DEVICE, DTYPE, eot_1, eot_2 = setup_model_and_engine(args)

MAX_LEN = args.max_len
BATCH_SIZE = args.B

def setup_dataset(args, tokenizer):
    """Setup and load the selected dataset"""
    print(f"Loading {args.dataset} dataset...")
    
    try:
        if args.dataset == 'gsm8k':
            dataset = DatasetWrapper(GSM8k, tokenizer=tokenizer, num_samples=args.num_samples)
        elif args.dataset == 'math500':
            dataset = DatasetWrapper(Math500, tokenizer=tokenizer, num_samples=args.num_samples)
        elif args.dataset == 'aime':
            dataset = DatasetWrapper(AIME, tokenizer=tokenizer, num_samples=args.num_samples)
        elif args.dataset == 'aime2024':
            dataset = DatasetWrapper(AIME, tokenizer=tokenizer, num_samples=args.num_samples, year_filter=2024)
        elif args.dataset == 'aime2025':
            dataset = DatasetWrapper(AIME, tokenizer=tokenizer, num_samples=args.num_samples, year_filter=2025)
        elif args.dataset == 'gpqa':
            dataset = DatasetWrapper(Gpqa, tokenizer=tokenizer, num_samples=args.num_samples)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        print(f"Dataset loaded: {len(dataset)} samples available")
        
        if len(dataset) > 0:
            sample_input, sample_gt = dataset[0]
            print(f"Sample input length: {len(sample_input)} tokens")
            print(f"Sample ground truth: {sample_gt}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to load dataset {args.dataset}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

dataset = setup_dataset(args, tokenizer)

def run_evaluation(engine, dataset, args, eot_1, eot_2):
    """Run the main evaluation logic"""
    print(f"Using scheduler mode with batch size {args.B}")
    
    all_input_ids = []
    all_ground_truths = []

    print(f"Collecting samples from {args.dataset}...")
    for i in range(len(dataset)):
        input_ids, ground_truth = dataset[i]
        all_input_ids.append(input_ids.to(DEVICE))
        
        ground_truth = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        all_ground_truths.append(ground_truth)

    print(f"Collected {len(all_input_ids)} samples")

    input_lengths = [len(ids) for ids in all_input_ids]
    avg_input_length = sum(input_lengths) // len(input_lengths)
    max_lengths = [args.max_len] * len(all_input_ids)
    
    
    eot_token_ids = [eot_1, eot_2]
    # eot_token_ids = []

    print(f"Average input length: {avg_input_length}, Max generation length: {args.max_len}")

    engine.add_requests(all_input_ids, max_lengths, eot_token_ids)

    print("Starting scheduler...")
    start_time = time.perf_counter()
    torch.cuda.synchronize()

    detailed_results = engine.run_scheduler_loop()

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time

    return detailed_results, all_ground_truths, total_time

def process_evaluation_results(detailed_results, all_ground_truths, total_time, args):
    """Process the evaluation results and calculate metrics"""
    processed_samples = []
    total_generated_tokens = 0
    correct_count = 0

    for i, result in enumerate(detailed_results):
        if i < len(all_ground_truths):
            predicted_answer = extract_answer_from_generation(result['output_text'], args.dataset)
            ground_truth_answer = str(all_ground_truths[i]).strip()
            
            is_correct = False
            if predicted_answer != "N/A":
                try:
                    from DELTA.Data.Reasoning.utils import math_equal
                    is_correct = math_equal(predicted_answer, ground_truth_answer)
                except:
                    is_correct = (predicted_answer == ground_truth_answer)
            
            if is_correct:
                correct_count += 1
            
            total_generated_tokens += result['output_length']
            
            sample_info = {
                'sample_id': i,
                'request_id': result['request_id'],
                'input_text': result.get('input_text', ''),
                'input_length': result['input_length'],
                'output_text': result['output_text'],
                'output_length': result['output_length'],
                'ground_truth': all_ground_truths[i],
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'time_to_first_token': result.get('time_to_first_token', 0),
                'total_generation_time': result.get('total_time', 0),
                'tokens_per_second': result.get('tokens_per_second', 0)
            }
            processed_samples.append(sample_info)

    return processed_samples, correct_count, total_generated_tokens

detailed_results, all_ground_truths, total_time = run_evaluation(engine, dataset, args, eot_1, eot_2)
processed_samples, correct_count, total_generated_tokens = process_evaluation_results(detailed_results, all_ground_truths, total_time, args)

def create_comprehensive_results(processed_samples, total_time, args, DEVICE, DTYPE, BATCH_SIZE, MAX_LEN):
    """Create comprehensive results structure"""
    correct_count = sum(1 for s in processed_samples if s['correct'])
    total_generated_tokens = sum(s['output_length'] for s in processed_samples)
    total_input_tokens = sum(s['input_length'] for s in processed_samples)
    
    accuracy = correct_count / len(processed_samples) if processed_samples else 0.0
    avg_generation_length = total_generated_tokens / len(processed_samples) if processed_samples else 0
    avg_processing_time = sum(s['total_generation_time'] for s in processed_samples) / len(processed_samples) if processed_samples else 0
    overall_tokens_per_second = total_generated_tokens / total_time if total_time > 0 else 0

    return {
        'runtime_metrics': {
            'total_runtime_seconds': total_time,
            'total_samples_processed': len(processed_samples),
            'total_input_tokens': total_input_tokens,
            'total_generated_tokens': total_generated_tokens,
            'overall_tokens_per_second': overall_tokens_per_second,
            'average_generation_length': avg_generation_length,
            'average_processing_time_per_sample': avg_processing_time,
            'requests_per_second': len(processed_samples) / total_time if total_time > 0 else 0
        },
        'configuration': {
            'model_name': args.model_name,
            'model_path': str(args.model),
            'dataset': args.dataset,
            'batch_size': BATCH_SIZE,
            'max_sequence_length': MAX_LEN,
            'device': DEVICE,
            'dtype': str(DTYPE),
            'num_samples_requested': args.num_samples,
            'num_samples_processed': len(processed_samples),
            'tokenizer_source': getattr(tokenizer, '_resolved_source', getattr(tokenizer, 'name_or_path', None)),
            'tokenizer_loader': getattr(tokenizer, '_resolved_loader', None),
            'selective_cache_enabled': args.enable_selective_cache,
            'full_cache_layers': args.full_cache_layers,
            'subset_cache_ratio': args.subset_cache_ratio if args.enable_selective_cache else None,
            'compression_ratio': args.compression_ratio if args.enable_selective_cache else None,
            'subset_cache_size': args.subset_cache_size if args.enable_selective_cache else None,
            'L': args.L if args.enable_selective_cache else None,
            **delta_backend_config_fields(args, selective_cache_enabled=args.enable_selective_cache),
        },
        
        'accuracy_metrics': {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_count,
            'total_predictions': len(processed_samples),
            'accuracy_percentage': accuracy * 100
        },
        
        'token_statistics': {
            'min_input_length': min(s['input_length'] for s in processed_samples) if processed_samples else 0,
            'max_input_length': max(s['input_length'] for s in processed_samples) if processed_samples else 0,
            'avg_input_length': total_input_tokens / len(processed_samples) if processed_samples else 0,
            'min_output_length': min(s['output_length'] for s in processed_samples) if processed_samples else 0,
            'max_output_length': max(s['output_length'] for s in processed_samples) if processed_samples else 0,
            'avg_output_length': avg_generation_length,
            'total_tokens': total_input_tokens + total_generated_tokens
        },
        
        'timing_statistics': {
            'min_generation_time': min(s['total_generation_time'] for s in processed_samples) if processed_samples else 0,
            'max_generation_time': max(s['total_generation_time'] for s in processed_samples) if processed_samples else 0,
            'avg_generation_time': avg_processing_time,
            'min_tokens_per_second': min(s['tokens_per_second'] for s in processed_samples if s['tokens_per_second'] > 0) if processed_samples else 0,
            'max_tokens_per_second': max(s['tokens_per_second'] for s in processed_samples) if processed_samples else 0,
            'avg_tokens_per_second': sum(s['tokens_per_second'] for s in processed_samples) / len(processed_samples) if processed_samples else 0
        },
        
        'sample_results': processed_samples,
        
        'metadata': {
            'evaluation_timestamp': int(time.time()),
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'command_line_args': vars(args)
        }
    }

def save_results(comprehensive_results, processed_samples, args, BATCH_SIZE):
    """Save results to a single JSON file with readable datetime"""
    from datetime import datetime
    
    # Create readable timestamp
    now = datetime.now()
    timestamp_readable = now.strftime("%Y%m%d_%H%M%S")
    timestamp_iso = now.isoformat()
    
    # Create unified filename
    filename = f"{args.dataset}_results_{args.num_samples}samples_{BATCH_SIZE}batch_{timestamp_readable}.json"
    
    # Get metrics for convenience
    metrics = comprehensive_results['runtime_metrics']
    accuracy_metrics = comprehensive_results['accuracy_metrics']
    
    # Create unified results structure
    unified_results = {
        # Metadata with readable timestamps
        'metadata': {
            'evaluation_timestamp_readable': timestamp_readable,
            'evaluation_datetime_iso': timestamp_iso,
            'evaluation_date_human': now.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'command_line_args': vars(args)
        },
        
        # Configuration
        'configuration': comprehensive_results['configuration'],
        
        # Summary metrics (for quick access)
        'summary': {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'total_samples': len(processed_samples),
            'accuracy': accuracy_metrics['overall_accuracy'],
            'accuracy_percentage': accuracy_metrics['overall_accuracy'] * 100,
            'correct_predictions': accuracy_metrics['correct_predictions'],
            'total_runtime_seconds': metrics['total_runtime_seconds'],
            'tokens_per_second': metrics['overall_tokens_per_second'],
            'avg_generation_length': metrics['average_generation_length'],
            'avg_processing_time_per_sample': metrics['average_processing_time_per_sample']
        },
        
        # Detailed metrics
        'detailed_metrics': {
            'runtime_metrics': comprehensive_results['runtime_metrics'],
            'accuracy_metrics': comprehensive_results['accuracy_metrics'],
            'token_statistics': comprehensive_results['token_statistics'],
            'timing_statistics': comprehensive_results['timing_statistics']
        },
        
        # Sample-level results
        'sample_results': processed_samples,
        
        # Legacy format (for backward compatibility)
        'legacy_format': {
            'dataset': args.dataset,
            'num_samples': len(processed_samples),
            'batch_size': BATCH_SIZE,
            'max_length': args.max_len,
            'model_name': args.model_name,
            'predictions': [s['output_text'] for s in processed_samples],
            'ground_truths': [s['ground_truth'] for s in processed_samples],
            'generation_lengths': [s['output_length'] for s in processed_samples],
            'processing_times': [s['total_generation_time'] for s in processed_samples],
            'individual_results': processed_samples,
            'total_time': metrics['total_runtime_seconds'],
            'accuracy': accuracy_metrics['overall_accuracy'],
            'avg_generation_length': metrics['average_generation_length'],
            'avg_processing_time': metrics['average_processing_time_per_sample'],
            'total_correct': accuracy_metrics['correct_predictions'],
            'total_samples': len(processed_samples),
            'tokens_per_second': metrics['overall_tokens_per_second']
        }
    }
    
    # Save unified results to single file
    with open(filename, 'w') as f:
        json.dump(unified_results, f, indent=2, default=str)
    
    return filename

def print_final_results(comprehensive_results, args):
    """Print final evaluation results"""
    metrics = comprehensive_results['runtime_metrics']
    accuracy_metrics = comprehensive_results['accuracy_metrics']
    
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Total samples: {metrics['total_samples_processed']}")
    print(f"Accuracy: {accuracy_metrics['overall_accuracy']:.2%} ({accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_predictions']})")
    print(f"Average generation length: {metrics['average_generation_length']:.1f} tokens")
    print(f"Average processing time per sample: {metrics['average_processing_time_per_sample']:.3f}s")
    print(f"Total processing time: {metrics['total_runtime_seconds']:.2f}s")
    print(f"Overall throughput: {metrics['overall_tokens_per_second']:.2f} tokens/s")
    print(f"{'='*60}")

def print_selective_cache_stats(args):
    """Print selective cache statistics if enabled"""
    if args.enable_selective_cache:
        print("\nSelective KV Cache Statistics:")
        full_layers = args.full_cache_layers
        total_layers = 32  # Could get this from engine.model.config.n_layer
        subset_layers = [i for i in range(total_layers) if i not in full_layers]
        print(f"  Total layers: {total_layers}")
        print(f"  Full cache layers ({len(full_layers)}): {full_layers}")
        print(f"  Subset cache layers ({len(subset_layers)}): {subset_layers}")
        print(f"  Memory bandwidth reduction estimate: {len(subset_layers)/total_layers * (1-args.subset_cache_ratio) * 100:.1f}%")

def print_sample_results(processed_samples, args):
    """Print sample results for verification"""
    if args.printoutput and processed_samples:
        print(f"\n🔍 Sample evaluation results:")
        for i, sample in enumerate(processed_samples[:5]):
            print(f"\nSample {i+1} (ID: {sample['sample_id']}):")
            print(f"  Request ID: {sample['request_id']}")
            print(f"  Input length: {sample['input_length']} tokens")
            print(f"  Output length: {sample['output_length']} tokens")
            print(f"  Ground truth: {sample['ground_truth']}")
            print(f"  Predicted answer: {sample['predicted_answer']}")
            print(f"  Correct: {'✅' if sample['correct'] else '❌'}")
            print(f"  Generation time: {sample['total_generation_time']:.3f}s")
            print(f"  Tokens/sec: {sample['tokens_per_second']:.2f}")
            
            # Show truncated input/output text
            input_preview = sample['input_text'][:100] + "..." if len(sample['input_text']) > 100 else sample['input_text']
            output_preview = sample['output_text'][:150] + "..." if len(sample['output_text']) > 150 else sample['output_text']
            print(f"  Input text: {input_preview}")
            print(f"  Generated text: {output_preview}")

comprehensive_results = create_comprehensive_results(processed_samples, total_time, args, DEVICE, DTYPE, BATCH_SIZE, MAX_LEN)
results_filename = save_results(comprehensive_results, processed_samples, args, BATCH_SIZE)
print_final_results(comprehensive_results, args)
print(f"\n💾 All results saved to: {results_filename}")
print_selective_cache_stats(args)
print_sample_results(processed_samples, args)
print(f"\n🎯 Evaluation completed! Check {results_filename} for all results.")
