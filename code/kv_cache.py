import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def compute_operations(d_model, sequence_lengths, num_layers=24, vocab_size=50257, batch_size=1):
    """
    Compute the number of operations required for transformer inference
    with and without KV Cache for different sequence lengths.
    
    Args:
        d_model (int): Model dimension
        sequence_lengths (list): List of sequence lengths to compute for
        num_layers (int): Number of transformer layers
        vocab_size (int): Size of vocabulary
        batch_size (int): Batch size
        
    Returns:
        dict: Dictionary containing operations data
    """
    results = {
        'sequence_length': sequence_lengths,
        'without_kv_cache': [],
        'with_kv_cache': [],
        'reduction_percentage': [],
        'speedup_factor': []
    }
    
    for s in sequence_lengths:
        # Without KV Cache: (2b*d_model^2*s^2 + 20b*d_model^2*s)*L + 2b*d_model*V*s
        ops_without_kv = (2 * batch_size * d_model**2 * s**2 + 20 * batch_size * d_model**2 * s) * num_layers + 2 * batch_size * d_model * vocab_size * s
        
        # With KV Cache: (24b*d_model^2*s)*L + 2b*d_model*V*s
        ops_with_kv = (24 * batch_size * d_model**2 * s) * num_layers + 2 * batch_size * d_model * vocab_size * s
        
        # Calculate reduction percentage
        reduction = (1 - ops_with_kv / ops_without_kv) * 100
        
        # Calculate speedup factor
        speedup = ops_without_kv / ops_with_kv
        
        results['without_kv_cache'].append(ops_without_kv)
        results['with_kv_cache'].append(ops_with_kv)
        results['reduction_percentage'].append(reduction)
        results['speedup_factor'].append(speedup)
    
    return results


def format_scientific(number):
    """Format large numbers in scientific notation"""
    return f"{number:.2e}"


def print_operations_table(d_model, sequence_lengths, num_layers=24, vocab_size=50257, batch_size=1):
    """
    Print a formatted table showing operations with and without KV Cache
    
    Args:
        d_model (int): Model dimension
        sequence_lengths (list): List of sequence lengths to compute for
        num_layers (int): Number of transformer layers
        vocab_size (int): Size of vocabulary
        batch_size (int): Batch size
    """
    results = compute_operations(d_model, sequence_lengths, num_layers, vocab_size, batch_size)
    
    # Prepare table data
    table_data = []
    for i, s in enumerate(results['sequence_length']):
        table_data.append([
            s,
            format_scientific(results['without_kv_cache'][i]),
            format_scientific(results['with_kv_cache'][i]),
            f"{results['reduction_percentage'][i]:.2f}%",
            f"{results['speedup_factor'][i]:.1f}x"
        ])
    
    # Print table
    headers = ["Sequence Length (s)", "Without KV Cache", "With KV Cache", 
               "Reduction Percentage", "Speedup Factor"]
    print(f"\nOperations comparison for GPT model with:")
    print(f"- Model dimension (d_model): {d_model}")
    print(f"- Number of layers: {num_layers}")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Batch size: {batch_size}\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def plot_operations(d_model, sequence_lengths, num_layers=24, vocab_size=50257, batch_size=1):
    """
    Create plots showing the efficiency gains from KV Cache
    
    Args:
        d_model (int): Model dimension
        sequence_lengths (list): List of sequence lengths to compute for
        num_layers (int): Number of transformer layers
        vocab_size (int): Size of vocabulary
        batch_size (int): Batch size
    """
    results = compute_operations(d_model, sequence_lengths, num_layers, vocab_size, batch_size)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'KV Cache Efficiency (GPT model, d_model = {d_model}, layers = {num_layers})', fontsize=16)
    
    # Plot 1: Operations count (log scale)
    axs[0, 0].plot(results['sequence_length'], results['without_kv_cache'], 'r-', label='Without KV Cache')
    axs[0, 0].plot(results['sequence_length'], results['with_kv_cache'], 'g-', label='With KV Cache')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel('Sequence Length')
    axs[0, 0].set_ylabel('Number of Operations (log scale)')
    axs[0, 0].set_title('Operations Count')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot 2: Reduction percentage
    axs[0, 1].plot(results['sequence_length'], results['reduction_percentage'], 'b-')
    axs[0, 1].set_xlabel('Sequence Length')
    axs[0, 1].set_ylabel('Reduction Percentage (%)')
    axs[0, 1].set_title('Reduction Percentage')
    axs[0, 1].grid(True)
    
    # Plot 3: Speedup factor
    axs[1, 0].plot(results['sequence_length'], results['speedup_factor'], 'm-')
    axs[1, 0].set_xlabel('Sequence Length')
    axs[1, 0].set_ylabel('Speedup Factor (x times)')
    axs[1, 0].set_title('Speedup Factor')
    axs[1, 0].grid(True)
    
    # Plot 4: Speedup factor (log scale)
    axs[1, 1].plot(results['sequence_length'], results['speedup_factor'], 'c-')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('Sequence Length')
    axs[1, 1].set_ylabel('Speedup Factor (log scale)')
    axs[1, 1].set_title('Speedup Factor (Log Scale)')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('kv_cache_efficiency.png')
    plt.show()


def print_formula_explanation():
    """Print explanation of the formulas used in the computation"""
    print("\nFormulas used in computation:")
    print("-----------------------------")
    print("Without KV Cache:")
    print("  FLOPs = (2b*d_model²*s² + 20b*d_model²*s)*L + 2b*d_model*V*s")
    print("  Where:")
    print("  - b: batch size")
    print("  - d_model: model dimension")
    print("  - s: sequence length")
    print("  - L: number of layers")
    print("  - V: vocabulary size")
    print("\nWith KV Cache:")
    print("  FLOPs = (24b*d_model²*s)*L + 2b*d_model*V*s")
    print("\nReduction comes from avoiding recomputation of keys and values for previous tokens.")


def test_transformer_kv_cache(model_name="gpt2", prompt="Hello, I'm a language model", 
                             num_new_tokens=50, num_runs=5, use_gpu=False):
    """
    Test the performance difference between using and not using KV cache with a transformer model.
    
    Args:
        model_name (str): Name of the Hugging Face model to use
        prompt (str): Input prompt for the model
        num_new_tokens (int): Number of new tokens to generate
        num_runs (int): Number of runs to average results over
        use_gpu (bool): Whether to use GPU for inference
        
    Returns:
        dict: Dictionary containing timing results
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_length = input_ids.shape[1]
    
    results = {
        "with_kv_cache": [],
        "without_kv_cache": []
    }
    
    print(f"Running inference with model: {model_name}")
    print(f"Input prompt: '{prompt}' (Length: {input_length} tokens)")
    print(f"Generating {num_new_tokens} new tokens, averaging over {num_runs} runs\n")
    
    # Test both with and without KV cache in a single loop
    for use_kv_cache in [False, True]:
        cache_status = "with" if use_kv_cache else "without"
        print(f"Testing {cache_status} KV cache...")
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Generate using model.generate with appropriate use_cache setting
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=num_new_tokens,
                    use_cache=use_kv_cache,
                    do_sample=False,  # Deterministic generation (greedy)
                    pad_token_id=tokenizer.eos_token_id
                )
            
            elapsed = time.time() - start_time
            results[f"{cache_status}_kv_cache"].append(elapsed)
            print(f"  Run {run+1}/{num_runs}: {elapsed:.4f} seconds")
        
        avg_time = sum(results[f"{cache_status}_kv_cache"]) / num_runs
        print(f"Average time {cache_status} KV cache: {avg_time:.4f} seconds\n")
    
    # Calculate speedup
    avg_time_without_kv = sum(results["without_kv_cache"]) / num_runs
    avg_time_with_kv = sum(results["with_kv_cache"]) / num_runs
    speedup = avg_time_without_kv / avg_time_with_kv
    reduction_percentage = (1 - avg_time_with_kv / avg_time_without_kv) * 100
    
    print("Results summary:")
    print(f"- Without KV cache: {avg_time_without_kv:.4f} seconds")
    print(f"- With KV cache: {avg_time_with_kv:.4f} seconds")
    print(f"- Speedup factor: {speedup:.2f}x")
    print(f"- Time reduction: {reduction_percentage:.2f}%")
    
    # Generate and print the output text (using KV cache for final output)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nGenerated text: '{output_text}'")
    
    return {
        "without_kv_cache": avg_time_without_kv,
        "with_kv_cache": avg_time_with_kv,
        "speedup_factor": speedup,
        "reduction_percentage": reduction_percentage,
        "input_length": input_length,
        "output_length": input_length + num_new_tokens,
        "model_name": model_name
    }


if __name__ == "__main__":
    # Model parameters (based on GPT3-medium)
    # d_model = 1024
    # num_layers = 24
    # vocab_size = 50257
    # batch_size = 1
    
    # # Sequence lengths to analyze
    # sequence_lengths = [10, 100, 500, 1000, 2000, 4000, 8000]
    
    # # Print formula explanation
    # print_formula_explanation()
    
    # # Print table of operations
    # print_operations_table(d_model, sequence_lengths, num_layers, vocab_size, batch_size)
    
    # # Create visualization
    # plot_operations(d_model, sequence_lengths, num_layers, vocab_size, batch_size)
    
    # Uncomment to run practical KV cache timing tests
    # Note: This requires the transformers library to be installed
    results = test_transformer_kv_cache(
        model_name="gpt2",
        prompt="Hello, I'm a language model",
        num_new_tokens=50,
        num_runs=3
    )