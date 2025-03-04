# KV Cache Operations Calculator

This script calculates and visualizes the computational efficiency gained by using KV Cache in transformer models.

## Features

- Calculates the number of operations required with and without KV Cache for different sequence lengths
- Prints a formatted table showing the comparison
- Generates visualizations of the efficiency gains
- Supports customizable model dimensions and sequence lengths

## Requirements

```
pip install numpy matplotlib tabulate
```

## Usage

Run the script with:

```
python kv_cache.py
```

## Output

The script will:

1. Print a table showing operations with and without KV Cache for various sequence lengths
2. Generate plots visualizing the efficiency gains
3. Save the plots as 'kv_cache_efficiency.png'

## Formulas Used

- Without KV Cache: `2d² + (2d²+2d)*(s²+s)`
- With KV Cache: `2d² + (4d²+4d)*s`

Where:
- `d` is the model dimension (default: 1024)
- `s` is the sequence length

## Example Output

For a model with dimension 1024 and sequence lengths [10, 100, 500, 1000, 2000, 4000, 8000], the script will calculate:

- Number of operations without KV Cache
- Number of operations with KV Cache
- Reduction percentage
- Speedup factor

The visualizations will show how these metrics change as sequence length increases.
