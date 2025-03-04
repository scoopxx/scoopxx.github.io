---
title: "KV Cache Explained"
date: 2025-03-03
ShowToc: true
TocOpen: true
tags: ["vllm", "KV Cache", "transformer"]
---

### What is KV Cache?
I am not intended to spend too much time on details of KV cache. But as a reference, I found the interpretation in this this post [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249) very intuitive, so I'll just steal the gif here.

![Comparison of self-attention with and without KV attention](/images/kv-cache.gif)

To summarize, in auto-regressive language model,when generating a new token, all its previous tokens are fed into the attention layer for computation. In an attention layer, denote the text input/generation sequence as $X$, where as $i$ th token is $x_i$. When in step $i$, we are predicting $X_i$, the formula is:
$$
q_{i} = embed_i * W_q \quad(1, d_{model})
$$
$$
k_{i} = embed_i * W_k \quad(1, d_{model})
$$
$$
v_{i} = embed_i * W_v \quad(1, d_{model})
$$
$$
K = concat(k_{0}, k_{1}, ..., k_{i})  \quad(i+1, d_{model})
$$
$$
Attn = softmax(q_{i} * K^T / \sqrt{d_{model}}) \quad(1, i+1)
$$
$$
Output = Attn * [v_{0}, v_{1}, ..., v_{i}]  \quad(1, d_{model})
$$

As we can see, at step $i$, its output is computed using that step's query $q_{i}$, as well as keys and values of all tokens up to $i$. So the intuition of KV Cache pretty straightforward: to store keys and values of all tokens up to $i$, so to avoid execssive computation during matrix multiplications.

### How many FLOPs are saved by KV Cache?
Let's run an analysis on the FLOPs of attention layer.

> **FLOPs for matrix multiplication:**
>
> If we are doing matrix multiplication between matrices of respective size of $(m, n)$ and $(n, p)$:
> 1. A signle multiplication is 1 operation.
> 2. A single addition is 1 operation.
> 3. Computing element at $(i, j)$ would take n multiplcaitions and (n-1)
> addtions, in total $2n - 1$ operations. 
>
> The output is a matrix of size $(m,p)$, and total operations is $(2n-1) * m * p$,
> we ignore the $-1$ notion for simplicity, so in total $2mnp$ operations.
>

Assuming we have GPT model with $n$ layers, each transformer block has $k$ heads. The model dimension is $d_{model}$, and each head has $d_{model} / k$ dimension. Assuming we are doing batch inference on $b$ samples with sequence length $s$.

#### Total flops without KV Cache: 

**1. Embedding Lookup**     
This part does not has arithmetic operations, only table lookups, ignore it.

**2. Self-Attention**       
For a self attention layer, at step $i$, 
1. Compute $Q$: compute $q_i$ only, $2b * d_{model}^2$ FLOPs.
$$
(b, 1, d_{model}) . (d_{model}, d_{model}) = (b, 1, d_{model})
$$
2. Compute $K$: compute $k_{0->i}$, $2b * i * d_{model}^2$ FLOPs.
$$
(b, i, d_{model}) . (d_{model}, d_{model}) = (b, i, d_{model})
$$
3. Compute $V$: similar to step 2, $2b * i * d_{model}^2$ FLOPs
4. $QK^T$, $2b * i * d_{model}$ FLOPs.
$$
(b, 1, d_{model}) . (b, i, d_{model}) = (b, 1, i)
$$
5. Weighted Value $attn*V$: $2b * i * d_{model}$ FLOPs.
$$
(b, 1, i) . (b, i, d_{model}) = (b, 1, d_model)
$$
6. Linear projection: $2b * d_{model}^2$ FLOPs.
$$
(b, 1, d_{model}) . (d_{model}, d_{model}) = (b, 1, d_{model})
$$

**3. MLP**  
There are two matrix multiplications in MLP, each with $8b*d_{model}^2$ FLOPs.

$$
(b, 1, d_{model}) . (d_{model}, 4d_{model}) = (b, 1, 4d_{model})
$$
$$
(b, 1, 4d_{model}) . (4d_{model}, d_{model}) = (b, 1, d_{model})
$$

**4. Final projection layer**
The final layer is to project the output to vocab size $V$, which is $2b * d_{model}* V$ FLOPs.
$$
(b, 1, d_{model}) . (d_{model}, V) = (b, 1, V)
$$

To sum these numbers up, as well as integral $i$ over $[1, s]$, in a GPT with $L$ layers, we have total flops: 

$$
FLOPs = (2b * d_{model}^2 * s^2 + 20b * d_{model}^2 * s) * L + 2b * d_{model} * V * s
$$


#### Flops with addtional KV Cache:
When KV Cache is used, the main optimization happened when computing $K$ and $V$ in self attention layer. Instead of doing matrix multiplication to compute $K_{j \in [0, i]}$ and $V_{j \in [0, i]}$, we cached and fetched $K_{j \in [0, i-1]}$ and $V_{j \in [0, i-1]}$, and only compute $K_j$ and $V_j$.       
The FLOPs at step $i$ is reduced from $2b \times d_{model}^2 \times i$ to $2b \times d_{model}^2 $. Integral over $i$, th quaratic part of $s$ decreasefrom $2bd_{model}^2s^2$ to $4bd_{model}^2s$.

The total FLOPs becomes:

$$
FLOPs_{sum_{i=1}^s} = (24b * d_{model}^2 * s) * L + 2b * d_{model} * V * s
$$

Without KV Cache, the operations scaled quadratically with the sequence length $s$. With KV Cache, the operations scale linearly with $s$, which makes it more efficient for longer sequences.

### FLOPs calculation with an example
Let's look at the FLOPs calculation using GPT3-medium as an example. Say we have:
$$
d_{model} = 1024, L = 24, V = 50257
$$ 

| Sequence Length (s) | Without KV Cache | With KV Cache | Reduction Percentage |
|---------------------|------------------|---------------|----------------------|
| 10 | $1.11 \times 10^{10}$ | $7.07 \times 10^9$ | 36.29% |
| 100 | $5.64 \times 10^{11}$ | $7.07 \times 10^{10}$ | 87.46% |
| 500 | $1.29 \times 10^{13}$ | $3.53 \times 10^{11}$ | 97.26% |
| 1000 | $5.09 \times 10^{13}$ | $7.07 \times 10^{11}$ | 98.61% |
| 2000 | $2.03 \times 10^{14}$ | $1.41 \times 10^{12}$ | 99.30% |
| 4000 | $8.08 \times 10^{14}$ | $2.83 \times 10^{12}$ | 99.65% |
| 8000 | $3.23 \times 10^{15}$ | $5.66 \times 10^{12}$ | 99.82% |

### Example run using KV Cache
Last not least, we can test the effectiveness of KV Cache using huggingface's transformer library.
```python 
def test_transformer_kv_cache(model_name="gpt2", prompt="Hello, I'm a language model", 
                             num_new_tokens=50, num_runs=5, use_gpu=False):
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
    return
```

We run `GPT2` on Google Colab with a T4 GPU. The results are as follows:
```
Using device: cuda
Running inference with model: gpt2
Input prompt: 'Hello, I'm a language model' (Length: 7 tokens)
Generating 1000 new tokens, averaging over 5 runs

Results summary:
- Without KV cache: 43.3307 seconds
- With KV cache: 8.3611 seconds
- Speedup factor: 5.18x
- Time reduction: 80.70%
```