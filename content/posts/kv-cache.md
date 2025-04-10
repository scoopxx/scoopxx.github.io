---
title: "KV Cache Explained"
date: 2025-03-03
ShowToc: true
TocOpen: false
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

### Test KV Cache in Huggingface's transformers
We can test the effectiveness of KV Cache using [huggingface's transformers](https://github.com/huggingface/transformers/tree/main).
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

### A brief peek into transformer's KV Cache implementation
To better understand KV Cache, we can look at the transformer's KV Cache implementation.

Let's use GPT2 as an example. 
The `GPT2Attention.forward` takes a `use_cache` boolean argument, it will return current KV matriices if `use_cache=True`.

```python
#src/transformers/models/gpt2/modeling_gpt2.py
class GPT2Attention(nn.Module):
    def forward(...,   use_cache: Optional[bool] = False):
        ...
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
```

The `GPT2Block` class does similar things, then `GPT2Model.forward` will output the KV matrics for all layers.
```python
#src/transformers/models/gpt2/modeling_gpt2.py
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self):
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(..., past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
use_cache: Optional[bool] = False):
        ...
        # presents is used to store KV matrics for all layers.
        presents = () if use_cache else None
        for i in range(len(self.h)):
            # Get previous KV matrics from input.
            block, layer_past = self.h[i], past_key_values[i]
            outputs = block(input_ids, layer_past=layer_past, use_cache=use_cache)
            if use_cache is True:
                presents = presents + (outputs[1],)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
```

The KV Cache in `past_key_values` of `GPT2Model.forward` is a `BaseModelOutputWithPastAndCrossAttentions`. It's of shape `(num_layers, 2)`, where the first dimension corresponds to the layer index and the second dimension is `key` at index 0 and `value` at index 1. Then each tensor is of shape `(batch_size, num_heads, seq_len, head_dim)`.

During generation, a `DynamicCache` instance is created in `GenerationMixin`.

```python
#/src/transformers/src/transformers/generation/utils.py
class GenerationMixin:
    ...
    def _prepare_cache_for_generation(self, model_kwargs: Dict[str, Any]):
        ...
        cache_name = "past_key_values"
        model_kwargs[cache_name] = DynamicCache()

#src/transformers/cache_utils.py
class DynamicCache(Cache):
    def __init__(self):
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens on layer 0.
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            # Initialization phase, the layer cache not there yet.
            if len(self.key_cache) <= layer_idx:
                ...
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                # Otherwise, only append current key and value to the cache.
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
```

Then the KV Cache is loaded and used for generation in `_sample`:
```python
src/transformers/generation/utils.py
class GenerationMixin:
    ...
    def _sample(self, ...):
        ...
        while self._has_unfinished_sequences():
            # Prepare KV Cache is in prepare_inputs_for_generation
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = model_forward(**model_inputs, return_dict=True)
        ...
        return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
```