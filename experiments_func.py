def run_experiment(model_3b, model_1b, tokenizer):
    # Base parameters (will be used when not being varied)
    base_prompt_length = 256  # in tokens
    base_max_tokens = 128     # maximum number of tokens to generate
    base_n = 2                # base number of speculative tokens
    
    # Parameters to test (varying one at a time)
    prompt_lengths = [64, 128, 256, 512, 1024]  # in tokens
    max_tokens_values = [32, 64, 128, 256, 512]
    n_values = [1, 2, 4, 8, 16, 32]
    
    num_samples = 5  # Number of runs for each parameter set for statistical significance
    
    # Generate a base prompt that we'll trim to the desired token length
    base_prompt = """
    The development of artificial intelligence has been a fascinating journey from simple rule-based systems to complex neural networks capable of generating human-like text and images. Recent advancements in large language models have revolutionized how we interact with technology, enabling more natural and intuitive interfaces.
    
    These models are trained on vast amounts of text data, learning patterns and relationships between words and concepts. Their ability to generate coherent and contextually relevant responses has improved dramatically in recent years, though challenges remain in areas such as factual accuracy, bias, and understanding of complex reasoning.
    
    Beyond language models, computer vision has seen remarkable progress with convolutional neural networks and transformers enabling systems to recognize objects, understand scenes, and generate realistic images. Reinforcement learning has allowed AI to master complex games and tasks, and self-supervised learning approaches have reduced the need for labeled data.
    
    The field continues to evolve with research into multimodal models that can process both text and images, more efficient architectures that reduce computational requirements, and approaches that enhance robustness and safety. As these technologies advance, they raise important questions about their impact on society, employment, privacy, and the future relationship between humans and intelligent machines.
    
    Ensuring that AI development benefits humanity broadly remains a crucial consideration for researchers, companies, and policymakers alike. This includes addressing issues of bias, transparency, accountability, and developing approaches that augment human capabilities rather than replacing them. The future of AI holds tremendous potential for positive impact across domains from healthcare and education to scientific discovery and sustainable development.
    """
    base_prompt = base_prompt * 10  # Make it long enough for our experiments
    
    # Create DataFrame to store results
    results = []
    
    # Main progress bar for overall experiment
    main_pbar = tqdm(total=len(prompt_lengths) + len(max_tokens_values) + len(n_values) + 1, 
                     desc="Overall Experiment Progress")
    
    # Experiment 1: Effect of prompt length (only varying prompt_length)
    for prompt_length in tqdm(prompt_lengths, desc="Experiment 1: Prompt Length", leave=False):
        # Tokenize and decode to get exact token length
        tokenized = tokenizer.encode(base_prompt)[:prompt_length]
        prompt = tokenizer.decode(tokenized)
        
        # Perform standard generation (without SpecDec)
        baseline_times = []
        for _ in tqdm(range(num_samples), desc=f"Normal Generation", leave=False):
            # Clear GPU memory if available
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            normal_generate(model_3b, prompt, base_max_tokens)
            end_time = time.time()
            baseline_times.append(end_time - start_time)
        
        avg_baseline_time = np.mean(baseline_times)
        std_baseline_time = np.std(baseline_times)
        
        results.append({
            'experiment': 'prompt_length',
            'prompt_length': prompt_length,
            'max_tokens': base_max_tokens,
            'n': None,
            'method': 'Normal',
            'time': avg_baseline_time,
            'time_std': std_baseline_time,  # Standard deviation for error bars
            'acceptance_rate': None
        })
        
        # Perform speculative generation
        spec_times = []
        acceptance_rates = []
        
        for _ in tqdm(range(num_samples), desc=f"Speculative Generation (n={base_n})", leave=False):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            _, accepted, total = speculative_generate(model_3b, model_1b, prompt, base_max_tokens, base_n)
            end_time = time.time()
            
            spec_times.append(end_time - start_time)
            acceptance_rates.append(accepted / total if total > 0 else 0)
        
        avg_spec_time = np.mean(spec_times)
        std_spec_time = np.std(spec_times)
        avg_acceptance_rate = np.mean(acceptance_rates)
        
        results.append({
            'experiment': 'prompt_length',
            'prompt_length': prompt_length,
            'max_tokens': base_max_tokens,
            'n': base_n,
            'method': 'Speculative',
            'time': avg_spec_time,
            'time_std': std_spec_time,
            'acceptance_rate': avg_acceptance_rate
        })
        
        main_pbar.update(1)
    
    # Experiment 2: Effect of max_tokens (only varying max_tokens)
    # Get prompt of base length
    tokenized = tokenizer.encode(base_prompt)[:base_prompt_length]
    prompt = tokenizer.decode(tokenized)
    
    for max_tokens in tqdm(max_tokens_values, desc="Experiment 2: Max Tokens", leave=False):
        # Perform standard generation
        baseline_times = []
        for _ in tqdm(range(num_samples), desc=f"Normal Generation", leave=False):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            normal_generate(model_3b, prompt, max_tokens)
            end_time = time.time()
            baseline_times.append(end_time - start_time)
        
        avg_baseline_time = np.mean(baseline_times)
        std_baseline_time = np.std(baseline_times)
        
        results.append({
            'experiment': 'max_tokens',
            'prompt_length': base_prompt_length,
            'max_tokens': max_tokens,
            'n': None,
            'method': 'Normal',
            'time': avg_baseline_time,
            'time_std': std_baseline_time,
            'acceptance_rate': None
        })
        
        # Perform speculative generation
        spec_times = []
        acceptance_rates = []
        
        for _ in tqdm(range(num_samples), desc=f"Speculative Generation (n={base_n})", leave=False):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            _, accepted, total = speculative_generate(model_3b, model_1b, prompt, max_tokens, base_n)
            end_time = time.time()
            
            spec_times.append(end_time - start_time)
            acceptance_rates.append(accepted / total if total > 0 else 0)
        
        avg_spec_time = np.mean(spec_times)
        std_spec_time = np.std(spec_times)
        avg_acceptance_rate = np.mean(acceptance_rates)
        
        results.append({
            'experiment': 'max_tokens',
            'prompt_length': base_prompt_length,
            'max_tokens': max_tokens,
            'n': base_n,
            'method': 'Speculative',
            'time': avg_spec_time,
            'time_std': std_spec_time,
            'acceptance_rate': avg_acceptance_rate
        })
        
        main_pbar.update(1)
    
    # Experiment 3: Effect of n (only varying n - number of speculative tokens)
    # Use prompt of base length
    
    # First perform standard generation (once, as n doesn't affect standard generation)
    baseline_times = []
    for _ in tqdm(range(num_samples), desc="Normal Generation (baseline)", leave=False):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_time = time.time()
        normal_generate(model_3b, prompt, base_max_tokens)
        end_time = time.time()
        baseline_times.append(end_time - start_time)
    
    avg_baseline_time = np.mean(baseline_times)
    std_baseline_time = np.std(baseline_times)
    
    results.append({
        'experiment': 'n_value',
        'prompt_length': base_prompt_length,
        'max_tokens': base_max_tokens,
        'n': None,
        'method': 'Normal',
        'time': avg_baseline_time,
        'time_std': std_baseline_time,
        'acceptance_rate': None
    })
    
    main_pbar.update(1)
    
    # Now test different values of n
    for n in tqdm(n_values, desc="Experiment 3: Speculative Tokens (n)", leave=False):
        spec_times = []
        acceptance_rates = []
        
        for _ in tqdm(range(num_samples), desc=f"Speculative Generation (n={n})", leave=False):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            _, accepted, total = speculative_generate(model_3b, model_1b, prompt, base_max_tokens, n)
            end_time = time.time()
            
            spec_times.append(end_time - start_time)
            acceptance_rates.append(accepted / total if total > 0 else 0)
        
        avg_spec_time = np.mean(spec_times)
        std_spec_time = np.std(spec_times)
        avg_acceptance_rate = np.mean(acceptance_rates)
        
        results.append({
            'experiment': 'n_value',
            'prompt_length': base_prompt_length,
            'max_tokens': base_max_tokens,
            'n': n,
            'method': 'Speculative',
            'time': avg_spec_time,
            'time_std': std_spec_time,
            'acceptance_rate': avg_acceptance_rate
        })
        
        main_pbar.update(1)
    
    main_pbar.close()
    return pd.DataFrame(results)

# Generate plots
def create_plots(results_df):
    # 1. Effect of prompt length on generation time
    plt.figure(figsize=(12, 6))
    
    # Select only data for prompt length experiment
    prompt_df = results_df[results_df['experiment'] == 'prompt_length']
    
    # Add error bars (standard deviation)
    for method in ['Normal', 'Speculative']:
        subset = prompt_df[prompt_df['method'] == method]
        
        # Plot with standard deviation error bars
        plt.errorbar(subset['prompt_length'], subset['time'], 
                    yerr=subset['time_std'], 
                    marker='o', linestyle='-', capsize=5,
                    label=f"{method} Generation" + (f" (n={subset['n'].iloc[0]})" if method == 'Speculative' else ""))
    
    plt.xlabel('Prompt Length (tokens)')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title('Effect of Prompt Length on Generation Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('prompt_length_effect.png')
    plt.close()
    
    # 2. Effect of max_tokens on generation time
    plt.figure(figsize=(12, 6))
    
    # Select only data for max_tokens experiment
    max_tokens_df = results_df[results_df['experiment'] == 'max_tokens']
    
    for method in ['Normal', 'Speculative']:
        subset = max_tokens_df[max_tokens_df['method'] == method]
        
        plt.errorbar(subset['max_tokens'], subset['time'], 
                    yerr=subset['time_std'],
                    marker='o', linestyle='-', capsize=5,
                    label=f"{method} Generation" + (f" (n={subset['n'].iloc[0]})" if method == 'Speculative' else ""))
    
    plt.xlabel('Maximum Generated Tokens')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title('Effect of Generated Tokens Count on Generation Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('max_tokens_effect.png')
    plt.close()
    
    # 3. Effect of n (speculative tokens) on generation time
    plt.figure(figsize=(12, 6))
    
    # Select only data for n experiment
    n_df = results_df[results_df['experiment'] == 'n_value']
    
    # For standard generation - just a horizontal line (baseline)
    normal_time = n_df[n_df['method'] == 'Normal']['time'].iloc[0]
    plt.axhline(y=normal_time, color='r', linestyle='-', label='Normal Generation')
    
    # For speculative generation - plot by n value
    spec_df = n_df[n_df['method'] == 'Speculative']
    plt.errorbar(spec_df['n'], spec_df['time'], 
                yerr=spec_df['time_std'],
                marker='o', linestyle='-', capsize=5,
                label='Speculative Generation')
    
    plt.xlabel('Speculative Tokens (n)')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title('Effect of Speculative Tokens on Generation Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('n_effect.png')
    plt.close()
    
    # 4. Additional plot: Token acceptance rate vs n
    plt.figure(figsize=(12, 6))
    
    # Select only data for n experiment and only speculative generation
    spec_n_df = n_df[n_df['method'] == 'Speculative']
    
    plt.plot(spec_n_df['n'], spec_n_df['acceptance_rate'], 
            marker='o', linestyle='-')
    
    plt.xlabel('Speculative Tokens (n)')
    plt.ylabel('Token Acceptance Rate')
    plt.title('Effect of Speculative Tokens on Token Acceptance Rate')
    plt.grid(True)
    plt.savefig('acceptance_rate.png')
    plt.close()
    
    # 5. Speedup plot - how much faster speculative generation is
    plt.figure(figsize=(12, 6))
    
    # For n experiment
    n_normal = n_df[n_df['method'] == 'Normal']['time'].iloc[0]
    n_spec = n_df[n_df['method'] == 'Speculative']
    
    # Calculate speedup factor
    speedup = n_normal / n_spec['time']
    
    plt.plot(n_spec['n'], speedup, marker='o', linestyle='-', color='green')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No speedup')
    
    plt.xlabel('Speculative Tokens (n)')
    plt.ylabel('Speedup Factor (×)')
    plt.title('Generation Speedup as a Function of Speculative Tokens')
    plt.grid(True)
    plt.legend()
    plt.savefig('speedup.png')
    plt.close()