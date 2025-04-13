
@torch.no_grad()
def speculative_generate(big_model, small_model, prefix: str, max_num_tokens: int, n: int) -> tuple[str, int, int]:
    """Vanilla speculative decoding for greedy generation.

    Args:
        big_model: original big HF model (verifier).
        small_model: small HF model (draft).
        prefix (str): prompt.
        max_num_tokens (int): max tokens to generate.
        n (int): number of tokens to speculate.

    Returns: generated text, number of accepted tokens, number of all tokens.
    """

    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    start_size = input_ids.size(1)
    cnt_accepted = 0
    cnt_all = 0
    while input_ids.size(1) - start_size < max_num_tokens:
        small_outputs = small_model.generate(
            input_ids=input_ids,
            max_new_tokens=n,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        small_generation = small_outputs.sequences
        num_generated_tokens = min(n, small_generation.size(1) - input_ids.size(1))

        if num_generated_tokens == 0:
            break

        candidate_tokens = small_generation[:, input_ids.size(1):input_ids.size(1) + num_generated_tokens]

        big_outputs = big_model(input_ids=input_ids)
        big_model_logits = big_outputs.logits[:, -1, :]
        big_model_generations = torch.argmax(big_model_logits, dim=-1, keepdim=True)

        mismatch = False
        for i in range(num_generated_tokens):
            if i == 0:
                if candidate_tokens[:, 0] != big_model_generations.squeeze():
                    mismatch = True
                    if i == 0:
                        input_ids = torch.cat(
                            (input_ids, big_model_generations),
                            dim=1
                        )
                    print(f"Accepted {i}/{num_generated_tokens} tokens")
                    cnt_accepted += i
                    cnt_all += num_generated_tokens
                    break
            else:
                context_ids = torch.cat([input_ids, candidate_tokens[:, :i]], dim=1)
                outputs = big_model(input_ids=context_ids)
                next_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                big_model_generations = torch.cat([big_model_generations, next_token], dim=1)

                if candidate_tokens[:, i] != next_token.squeeze():
                    mismatch = True
                    input_ids = torch.cat(
                        (input_ids, candidate_tokens[:, :i], next_token),
                        dim=1
                    )
                    print(f"Accepted {i}/{num_generated_tokens} tokens")
                    cnt_accepted += i
                    cnt_all += num_generated_tokens
                    break

        if not mismatch:
            input_ids = torch.cat((input_ids, candidate_tokens), dim=1)
            print(f"Accepted {num_generated_tokens}/{num_generated_tokens} tokens")
            cnt_accepted += num_generated_tokens
            cnt_all += num_generated_tokens

        if input_ids.size(1) - start_size >= max_num_tokens:
            if input_ids.size(1) - start_size > max_num_tokens:
                input_ids = input_ids[:, :(start_size + max_num_tokens)]
            break

    decoded_text = tokenizer.decode(input_ids[0, start_size:], skip_special_tokens=True)
    return decoded_text, cnt_accepted, cnt_all




# Baseline normal generation function (without speculative decoding)
@torch.no_grad()
def normal_generate(model, prefix: str, max_num_tokens: int) -> str:
    """Normal generation with the big model only.
    
    Args:
        model: HF model for generation.
        prefix (str): prompt.
        max_num_tokens (int): max tokens to generate.
        
    Returns: generated text.
    """
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_num_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0, input_ids.size(1):], skip_special_tokens=True)