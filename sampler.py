import requests
from typing import List, Dict, Any, Union
from dataclasses import dataclass
import math
import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Config
n_logprobs = 50

config = {
    'api_url': 'http://localhost:8081',
    'logprobs': 10
}

@dataclass
class TokenProb:
    token: str
    probability: float

def get_logprobs(prompt: Union[str, List[int]], config: Dict[str, Any]) -> List[TokenProb]:
    url: str = config['api_url'] + '/completion'
    logprobs: int = config['logprobs']    
    
    payload: Dict[str, Union[str, bool, float, int]] = {
        'prompt': prompt,
        'cache_prompt': True,
        'n_predict': 1,
        'top_k': 50,
        'samplers': ['top_k'],
        'n_probs': logprobs
    }
    
    try:
        response: requests.Response = requests.post(url, json=payload)
        response.raise_for_status()
        probs: List[Dict[str, Union[str, float]]] = response.json()['completion_probabilities'][0]['probs']
    except Exception as e:
        print(f"Error fetching logprobs: {e}")
        probs = []
    
    return [TokenProb(prob['tok_str'], prob['prob']) for prob in probs]

class ModernSampler:
    def __init__(self,
                 enabled_methods: Dict[str, bool] = None,
                 repetition_penalty: float = 1.2,
                 dynamic_penalty_factor: float = 0.5,
                 entropy_clip_threshold: float = 1.0,
                 adaptive_temperature_base: float = 1.0,
                 cluster_boost: Dict[str, float] = None,
                 beam_width: int = 5,
                 parallel_cot: int = 3):
        """
        Initialize the ModernSampler with configurable sampling methods.

        Parameters:
        - enabled_methods: Dict specifying which methods to enable.
        - repetition_penalty: Base penalty for repetition.
        - dynamic_penalty_factor: Factor to adjust penalty dynamically.
        - entropy_clip_threshold: Threshold for entropy-based clipping.
        - adaptive_temperature_base: Base temperature for adaptive scaling.
        - cluster_boost: Dictionary for token cluster boosting.
        - beam_width: Number of beams for beam search.
        - parallel_cot: Number of parallel chains-of-thought.
        """
        default_methods = {
            'dry_repetition_penalty': True,
            'dynamic_repetition_penalty': True,
            'contextual_repetition_penalty': False,  # Placeholder
            'entropy_clipping': True,
            'adaptive_temperature': True,
            'token_cluster_boosting': False,
            'reasoning_chain_beam_search': False,
            'parallel_cot_sampling': False
        }
        if enabled_methods:
            default_methods.update(enabled_methods)
        self.enabled_methods = default_methods

        self.repetition_penalty = repetition_penalty
        self.dynamic_penalty_factor = dynamic_penalty_factor
        self.entropy_clip_threshold = entropy_clip_threshold
        self.adaptive_temperature_base = adaptive_temperature_base
        self.cluster_boost = cluster_boost if cluster_boost else {}
        self.beam_width = beam_width
        self.parallel_cot = parallel_cot
        self.token_history = {}
        self.lock = threading.Lock()  # To ensure thread-safe operations on token_history
    
    def apply_dry_repetition_penalty(self, tokens: List[TokenProb]) -> List[TokenProb]:
        """
        Applies a DRY-style repetition penalty to discourage repeated tokens.
        """
        if not self.enabled_methods.get('dry_repetition_penalty', False):
            return tokens
        penalized_tokens = []
        for token_prob in tokens:
            with self.lock:
                count = self.token_history.get(token_prob.token, 0)
            penalty = self.repetition_penalty
            adjusted_prob = token_prob.probability / penalty if count > 0 else token_prob.probability
            penalized_tokens.append(TokenProb(token_prob.token, adjusted_prob))
        return penalized_tokens
    
    def apply_dynamic_repetition_penalty(self, tokens: List[TokenProb]) -> List[TokenProb]:
        """
        Adjusts the repetition penalty dynamically based on repetition count.
        """
        if not self.enabled_methods.get('dynamic_repetition_penalty', False):
            return tokens
        penalized_tokens = []
        for token_prob in tokens:
            with self.lock:
                count = self.token_history.get(token_prob.token, 0)
            penalty = self.repetition_penalty + (count * self.dynamic_penalty_factor)
            adjusted_prob = token_prob.probability / penalty if count > 0 else token_prob.probability
            penalized_tokens.append(TokenProb(token_prob.token, adjusted_prob))
        return penalized_tokens
    
    def apply_contextual_repetition_penalty(self, tokens: List[TokenProb], context: str) -> List[TokenProb]:
        """
        Applies a contextual repetition penalty based on the given context.
        Placeholder for advanced contextual logic.
        """
        if not self.enabled_methods.get('contextual_repetition_penalty', False):
            return tokens
        # Implement contextual logic here
        # For example, penalize more in certain topics or contexts
        return tokens
    
    def entropy(self, tokens: List[TokenProb]) -> float:
        """
        Calculates the entropy of the current token probabilities.
        """
        probs = np.array([tp.probability for tp in tokens], dtype=np.float64)
        probs = probs / probs.sum()
        return -np.sum(probs * np.log(probs + 1e-12))
    
    def apply_entropy_clipping(self, tokens: List[TokenProb], current_entropy: float) -> List[TokenProb]:
        """
        Clips token probabilities based on entropy to reduce randomness.
        """
        if not self.enabled_methods.get('entropy_clipping', False):
            return tokens
        if current_entropy > self.entropy_clip_threshold:
            # Clip probabilities below a certain threshold
            threshold = np.percentile([tp.probability for tp in tokens], 25)
            clipped_tokens = [tp for tp in tokens if tp.probability >= threshold]
            return clipped_tokens
        return tokens
    
    def apply_adaptive_temperature(self, tokens: List[TokenProb], current_entropy: float) -> List[TokenProb]:
        """
        Adjusts the sampling temperature based on entropy to control randomness.
        """
        if not self.enabled_methods.get('adaptive_temperature', False):
            return tokens
        temperature = self.adaptive_temperature_base
        if current_entropy > self.entropy_clip_threshold:
            temperature *= 1.2  # Increase temperature to add randomness
        elif current_entropy < self.entropy_clip_threshold / 2:
            temperature *= 0.8  # Decrease temperature to make predictions more deterministic
        adjusted_tokens = [TokenProb(tp.token, math.pow(tp.probability, 1/temperature)) for tp in tokens]
        # Normalize probabilities
        total = sum(tp.probability for tp in adjusted_tokens)
        if total == 0:
            print("Total probability is zero after adaptive temperature adjustment.")
            return tokens
        return [TokenProb(tp.token, tp.probability / total) for tp in adjusted_tokens]
    
    def apply_token_cluster_boosting(self, tokens: List[TokenProb]) -> List[TokenProb]:
        """
        Boosts the probabilities of tokens that belong to specified clusters.
        """
        if not self.enabled_methods.get('token_cluster_boosting', False):
            return tokens
        boosted_tokens = []
        for tp in tokens:
            boost = self.cluster_boost.get(tp.token, 1.0)
            boosted_prob = tp.probability * boost
            boosted_tokens.append(TokenProb(tp.token, boosted_prob))
        # Normalize probabilities
        total = sum(tp.probability for tp in boosted_tokens)
        if total == 0:
            print("Total probability is zero after token cluster boosting.")
            return tokens
        return [TokenProb(tp.token, tp.probability / total) for tp in boosted_tokens]
    
    def reasoning_chain_beam_search(self, prompt: str, beams: List[str]) -> List[str]:
        """
        Implements beam search to maintain multiple reasoning paths.
        """
        if not self.enabled_methods.get('reasoning_chain_beam_search', False):
            return beams
        new_beams = []
        for beam in beams:
            tokens = get_logprobs(beam, config)
            if not tokens:
                continue
            tokens = self.apply_dry_repetition_penalty(tokens)
            tokens = self.apply_dynamic_repetition_penalty(tokens)
            tokens = self.apply_entropy_clipping(tokens, self.entropy(tokens))
            tokens = self.apply_adaptive_temperature(tokens, self.entropy(tokens))
            tokens = self.apply_token_cluster_boosting(tokens)
            top_tokens = heapq.nlargest(self.beam_width, tokens, key=lambda x: x.probability)
            for tp in top_tokens:
                new_beams.append(beam + tp.token)
        # Select top beams based on entropy or another heuristic
        scored_beams = []
        for beam in new_beams:
            tokens = get_logprobs(beam, config)
            if not tokens:
                entropy_score = 0
            else:
                entropy_score = self.entropy(tokens)
            scored_beams.append((entropy_score, beam))
        # Sort beams by entropy score descending and select top beams
        scored_beams.sort(key=lambda x: x[0], reverse=True)
        selected_beams = [beam for _, beam in scored_beams[:self.beam_width]]
        return selected_beams
    
    def parallel_cot_sampling(self, prompt: str) -> List[str]:
        """
        Generates multiple chains of thought in parallel using ThreadPoolExecutor.
        """
        if not self.enabled_methods.get('parallel_cot_sampling', False):
            return [prompt]
        
        def generate_cot(prompt: str) -> str:
            beams = [prompt]
            for _ in range(5):  # Assume 5 steps of reasoning
                beams = self.reasoning_chain_beam_search(prompt, beams)
                if not beams:
                    break
            # Select the beam with the highest entropy or any other heuristic
            if beams:
                return beams[0]
            else:
                return prompt
        
        cot_samples = []
        with ThreadPoolExecutor(max_workers=self.parallel_cot) as executor:
            # Launch parallel CoT generation tasks
            futures = [executor.submit(generate_cot, prompt) for _ in range(self.parallel_cot)]
            for future in as_completed(futures):
                try:
                    cot = future.result()
                    cot_samples.append(cot)
                except Exception as e:
                    print(f"Error generating CoT sample: {e}")
        
        return cot_samples
    
    def update_token_history(self, token: str):
        """
        Updates the token history with the newly generated token.
        """
        with self.lock:
            self.token_history[token] = self.token_history.get(token, 0) + 1
    
    def sample(self, prompt: str) -> str:
        """
        Samples the next token based on the enabled sampling methods.
        """
        tokens = get_logprobs(prompt, config)
        if not tokens:
            return ""
        
        # Update token history with the last token from the prompt
        last_token = prompt.split()[-1] if isinstance(prompt, str) and prompt.split() else ''
        if last_token:
            self.update_token_history(last_token)
        
        # Apply sampling strategies in sequence
        tokens = self.apply_dry_repetition_penalty(tokens)
        tokens = self.apply_dynamic_repetition_penalty(tokens)
        tokens = self.apply_contextual_repetition_penalty(tokens, prompt)
        current_entropy = self.entropy(tokens)
        tokens = self.apply_entropy_clipping(tokens, current_entropy)
        tokens = self.apply_adaptive_temperature(tokens, current_entropy)
        tokens = self.apply_token_cluster_boosting(tokens)
        
        # Normalize probabilities
        total_prob = sum(tp.probability for tp in tokens)
        if total_prob == 0:
            print("Total probability is zero after applying penalties. Returning empty string.")
            return ""
        normalized_tokens = [TokenProb(tp.token, tp.probability / total_prob) for tp in tokens]
        
        # Sample token based on adjusted probabilities
        tokens_list = [tp.token for tp in normalized_tokens]
        probs_list = [tp.probability for tp in normalized_tokens]
        try:
            chosen_token = np.random.choice(tokens_list, p=probs_list)
        except ValueError as e:
            print(f"Error during sampling: {e}")
            return ""
        
        return chosen_token
    
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generates a sequence of tokens based on the prompt and enabled sampling methods.
        """
        generated = prompt
        for _ in range(max_tokens):
            next_token = self.sample(generated)
            if not next_token:
                break
            generated += next_token
        return generated
    
    def generate_with_parallel_cot(self, prompt: str, max_tokens: int = 50) -> List[str]:
        """
        Generates multiple sequences using parallel chains-of-thought (CoT) with ThreadPoolExecutor.
        """
        if not self.enabled_methods.get('parallel_cot_sampling', False):
            return [self.generate(prompt, max_tokens)]
        
        cot_samples = self.parallel_cot_sampling(prompt)
        generated_samples = []
        
        def generate_full_text(cot: str) -> str:
            return self.generate(cot, max_tokens)
        
        with ThreadPoolExecutor(max_workers=self.parallel_cot) as executor:
            futures = [executor.submit(generate_full_text, cot) for cot in cot_samples]
            for future in as_completed(futures):
                try:
                    text = future.result()
                    generated_samples.append(text)
                except Exception as e:
                    print(f"Error generating full text from CoT sample: {e}")
        
        return generated_samples

# Example Usage
if __name__ == "__main__":
    # Define which sampling methods to enable
    enabled_methods = {
        'dry_repetition_penalty': True,
        'dynamic_repetition_penalty': True,
        'contextual_repetition_penalty': False,
        'entropy_clipping': True,
        'adaptive_temperature': True,
        'token_cluster_boosting': True,
        'reasoning_chain_beam_search': True,
        'parallel_cot_sampling': True
    }
    
    sampler = ModernSampler(
        enabled_methods=enabled_methods,
        repetition_penalty=1.2,
        dynamic_penalty_factor=0.3,
        entropy_clip_threshold=1.5,
        adaptive_temperature_base=1.0,
        cluster_boost={'science': 1.5, 'technology': 1.3},  # Example clusters
        beam_width=5,
        parallel_cot=10 
    )
    
    prompt = "In the realm of artificial intelligence,"
    
    # Sample the next token
    next_token = sampler.sample(prompt)
    print(f"Next Token: {next_token}")
    
    # Generate a full sequence
    generated_text = sampler.generate(prompt, max_tokens=20)
    print(f"Generated Text: {generated_text}")
    b
    # Generate multiple sequences using Parallel CoT Sampling
    cot_samples = sampler.generate_with_parallel_cot(prompt, max_tokens=20)
    for idx, cot in enumerate(cot_samples):
        print(f"CoT Sample {idx+1}: {cot}")
