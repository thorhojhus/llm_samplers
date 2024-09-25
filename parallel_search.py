import numpy as np
import requests
import graphviz

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Config
config = {
    'api_url': 'http://localhost:8081',
    'logprobs': 10
}

@dataclass
class TokenProb:
    token: str
    probability: float

@dataclass
class GenerationStream:
    prompt: str
    tokens_generated: int = 0
    path: List[str] = field(default_factory=list)
    completed: bool = False
    stop_reason: Optional[str] = None

def get_logprobs(prompt: Union[str, List[int]], config: Dict[str, Any]) -> List[TokenProb]:
    url: str = config['api_url'] + '/completion'
    logprobs: int = config['logprobs']    
    
    payload: Dict[str, Union[str, bool, float, int]] = {
        'prompt': prompt,
        'cache_prompt': True,
        'temperature': 1.0,
        'n_predict': 1,
        'top_k': 10,
        'top_p': 1.0,
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

def generate_next_token(stream: GenerationStream, config: Dict[str, Any],
                       threshold: float, stop_tokens: List[str]) -> List[GenerationStream]:
    if stream.completed:
        return []
    
    logprobs = get_logprobs(stream.prompt, config)
    if not logprobs:
        stream.completed = True
        stream.stop_reason = 'No probabilities returned'
        return []
    
    # Identify tokens above the threshold
    splits = [tp for tp in logprobs if tp.probability >= threshold]
    new_streams = []
    
    for tp in splits:
        new_prompt = stream.prompt + tp.token
        new_path = stream.path + [tp.token]
        
        # Check for stop tokens
        if any(new_prompt.endswith(stop_token) for stop_token in stop_tokens):
            completed_stream = GenerationStream(
                prompt=new_prompt,
                tokens_generated=stream.tokens_generated + 1,
                path=new_path,
                completed=True,
                stop_reason='Stop token encountered'
            )
            new_streams.append(completed_stream)
            continue
        
        # Create a new active stream
        new_stream = GenerationStream(
            prompt=new_prompt,
            tokens_generated=stream.tokens_generated + 1,
            path=new_path
        )
        new_streams.append(new_stream)
    
    # If no splits, continue with the highest probability token
    if not new_streams:
        top_token = logprobs[0]
        new_prompt = stream.prompt + top_token.token
        new_path = stream.path + [top_token.token]
        
        # Check for stop tokens
        if any(new_prompt.endswith(stop_token) for stop_token in stop_tokens):
            stream.completed = True
            stream.stop_reason = 'Stop token encountered'
        else:
            stream.prompt = new_prompt
            stream.tokens_generated += 1
        
        new_streams.append(stream)
    
    return new_streams

def parallel_search(initial_prompt: str, config: Dict[str, Any],
                   threshold: float = 0.2, max_tokens: int = 10,
                   max_paths: int = 10, max_threads: int = 4,
                   stop_tokens: List[str] = []) -> List[GenerationStream]:
    
    active_streams: List[GenerationStream] = [GenerationStream(prompt=initial_prompt)]
    completed_streams: List[GenerationStream] = []
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        while active_streams and len(completed_streams) < max_paths:
            futures = {executor.submit(generate_next_token, stream, config, threshold, stop_tokens): stream 
                       for stream in active_streams}
            active_streams = []
            
            for future in as_completed(futures):
                new_streams = future.result()
                for ns in new_streams:
                    if ns.completed:
                        with lock:
                            completed_streams.append(ns)
                            if len(completed_streams) >= max_paths:
                                break
                    else:
                        active_streams.append(ns)
                        if len(active_streams) + len(completed_streams) >= max_paths:
                            break
                if len(completed_streams) >= max_paths:
                    break
            
            if len(completed_streams) + len(active_streams) > max_paths:
                excess = (len(completed_streams) + len(active_streams)) - max_paths
                active_streams = active_streams[:-excess] if excess < len(active_streams) else []
        
        for stream in active_streams:
            if stream.tokens_generated >= max_tokens:
                stream.completed = True
                stream.stop_reason = 'Max tokens reached'
                completed_streams.append(stream)
                if len(completed_streams) >= max_paths:
                    break
            else:
                while stream.tokens_generated < max_tokens and not stream.completed:
                    next_streams = generate_next_token(stream, config, threshold, stop_tokens)
                    if not next_streams:
                        break
                    for ns in next_streams:
                        if ns.completed:
                            completed_streams.append(ns)
                            if len(completed_streams) >= max_paths:
                                break
                        else:
                            stream = ns
                    if len(completed_streams) >= max_paths:
                        break
        return completed_streams[:max_paths]


def build_graph(strings, prompt):
    pass


if __name__ == "__main__":
    prompt = "The quick brown fox jumps over the lazy dog."
    threshold = 0.2
    max_tokens = 5
    max_paths = 10
    max_threads = 4
    stop_tokens = ['\n', '.', '!', '?']
    
    results = parallel_search(
        initial_prompt=prompt,
        config=config,
        threshold=threshold,
        max_tokens=max_tokens,
        max_paths=max_paths,
        max_threads=max_threads,
        stop_tokens=stop_tokens
    )
    
    for idx, stream in enumerate(results, 1):
        print(f"Path {idx}:")
        print(f"Prompt: {stream.prompt}")
        print(f"Tokens Generated: {stream.tokens_generated}")
        print(f"Completed: {stream.completed} ({stream.stop_reason})")
        print(f"Path Tokens: {''.join(stream.path)}")
        print("-" * 40)    
