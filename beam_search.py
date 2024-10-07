import requests
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass
import math
import concurrent.futures

# Config
config = {
    'api_url': 'http://localhost:8081',
    'logprobs': 10
}

@dataclass
class TokenProb:
    token: str
    probability: float  # Can be raw probability or log-prob depending on API

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
        print(f"Error fetching logprobs for prompt '{prompt}': {e}")
        probs = []
    
    return [TokenProb(prob['tok_str'], prob['prob']) for prob in probs]

def beam_search(
    prompt: str,
    config: Dict[str, Any],
    beam_width: int = 3,
    max_length: int = 20,
    end_token: Optional[str] = None,
    normalize_scores: bool = True,
    num_parallel: int = 2,
    probs_are_log: bool = False  # Indicates if 'prob' is already a log-prob
) -> List[Tuple[str, float]]:
    """
    Performs beam search to generate sequences using parallel requests.

    Args:
        prompt (str): The initial prompt to start generation.
        config (Dict[str, Any]): Configuration dictionary for API access.
        beam_width (int, optional): Number of beams to keep. Defaults to 3.
        max_length (int, optional): Maximum length of the generated sequence. Defaults to 20.
        end_token (Optional[str], optional): Token that signifies the end of generation. Defaults to None.
        normalize_scores (bool, optional): Whether to normalize scores by token length. Defaults to True.
        num_parallel (int, optional): Number of parallel streams to use for API requests. Defaults to 2.
        probs_are_log (bool, optional): Whether the 'prob' returned by the API is already a log-prob. Defaults to False.

    Returns:
        List[Tuple[str, float]]: List of generated sequences with their scores.
    """
    # Each beam is a tuple: (sequence, cumulative_score, token_count)
    beams: List[Tuple[str, float, int]] = [(prompt, 0.0, 0)]
    completed_beams: List[Tuple[str, float]] = []

    for step in range(max_length):
        all_candidates: List[Tuple[str, float, int]] = []

        def process_beam(beam: Tuple[str, float, int]) -> List[Tuple[str, float, int]]:
            """Function to process each beam in parallel."""
            seq, score, token_count = beam

            # If the sequence is already completed, return it as is
            if end_token and seq.endswith(end_token):
                return [(seq, score, token_count)]

            # Get log-probabilities for the next token
            next_token_probs = get_logprobs(seq, config)

            # If no probabilities are returned, treat as completed
            if not next_token_probs:
                return [(seq, score, token_count)]

            candidates = []
            for token_prob in next_token_probs:
                token = token_prob.token
                prob = token_prob.probability

                if probs_are_log:
                    token_log_prob = prob
                else:
                    # Avoid math domain error by ensuring prob > 0
                    if prob > 0:
                        token_log_prob = math.log(prob)
                    else:
                        token_log_prob = -math.inf

                # Append the new token to the sequence
                new_seq = seq + token

                # Update the cumulative log-probability
                new_score = score + token_log_prob

                # Update token count
                new_token_count = token_count + 1

                candidates.append((new_seq, new_score, new_token_count))
            return candidates

        # Use ThreadPoolExecutor to process multiple beams concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = [executor.submit(process_beam, beam) for beam in beams]
            for future in concurrent.futures.as_completed(futures):
                all_candidates.extend(future.result())

        # If no candidates were generated, terminate
        if not all_candidates:
            break

        # Sort all candidates by score (descending) and select top beam_width
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

        # Optionally normalize scores by token length
        if normalize_scores:
            beams = [(seq, score / x[2], x[2]) for seq, score, x in zip([b[0] for b in beams], [b[1] for b in beams], beams)]

        # Debug: Print intermediate beams
        print(f"Step {step + 1}:")
        for i, (seq, score, token_count) in enumerate(beams, 1):
            print(f"  Beam {i}: '{seq}' | Score: {score:.4f} | Tokens: {token_count}")
        print("-" * 50)

    # After generation, add any remaining beams to completed_beams
    for beam in beams:
        seq, score, token_count = beam
        if normalize_scores:
            normalized_score = score  # Already normalized
        else:
            normalized_score = score
        completed_beams.append((seq, normalized_score))

    # Sort all completed beams by score (descending) and select top beam_width
    completed_beams.sort(key=lambda x: x[1], reverse=True)
    return completed_beams[:beam_width]

# Example Usage
if __name__ == "__main__":
    initial_prompt = "Once upon a time"
    beams = beam_search(
        prompt=initial_prompt,
        config=config,
        beam_width=3,
        max_length=50,
        end_token=".",
        normalize_scores=False,
        num_parallel=3,  # Use 3 parallel streams
        probs_are_log=False  # Set to True if API returns log-probabilities
    )

    print("\nFinal Generated Sequences:")
    for i, (seq, score) in enumerate(beams, 1):
        print(f"{i}: {seq} (Score: {score:.4f})")
