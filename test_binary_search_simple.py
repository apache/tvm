#!/usr/bin/env python3

import numpy as np

def binary_search_test(scores, score_threshold):
    """Test binary search logic for score threshold"""
    num_boxes = len(scores)
    lo = 0
    hi = num_boxes
    
    while lo < hi:
        mid = (lo + hi) // 2
        if scores[mid] > score_threshold:
            lo = mid + 1
        else:
            hi = mid
    
    return lo

def test_score_threshold_logic():
    """Test score threshold logic step by step"""
    # Test case: scores [0.9, 0.3, 0.1], threshold 0.2
    scores = np.array([0.9, 0.3, 0.1])
    score_threshold = 0.2
    
    print(f"Scores: {scores}")
    print(f"Score threshold: {score_threshold}")
    
    # Expected: only scores 0.9 and 0.3 should be kept (indices 0, 1)
    # So valid_count should be 2
    valid_count = binary_search_test(scores, score_threshold)
    print(f"Binary search result: {valid_count}")
    print(f"Expected: 2 (indices 0 and 1 should be kept)")
    
    # Check which scores are actually > threshold
    valid_scores = scores[scores > score_threshold]
    print(f"Scores > threshold: {valid_scores}")
    print(f"Count of scores > threshold: {len(valid_scores)}")
    
    # The binary search should return the count of scores > threshold
    assert valid_count == len(valid_scores), f"Expected {len(valid_scores)}, got {valid_count}"
    
    print("✓ Binary search logic is correct")
    
    # Now test the NMS logic
    print(f"\nNMS logic test:")
    print(f"valid_count = {valid_count}")
    print(f"This means we should only process the first {valid_count} boxes")
    print(f"Boxes to process: indices 0 to {valid_count-1}")
    print(f"Expected selected boxes: [0, 1] (scores 0.9, 0.3)")

if __name__ == "__main__":
    test_score_threshold_logic()
