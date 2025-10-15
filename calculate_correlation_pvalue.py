#!/usr/bin/env python3
import json
import numpy as np
from scipy import stats

def extract_data_from_files(num_injections=16):
    """Extract avg_log_prob and judgment_sum from rollouts_injection_i_cake.json files."""
    data_points = []
    
    for i in range(num_injections):
        file_path = f"rollouts/rollouts_injection_{i}_cake.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract the required fields
            avg_log_prob = data.get('avg_log_prob', None)
            judgment_sum = data.get('judgment_sum', None)
            
            if avg_log_prob is not None and judgment_sum is not None:
                data_points.append({
                    'injection_id': i,
                    'avg_log_prob': avg_log_prob,
                    'judgment_sum': judgment_sum
                })
            else:
                print(f"Injection {i}: Missing data - avg_log_prob: {avg_log_prob}, judgment_sum: {judgment_sum}")
                
        except FileNotFoundError:
            print(f"File {file_path} not found")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return data_points

def calculate_correlation_test(data_points):
    """Calculate correlation and p-value for the hypothesis that correlation is negative."""
    if len(data_points) < 3:
        print("Not enough data points for correlation test")
        return None, None, None
    
    # Extract data
    avg_log_probs = [point['avg_log_prob'] for point in data_points]
    judgment_sums = [point['judgment_sum'] for point in data_points]
    
    # Calculate Pearson correlation coefficient and p-value
    correlation, p_value = stats.pearsonr(avg_log_probs, judgment_sums)
    
    # For one-tailed test (H1: correlation < 0), we need to divide p-value by 2
    # if the correlation is negative, otherwise p-value = 1 - p_value/2
    if correlation < 0:
        one_tailed_p_value = p_value / 2
    else:
        one_tailed_p_value = 1 - (p_value / 2)
    
    return correlation, p_value, one_tailed_p_value

def main():
    """Main function to calculate correlation statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate correlation p-value for log probability vs judgment sum")
    parser.add_argument("--num-injections", type=int, default=16,
                       help="Number of injection files to process (default: 16)")
    
    args = parser.parse_args()
    
    print(f"Extracting data from rollouts_injection_i_cake.json files (i=0 to {args.num_injections-1})...")
    data_points = extract_data_from_files(args.num_injections)
    
    if len(data_points) < 3:
        print("Not enough data points for correlation test")
        return
    
    print(f"\nFound {len(data_points)} data points")
    
    # Calculate correlation statistics
    correlation, two_tailed_p, one_tailed_p = calculate_correlation_test(data_points)
    
    if correlation is not None:
        print(f"\nCorrelation Analysis:")
        print(f"Pearson correlation coefficient: {correlation:.4f}")
        print(f"Two-tailed p-value: {two_tailed_p:.4f}")
        print(f"One-tailed p-value (H1: correlation < 0): {one_tailed_p:.4f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if one_tailed_p < 0.05:
            print(f"✓ Significant negative correlation (p = {one_tailed_p:.4f} < 0.05)")
        elif one_tailed_p < 0.10:
            print(f"~ Marginally significant negative correlation (p = {one_tailed_p:.4f} < 0.10)")
        else:
            print(f"✗ Not significant negative correlation (p = {one_tailed_p:.4f} >= 0.05)")
        
        # Effect size interpretation
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            effect_size = "strong"
        elif abs_corr >= 0.5:
            effect_size = "moderate"
        elif abs_corr >= 0.3:
            effect_size = "weak"
        else:
            effect_size = "very weak"
        
        print(f"Effect size: {effect_size} ({abs_corr:.3f})")
        
        # Print the data for verification
        print(f"\nData Summary:")
        print(f"{'Injection ID':<12} {'Avg Log Prob':<15} {'Judgment Sum':<12}")
        print("-" * 40)
        for point in sorted(data_points, key=lambda x: x['injection_id']):
            print(f"{point['injection_id']:<12} {point['avg_log_prob']:<15.4f} {point['judgment_sum']:<12}")

if __name__ == "__main__":
    main()
