#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_data_from_files(deepseek_count=24, llama_count=16):
    """Extract avg_log_prob and judgment_sum from both model datasets."""
    data_points = []
    
    # Extract DeepSeek data (blue)
    print("Extracting DeepSeek data...")
    for i in range(deepseek_count):
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
                    'judgment_sum': judgment_sum,
                    'model': 'DeepSeek-R1-Distill-Qwen-14B'
                })
                print(f"DeepSeek Injection {i}: avg_log_prob = {avg_log_prob:.4f}, judgment_sum = {judgment_sum}")
            else:
                print(f"DeepSeek Injection {i}: Missing data - avg_log_prob: {avg_log_prob}, judgment_sum: {judgment_sum}")
                
        except FileNotFoundError:
            print(f"DeepSeek file {file_path} not found")
        except Exception as e:
            print(f"Error reading DeepSeek file {file_path}: {e}")
    
    # Extract Llama data (red)
    print("Extracting Llama data...")
    for i in range(llama_count):
        file_path = f"rollouts/llama/rollouts_injection_{i}_cake.json"
        
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
                    'judgment_sum': judgment_sum,
                    'model': 'Llama'
                })
                print(f"Llama Injection {i}: avg_log_prob = {avg_log_prob:.4f}, judgment_sum = {judgment_sum}")
            else:
                print(f"Llama Injection {i}: Missing data - avg_log_prob: {avg_log_prob}, judgment_sum: {judgment_sum}")
                
        except FileNotFoundError:
            print(f"Llama file {file_path} not found")
        except Exception as e:
            print(f"Error reading Llama file {file_path}: {e}")
    
    return data_points

def create_plot(data_points):
    """Create a scatter plot of avg_log_prob vs judgment_sum with different colors for each model."""
    if not data_points:
        print("No data points to plot")
        return
    
    # Separate data by model
    deepseek_data = [point for point in data_points if point['model'] == 'DeepSeek-R1-Distill-Qwen-14B']
    llama_data = [point for point in data_points if point['model'] == 'Llama']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot DeepSeek data in blue
    if deepseek_data:
        deepseek_log_probs = [point['avg_log_prob'] for point in deepseek_data]
        deepseek_judgments = [point['judgment_sum'] for point in deepseek_data]
        deepseek_ids = [point['injection_id'] for point in deepseek_data]
        
        plt.scatter(deepseek_log_probs, deepseek_judgments, 
                   c='blue', s=100, alpha=0.7, label=f'DeepSeek-R1-Distill-Qwen-14B (n={len(deepseek_data)})')
        
        # Add annotations for DeepSeek points
        for i, (x, y) in enumerate(zip(deepseek_log_probs, deepseek_judgments)):
            plt.annotate(f'DS{deepseek_ids[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='blue')
    
    # Plot Llama data in red
    if llama_data:
        llama_log_probs = [point['avg_log_prob'] for point in llama_data]
        llama_judgments = [point['judgment_sum'] for point in llama_data]
        llama_ids = [point['injection_id'] for point in llama_data]
        
        plt.scatter(llama_log_probs, llama_judgments, 
                   c='red', s=100, alpha=0.7, label=f'Llama (n={len(llama_data)})')
        
        # Add annotations for Llama points
        for i, (x, y) in enumerate(zip(llama_log_probs, llama_judgments)):
            plt.annotate(f'L{llama_ids[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='red')
    
    # Add labels and title
    plt.xlabel('Average Log Probability')
    plt.ylabel('Judgment Sum')
    plt.title('Average Log Probability vs Judgment Sum\nDeepSeek-R1-Distill-Qwen-14B vs Llama')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlations for each model
    all_log_probs = [point['avg_log_prob'] for point in data_points]
    all_judgments = [point['judgment_sum'] for point in data_points]
    overall_correlation = np.corrcoef(all_log_probs, all_judgments)[0, 1]
    
    # Add correlation info
    correlation_text = f'Overall Correlation: {overall_correlation:.3f}'
    if deepseek_data and llama_data:
        deepseek_corr = np.corrcoef([point['avg_log_prob'] for point in deepseek_data], 
                                   [point['judgment_sum'] for point in deepseek_data])[0, 1]
        llama_corr = np.corrcoef([point['avg_log_prob'] for point in llama_data], 
                                [point['judgment_sum'] for point in llama_data])[0, 1]
        correlation_text += f'\nDeepSeek: {deepseek_corr:.3f}, Llama: {llama_corr:.3f}'
    
    plt.text(0.05, 0.95, correlation_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('log_probs_vs_judgments_dual_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total data points: {len(data_points)}")
    print(f"DeepSeek data points: {len(deepseek_data)}")
    print(f"Llama data points: {len(llama_data)}")
    print(f"Overall correlation coefficient: {overall_correlation:.4f}")
    
    if deepseek_data:
        print(f"DeepSeek log prob range: {min([p['avg_log_prob'] for p in deepseek_data]):.4f} to {max([p['avg_log_prob'] for p in deepseek_data]):.4f}")
        print(f"DeepSeek judgment range: {min([p['judgment_sum'] for p in deepseek_data])} to {max([p['judgment_sum'] for p in deepseek_data])}")
    
    if llama_data:
        print(f"Llama log prob range: {min([p['avg_log_prob'] for p in llama_data]):.4f} to {max([p['avg_log_prob'] for p in llama_data]):.4f}")
        print(f"Llama judgment range: {min([p['judgment_sum'] for p in llama_data])} to {max([p['judgment_sum'] for p in llama_data])}")

def print_data_table(data_points):
    """Print a formatted table of the data."""
    print(f"\n{'Model':<25} {'Injection ID':<12} {'Avg Log Prob':<15} {'Judgment Sum':<12}")
    print("-" * 70)
    
    for point in sorted(data_points, key=lambda x: (x['model'], x['injection_id'])):
        model_short = 'DS' if 'DeepSeek' in point['model'] else 'L'
        print(f"{model_short + str(point['injection_id']):<25} {point['injection_id']:<12} {point['avg_log_prob']:<15.4f} {point['judgment_sum']:<12}")

def main():
    """Main function to extract data and create plot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot average log probability vs judgment sum for dual models")
    parser.add_argument("--deepseek-count", type=int, default=24,
                       help="Number of DeepSeek injection files to process (default: 24)")
    parser.add_argument("--llama-count", type=int, default=16,
                       help="Number of Llama injection files to process (default: 16)")
    
    args = parser.parse_args()
    
    print(f"Extracting data from DeepSeek files (i=0 to {args.deepseek_count-1}) and Llama files (i=0 to {args.llama_count-1})...")
    data_points = extract_data_from_files(args.deepseek_count, args.llama_count)
    
    if data_points:
        print(f"\nFound {len(data_points)} data points")
        print_data_table(data_points)
        create_plot(data_points)
    else:
        print("No valid data points found")

if __name__ == "__main__":
    main()