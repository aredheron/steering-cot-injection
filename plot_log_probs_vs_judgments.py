#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

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
            avg_log_prob_top_p = data.get('avg_log_prob_top_p', None)
            judgment_sum = data.get('judgment_sum', None)
            log_probs = data.get('log_probs', [])
            sentence_length = len(log_probs) if log_probs else None
            
            if avg_log_prob is not None and avg_log_prob_top_p is not None and judgment_sum is not None and sentence_length is not None:
                data_points.append({
                    'injection_id': i,
                    'avg_log_prob': avg_log_prob,
                    'avg_log_prob_top_p': avg_log_prob_top_p,
                    'judgment_sum': judgment_sum,
                    'sentence_length': sentence_length,
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
            avg_log_prob_top_p = data.get('avg_log_prob_top_p', None)
            judgment_sum = data.get('judgment_sum', None)
            log_probs = data.get('log_probs', [])
            sentence_length = len(log_probs) if log_probs else None
            
            if avg_log_prob is not None and avg_log_prob_top_p is not None and judgment_sum is not None and sentence_length is not None:
                data_points.append({
                    'injection_id': i,
                    'avg_log_prob': avg_log_prob,
                    'avg_log_prob_top_p': avg_log_prob_top_p,
                    'judgment_sum': judgment_sum,
                    'sentence_length': sentence_length,
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

def calculate_correlation_with_pvalue(x, y):
    """Calculate Pearson correlation coefficient and p-value."""
    correlation, p_value = stats.pearsonr(x, y)
    return correlation, p_value

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
    overall_correlation, overall_p_value = calculate_correlation_with_pvalue(all_log_probs, all_judgments)
    
    # Add correlation info
    correlation_text = f'Overall Correlation: {overall_correlation:.3f} (p={overall_p_value:.3f})'
    if deepseek_data and llama_data:
        deepseek_corr, deepseek_p = calculate_correlation_with_pvalue([point['avg_log_prob'] for point in deepseek_data], 
                                                                     [point['judgment_sum'] for point in deepseek_data])
        llama_corr, llama_p = calculate_correlation_with_pvalue([point['avg_log_prob'] for point in llama_data], 
                                                               [point['judgment_sum'] for point in llama_data])
        correlation_text += f'\nDeepSeek: {deepseek_corr:.3f} (p={deepseek_p:.3f})'
        correlation_text += f'\nLlama: {llama_corr:.3f} (p={llama_p:.3f})'
    
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
    print(f"Overall correlation coefficient: {overall_correlation:.4f} (p-value: {overall_p_value:.4f})")
    
    if deepseek_data:
        print(f"DeepSeek correlation: {deepseek_corr:.4f} (p-value: {deepseek_p:.4f})")
        print(f"DeepSeek log prob range: {min([p['avg_log_prob'] for p in deepseek_data]):.4f} to {max([p['avg_log_prob'] for p in deepseek_data]):.4f}")
        print(f"DeepSeek judgment range: {min([p['judgment_sum'] for p in deepseek_data])} to {max([p['judgment_sum'] for p in deepseek_data])}")
    
    if llama_data:
        print(f"Llama correlation: {llama_corr:.4f} (p-value: {llama_p:.4f})")
        print(f"Llama log prob range: {min([p['avg_log_prob'] for p in llama_data]):.4f} to {max([p['avg_log_prob'] for p in llama_data]):.4f}")
        print(f"Llama judgment range: {min([p['judgment_sum'] for p in llama_data])} to {max([p['judgment_sum'] for p in llama_data])}")

def create_sentence_length_plot(data_points):
    """Create a scatter plot of sentence_length vs judgment_sum with different colors for each model."""
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
        deepseek_lengths = [point['sentence_length'] for point in deepseek_data]
        deepseek_judgments = [point['judgment_sum'] for point in deepseek_data]
        deepseek_ids = [point['injection_id'] for point in deepseek_data]
        
        plt.scatter(deepseek_lengths, deepseek_judgments, 
                   c='blue', s=100, alpha=0.7, label=f'DeepSeek-R1-Distill-Qwen-14B (n={len(deepseek_data)})')
        
        # Add annotations for DeepSeek points
        for i, (x, y) in enumerate(zip(deepseek_lengths, deepseek_judgments)):
            plt.annotate(f'DS{deepseek_ids[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='blue')
    
    # Plot Llama data in red
    if llama_data:
        llama_lengths = [point['sentence_length'] for point in llama_data]
        llama_judgments = [point['judgment_sum'] for point in llama_data]
        llama_ids = [point['injection_id'] for point in llama_data]
        
        plt.scatter(llama_lengths, llama_judgments, 
                   c='red', s=100, alpha=0.7, label=f'Llama (n={len(llama_data)})')
        
        # Add annotations for Llama points
        for i, (x, y) in enumerate(zip(llama_lengths, llama_judgments)):
            plt.annotate(f'L{llama_ids[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='red')
    
    # Add labels and title
    plt.xlabel('Sentence Length (Number of Tokens)')
    plt.ylabel('Judgment Sum')
    plt.title('Sentence Length vs Judgment Sum\nDeepSeek-R1-Distill-Qwen-14B vs Llama')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlations for each model
    all_lengths = [point['sentence_length'] for point in data_points]
    all_judgments = [point['judgment_sum'] for point in data_points]
    overall_correlation, overall_p_value = calculate_correlation_with_pvalue(all_lengths, all_judgments)
    
    # Add correlation info
    correlation_text = f'Overall Correlation: {overall_correlation:.3f} (p={overall_p_value:.3f})'
    if deepseek_data and llama_data:
        deepseek_corr, deepseek_p = calculate_correlation_with_pvalue([point['sentence_length'] for point in deepseek_data], 
                                                                     [point['judgment_sum'] for point in deepseek_data])
        llama_corr, llama_p = calculate_correlation_with_pvalue([point['sentence_length'] for point in llama_data], 
                                                               [point['judgment_sum'] for point in llama_data])
        correlation_text += f'\nDeepSeek: {deepseek_corr:.3f} (p={deepseek_p:.3f})'
        correlation_text += f'\nLlama: {llama_corr:.3f} (p={llama_p:.3f})'
    
    plt.text(0.05, 0.95, correlation_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('sentence_length_vs_judgments_dual_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nSentence Length vs Judgment Sum Summary Statistics:")
    print(f"Total data points: {len(data_points)}")
    print(f"DeepSeek data points: {len(deepseek_data)}")
    print(f"Llama data points: {len(llama_data)}")
    print(f"Overall correlation coefficient: {overall_correlation:.4f} (p-value: {overall_p_value:.4f})")
    
    if deepseek_data:
        print(f"DeepSeek sentence length range: {min([p['sentence_length'] for p in deepseek_data])} to {max([p['sentence_length'] for p in deepseek_data])}")
        print(f"DeepSeek judgment range: {min([p['judgment_sum'] for p in deepseek_data])} to {max([p['judgment_sum'] for p in deepseek_data])}")
        print(f"DeepSeek correlation: {deepseek_corr:.4f} (p-value: {deepseek_p:.4f})")
    
    if llama_data:
        print(f"Llama sentence length range: {min([p['sentence_length'] for p in llama_data])} to {max([p['sentence_length'] for p in llama_data])}")
        print(f"Llama judgment range: {min([p['judgment_sum'] for p in llama_data])} to {max([p['judgment_sum'] for p in llama_data])}")
        print(f"Llama correlation: {llama_corr:.4f} (p-value: {llama_p:.4f})")

def create_top_p_plot(data_points):
    """Create a scatter plot of avg_log_prob_top_p vs judgment_sum with different colors for each model."""
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
        deepseek_log_probs = [point['avg_log_prob_top_p'] for point in deepseek_data]
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
        llama_log_probs = [point['avg_log_prob_top_p'] for point in llama_data]
        llama_judgments = [point['judgment_sum'] for point in llama_data]
        llama_ids = [point['injection_id'] for point in llama_data]
        
        plt.scatter(llama_log_probs, llama_judgments, 
                   c='red', s=100, alpha=0.7, label=f'Llama (n={len(llama_data)})')
        
        # Add annotations for Llama points
        for i, (x, y) in enumerate(zip(llama_log_probs, llama_judgments)):
            plt.annotate(f'L{llama_ids[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='red')
    
    # Add labels and title
    plt.xlabel('Average Log Probability (Top-p)')
    plt.ylabel('Judgment Sum')
    plt.title('Average Log Probability (Top-p) vs Judgment Sum\nDeepSeek-R1-Distill-Qwen-14B vs Llama')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlations for each model
    all_log_probs = [point['avg_log_prob_top_p'] for point in data_points]
    all_judgments = [point['judgment_sum'] for point in data_points]
    overall_correlation, overall_p_value = calculate_correlation_with_pvalue(all_log_probs, all_judgments)
    
    # Add correlation info
    correlation_text = f'Overall Correlation: {overall_correlation:.3f} (p={overall_p_value:.3f})'
    if deepseek_data and llama_data:
        deepseek_corr, deepseek_p = calculate_correlation_with_pvalue([point['avg_log_prob_top_p'] for point in deepseek_data], 
                                                                     [point['judgment_sum'] for point in deepseek_data])
        llama_corr, llama_p = calculate_correlation_with_pvalue([point['avg_log_prob_top_p'] for point in llama_data], 
                                                               [point['judgment_sum'] for point in llama_data])
        correlation_text += f'\nDeepSeek: {deepseek_corr:.3f} (p={deepseek_p:.3f})'
        correlation_text += f'\nLlama: {llama_corr:.3f} (p={llama_p:.3f})'
    
    plt.text(0.05, 0.95, correlation_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('log_probs_top_p_vs_judgments_dual_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nTop-p Log Probability vs Judgment Sum Summary Statistics:")
    print(f"Total data points: {len(data_points)}")
    print(f"DeepSeek data points: {len(deepseek_data)}")
    print(f"Llama data points: {len(llama_data)}")
    print(f"Overall correlation coefficient: {overall_correlation:.4f} (p-value: {overall_p_value:.4f})")
    
    if deepseek_data:
        print(f"DeepSeek correlation: {deepseek_corr:.4f} (p-value: {deepseek_p:.4f})")
        print(f"DeepSeek top-p log prob range: {min([p['avg_log_prob_top_p'] for p in deepseek_data]):.4f} to {max([p['avg_log_prob_top_p'] for p in deepseek_data]):.4f}")
        print(f"DeepSeek judgment range: {min([p['judgment_sum'] for p in deepseek_data])} to {max([p['judgment_sum'] for p in deepseek_data])}")
    
    if llama_data:
        print(f"Llama correlation: {llama_corr:.4f} (p-value: {llama_p:.4f})")
        print(f"Llama top-p log prob range: {min([p['avg_log_prob_top_p'] for p in llama_data]):.4f} to {max([p['avg_log_prob_top_p'] for p in llama_data]):.4f}")
        print(f"Llama judgment range: {min([p['judgment_sum'] for p in llama_data])} to {max([p['judgment_sum'] for p in llama_data])}")

def print_data_table(data_points):
    """Print a formatted table of the data."""
    print(f"\n{'Model':<25} {'Injection ID':<12} {'Avg Log Prob':<15} {'Avg Log Prob Top-p':<20} {'Judgment Sum':<12} {'Sentence Length':<15}")
    print("-" * 100)
    
    for point in sorted(data_points, key=lambda x: (x['model'], x['injection_id'])):
        model_short = 'DS' if 'DeepSeek' in point['model'] else 'L'
        print(f"{model_short + str(point['injection_id']):<25} {point['injection_id']:<12} {point['avg_log_prob']:<15.4f} {point['avg_log_prob_top_p']:<20.4f} {point['judgment_sum']:<12} {point['sentence_length']:<15}")

def main():
    """Main function to extract data and create plot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot average log probability vs judgment sum for dual models")
    parser.add_argument("--deepseek-count", type=int, default=85,
                       help="Number of DeepSeek injection files to process (default: 24)")
    parser.add_argument("--llama-count", type=int, default=64,
                       help="Number of Llama injection files to process (default: 16)")
    
    args = parser.parse_args()
    
    print(f"Extracting data from DeepSeek files (i=0 to {args.deepseek_count-1}) and Llama files (i=0 to {args.llama_count-1})...")
    data_points = extract_data_from_files(args.deepseek_count, args.llama_count)
    
    if data_points:
        print(f"\nFound {len(data_points)} data points")
        print_data_table(data_points)
        create_plot(data_points)
        create_sentence_length_plot(data_points)
        create_top_p_plot(data_points)
    else:
        print("No valid data points found")

if __name__ == "__main__":
    main()