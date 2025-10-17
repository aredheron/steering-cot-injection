#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

def extract_judgment_data(chocolate_dir="rollouts/deepseek-distill", cheese_dir="rollouts/deepseek-distill-cheese", max_id=18):
    """Extract judgment sums from both chocolate and cheese rollouts."""
    data_points = []
    
    print(f"Extracting judgment data from injection IDs 0 to {max_id}...")
    
    for i in range(max_id + 1):
        chocolate_file = f"{chocolate_dir}/rollouts_injection_{i}_cake.json"
        cheese_file = f"{cheese_dir}/rollouts_injection_{i}_cake.json"
        
        try:
            # Load chocolate data
            with open(chocolate_file, 'r', encoding='utf-8') as f:
                chocolate_data = json.load(f)
            chocolate_judgment = chocolate_data.get('judgment_sum', None)
            
            # Load cheese data
            with open(cheese_file, 'r', encoding='utf-8') as f:
                cheese_data = json.load(f)
            cheese_judgment = cheese_data.get('judgment_sum', None)
            
            if chocolate_judgment is not None and cheese_judgment is not None:
                data_points.append({
                    'injection_id': i,
                    'chocolate_judgment': chocolate_judgment,
                    'cheese_judgment': cheese_judgment
                })
                print(f"Injection {i}: Chocolate={chocolate_judgment}, Cheese={cheese_judgment}")
            else:
                print(f"Injection {i}: Missing data - Chocolate: {chocolate_judgment}, Cheese: {cheese_judgment}")
                
        except FileNotFoundError as e:
            print(f"File not found for injection {i}: {e}")
        except Exception as e:
            print(f"Error reading injection {i}: {e}")
    
    return data_points

def create_plot(data_points):
    """Create a scatter plot of chocolate vs cheese judgments."""
    if not data_points:
        print("No data points to plot")
        return
    
    # Extract coordinates
    chocolate_judgments = [point['chocolate_judgment'] for point in data_points]
    cheese_judgments = [point['cheese_judgment'] for point in data_points]
    injection_ids = [point['injection_id'] for point in data_points]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the points
    plt.scatter(chocolate_judgments, cheese_judgments, 
               c='purple', s=100, alpha=0.7, label=f'Injection Points (n={len(data_points)})')
    
    # Add annotations for each point
    for i, (x, y) in enumerate(zip(chocolate_judgments, cheese_judgments)):
        plt.annotate(f'{injection_ids[i]}', (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color='purple')
    
    # Add labels and title
    plt.xlabel('Chocolate Cake Judgment Sum')
    plt.ylabel('Cheesecake Judgment Sum')
    plt.title('Chocolate Cake vs Cheesecake Judgment Sums\nDeepSeek-Distill Model')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    correlation, p_value = stats.pearsonr(chocolate_judgments, cheese_judgments)
    
    # Add correlation info
    correlation_text = f'Correlation: {correlation:.3f} (p={p_value:.3f})'
    plt.text(0.05, 0.95, correlation_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add diagonal line for reference (perfect correlation)
    min_val = min(min(chocolate_judgments), min(cheese_judgments))
    max_val = max(max(chocolate_judgments), max(cheese_judgments))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('chocolate_vs_cheese_judgments.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total data points: {len(data_points)}")
    print(f"Correlation coefficient: {correlation:.4f} (p-value: {p_value:.4f})")
    print(f"Chocolate judgment range: {min(chocolate_judgments)} to {max(chocolate_judgments)}")
    print(f"Cheese judgment range: {min(cheese_judgments)} to {max(cheese_judgments)}")
    
    # Calculate some additional statistics
    chocolate_mean = np.mean(chocolate_judgments)
    cheese_mean = np.mean(cheese_judgments)
    chocolate_std = np.std(chocolate_judgments)
    cheese_std = np.std(cheese_judgments)
    
    print(f"Chocolate mean ± std: {chocolate_mean:.2f} ± {chocolate_std:.2f}")
    print(f"Cheese mean ± std: {cheese_mean:.2f} ± {cheese_std:.2f}")

def print_data_table(data_points):
    """Print a formatted table of the data."""
    print(f"\n{'Injection ID':<12} {'Chocolate':<10} {'Cheese':<10} {'Difference':<12}")
    print("-" * 50)
    
    for point in sorted(data_points, key=lambda x: x['injection_id']):
        diff = point['chocolate_judgment'] - point['cheese_judgment']
        print(f"{point['injection_id']:<12} {point['chocolate_judgment']:<10} {point['cheese_judgment']:<10} {diff:<12}")

def main():
    """Main function to extract data and create plot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot chocolate vs cheese judgment sums")
    parser.add_argument("--chocolate-dir", type=str, default="rollouts/deepseek-distill",
                       help="Directory containing chocolate cake rollouts")
    parser.add_argument("--cheese-dir", type=str, default="rollouts/deepseek-distill-cheese",
                       help="Directory containing cheesecake rollouts")
    parser.add_argument("--max-id", type=int, default=18,
                       help="Maximum injection ID to process (default: 18)")
    
    args = parser.parse_args()
    
    print(f"Extracting judgment data from {args.chocolate_dir} and {args.cheese_dir}...")
    data_points = extract_judgment_data(args.chocolate_dir, args.cheese_dir, args.max_id)
    
    if data_points:
        print(f"\nFound {len(data_points)} data points")
        print_data_table(data_points)
        create_plot(data_points)
    else:
        print("No valid data points found")

if __name__ == "__main__":
    main()
