#!/usr/bin/env python3
"""
analyze_best_fitness.py - Analyze Taguchi experiment results to find best fitness for each dataset
"""

import pandas as pd
import numpy as np
import sys

def analyze_best_fitness(csv_file='experiment_summary.csv'):
    """
    Analyze the experiment results to find the best fitness for each dataset
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} experiment results")
        print(f"Columns: {df.columns.tolist()}\n")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Group by dataset and find best (minimum) fitness
    best_results = df.groupby('Dataset').agg({
        'Best_Fitness': 'min',
        'Solution_Gen': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Total_Gens': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Total_Time': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Pop_Size': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Mut_Rate': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Tour_Size': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Div_Perc': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Stag_Lim': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Cost_Aware': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Experiment': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()],
        'Run': lambda x: x[df.loc[x.index, 'Best_Fitness'].idxmin()]
    }).round(2)
    
    # Sort by dataset name
    best_results = best_results.sort_index()
    
    # Print results in a formatted table
    print("="*100)
    print("BEST FITNESS RESULTS BY DATASET")
    print("="*100)
    
    for dataset in best_results.index:
        row = best_results.loc[dataset]
        print(f"\nDataset: {dataset}")
        print(f"  Best Fitness: {row['Best_Fitness']}")
        print(f"  Found at Generation: {int(row['Solution_Gen'])} / {int(row['Total_Gens'])}")
        print(f"  Time: {row['Total_Time']:.2f} seconds")
        print(f"  Configuration: Exp {int(row['Experiment'])}, Run {int(row['Run'])}")
        print(f"    - Population Size: {int(row['Pop_Size'])}")
        print(f"    - Mutation Rate: {row['Mut_Rate']}")
        print(f"    - Tournament Size: {int(row['Tour_Size'])}")
        print(f"    - Diversity %: {int(row['Div_Perc'])}")
        print(f"    - Stagnation Limit: {int(row['Stag_Lim'])}")
        print(f"    - Cost Aware: {'Yes' if row['Cost_Aware'] == 1 else 'No'}")
    
    # Create comparison table with paper results
    paper_results = {
        'bayes29': 2937,
        'berlin52': 11087,
        'eil101': 947.4,
        'eil51': 630,
        'eil76': 802,
        'fri26': 1283,
        'gr17': 2760,
        'gr21': 7788,
        'gr24': 1806,
        'gr48': 7288
    }
    
    print("\n" + "="*100)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*100)
    print(f"{'Dataset':<12} {'Your Best':<12} {'Paper Best':<12} {'Gap':<10} {'Gap %':<10}")
    print("-"*60)
    
    for dataset in sorted(best_results.index):
        your_best = best_results.loc[dataset, 'Best_Fitness']
        paper_best = paper_results.get(dataset, 'N/A')
        
        if paper_best != 'N/A':
            gap = your_best - paper_best
            gap_pct = (gap / paper_best) * 100
            print(f"{dataset:<12} {your_best:<12.2f} {paper_best:<12} {gap:<10.2f} {gap_pct:<10.2f}%")
        else:
            print(f"{dataset:<12} {your_best:<12.2f} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
    
    # Statistical summary
    print("\n" + "="*100)
    print("STATISTICAL SUMMARY")
    print("="*100)
    
    # Count experiments by configuration
    config_counts = df.groupby(['Pop_Size', 'Mut_Rate', 'Tour_Size', 'Cost_Aware']).size()
    print(f"\nTotal unique configurations tested: {len(config_counts)}")
    print(f"Total experiment runs: {len(df)}")
    print(f"Datasets tested: {df['Dataset'].nunique()}")
    
    # Best performing configurations (top 5)
    print("\nTop 5 configurations by average fitness:")
    avg_fitness_by_config = df.groupby(['Experiment', 'Pop_Size', 'Mut_Rate', 'Tour_Size', 'Div_Perc', 'Cost_Aware'])['Best_Fitness'].mean()
    top_configs = avg_fitness_by_config.nsmallest(5)
    
    for i, (config, avg_fitness) in enumerate(top_configs.items(), 1):
        exp, pop, mut, tour, div, cost = config
        print(f"{i}. Exp {exp}: Pop={pop}, Mut={mut}, Tour={tour}, Div={div}%, Cost={'Yes' if cost==1 else 'No'} - Avg: {avg_fitness:.2f}")
    
    # Export best results to CSV
    output_file = 'best_fitness_results.csv'
    best_results.to_csv(output_file)
    print(f"\nBest results exported to: {output_file}")
    
    # Create a summary comparison CSV
    comparison_df = pd.DataFrame({
        'Dataset': sorted(best_results.index),
        'Your_Best': [best_results.loc[d, 'Best_Fitness'] for d in sorted(best_results.index)],
        'Paper_Best': [paper_results.get(d, np.nan) for d in sorted(best_results.index)]
    })
    comparison_df['Gap'] = comparison_df['Your_Best'] - comparison_df['Paper_Best']
    comparison_df['Gap_Percent'] = (comparison_df['Gap'] / comparison_df['Paper_Best']) * 100
    
    comparison_file = 'fitness_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Comparison results exported to: {comparison_file}")

if __name__ == "__main__":
    # Check if a different CSV file is provided as argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'experiment_summary.csv'
    
    analyze_best_fitness(csv_file)