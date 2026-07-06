import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def calculate_entropies(observations):
    # Calculate Raw Data Shannon Entropy
    unique_vals = np.arange(1, 11)
    counts = np.zeros_like(unique_vals, dtype=float)
    for i, v in enumerate(unique_vals):
        counts[i] = np.sum(observations == v)
    
    pk_raw = counts / len(observations)
    raw_entropy = entropy(pk_raw)  # in Nats (natural log)
    
    return unique_vals, counts, raw_entropy

def main():
    np.random.seed(42)  # Set seed for reproducible distribution
    
    # ------------------------------------------------------------
    # SCENARIO 1: Uniform distribution of integers between 1 and 10
    # ------------------------------------------------------------
    obs1 = np.random.randint(1, 11, size=100)
    vals1, counts1, raw_ent1 = calculate_entropies(obs1)

    # ------------------------------------------------------------
    # SCENARIO 2: 50 observations = 1 and 50 spread uniformly
    # ------------------------------------------------------------
    obs2 = np.concatenate([np.ones(50, dtype=int), np.random.randint(1, 11, size=50)])
    vals2, counts2, raw_ent2 = calculate_entropies(obs2)

    # ------------------------------------------------------------
    # SCENARIO 3: 100 observations = 3, and all others = 0
    # ------------------------------------------------------------
    obs3 = np.full(100, 3, dtype=int)
    vals3, counts3, raw_ent3 = calculate_entropies(obs3)

    max_theoretical_entropy = np.log(10)

    # Print results to the console
    print("=" * 65)
    print("           ENTROPY VERIFICATION RESULTS")
    print("=" * 65)
    print("SCENARIO 1: Uniformly Distributed Integers (1 to 10)")
    print(f"Observations Count: {len(obs1)}")
    print(f"Unique Values:      {vals1}")
    print(f"Frequencies:        {counts1.astype(int)}")
    print(f"Raw Data Entropy:   {raw_ent1:.6f} Nats")
    print("-" * 65)
    print("SCENARIO 2: 50 Observations = 1, 50 Uniformly Spread")
    print(f"Observations Count: {len(obs2)}")
    print(f"Unique Values:      {vals2}")
    print(f"Frequencies:        {counts2.astype(int)}")
    print(f"Raw Data Entropy:   {raw_ent2:.6f} Nats")
    print("-" * 65)
    print("SCENARIO 3: 100 Observations = 3, All Others = 0")
    print(f"Observations Count: {len(obs3)}")
    print(f"Unique Values:      {vals3}")
    print(f"Frequencies:        {counts3.astype(int)}")
    print(f"Raw Data Entropy:   {raw_ent3:.6f} Nats")
    print("-" * 65)
    print(f"Max Theoretical Entropy:       {max_theoretical_entropy:.6f} Nats")
    print("=" * 65)

    # Plot three histograms side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    sns.set_theme(style="whitegrid")
    
    # Text box properties
    props = dict(boxstyle='round', facecolor='white', edgecolor='#e2e8f0', alpha=0.9, pad=0.8)

    # Scenario 1 Plot (Left)
    sns.histplot(obs1, bins=np.arange(0.5, 11.5, 1), kde=False, color="#4F46E5", edgecolor="white", alpha=0.85, ax=axes[0])
    axes[0].set_title("Scenario 1: Uniform Integers (1 to 10)", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel("Observation Value (Integer)", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Frequency (Counts)", fontsize=11, labelpad=8)
    axes[0].set_xlim(0.5, 10.5)
    axes[0].set_xticks(np.arange(1, 11))
    
    text_content1 = (
        f"Raw Shannon Entropy:   {raw_ent1:.5f} Nats\n"
        f"Max Theoretical:       {max_theoretical_entropy:.5f} Nats"
    )
    axes[0].text(0.35, 0.93, text_content1, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props, fontfamily='monospace')

    # Scenario 2 Plot (Middle)
    sns.histplot(obs2, bins=np.arange(0.5, 11.5, 1), kde=False, color="#E11D48", edgecolor="white", alpha=0.85, ax=axes[1])
    axes[1].set_title("Scenario 2: 50 Obs = 1, 50 Uniform", fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel("Observation Value (Integer)", fontsize=11, labelpad=8)
    axes[1].set_ylabel("Frequency (Counts)", fontsize=11, labelpad=8)
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_xticks(np.arange(1, 11))
    
    text_content2 = (
        f"Raw Shannon Entropy:   {raw_ent2:.5f} Nats\n"
        f"Max Theoretical:       {max_theoretical_entropy:.5f} Nats"
    )
    axes[1].text(0.35, 0.93, text_content2, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props, fontfamily='monospace')

    # Scenario 3 Plot (Right)
    sns.histplot(obs3, bins=np.arange(0.5, 11.5, 1), kde=False, color="#10B981", edgecolor="white", alpha=0.85, ax=axes[2])
    axes[2].set_title("Scenario 3: 100 Obs = 3, All Others = 0", fontsize=12, fontweight='bold', pad=12)
    axes[2].set_xlabel("Observation Value (Integer)", fontsize=11, labelpad=8)
    axes[2].set_ylabel("Frequency (Counts)", fontsize=11, labelpad=8)
    axes[2].set_xlim(0.5, 10.5)
    axes[2].set_xticks(np.arange(1, 11))
    
    text_content3 = (
        f"Raw Shannon Entropy:   {raw_ent3:.5f} Nats\n"
        f"Max Theoretical:       {max_theoretical_entropy:.5f} Nats"
    )
    axes[2].text(0.35, 0.93, text_content3, transform=axes[2].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()
    # Show the plot in the IDE/visual window
    plt.show()

if __name__ == "__main__":
    main()
