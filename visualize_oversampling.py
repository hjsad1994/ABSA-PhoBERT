"""
Visualize Per-Aspect Oversampling
Creates comparison charts before and after oversampling
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def plot_overall_comparison(df_before, df_after, output_dir):
    """Plot overall sentiment distribution comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    counts_before = df_before['sentiment'].value_counts().sort_index()
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']  # Red, Gray, Green
    
    ax1.bar(counts_before.index, counts_before.values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Before Oversampling', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_ylim(0, max(counts_before.values) * 1.2)
    
    # Add value labels and percentages
    for i, (sentiment, count) in enumerate(counts_before.items()):
        pct = (count / len(df_before)) * 100
        ax1.text(i, count + 100, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    ax1.text(0.5, 0.95, f'Total: {len(df_before):,} samples', 
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # After
    counts_after = df_after['sentiment'].value_counts().sort_index()
    
    ax2.bar(counts_after.index, counts_after.values, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('After Oversampling', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_ylim(0, max(counts_after.values) * 1.2)
    
    # Add value labels and percentages
    for i, (sentiment, count) in enumerate(counts_after.items()):
        pct = (count / len(df_after)) * 100
        ax2.text(i, count + 100, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    increase = len(df_after) - len(df_before)
    increase_pct = (increase / len(df_before)) * 100
    ax2.text(0.5, 0.95, f'Total: {len(df_after):,} samples (+{increase:,}, +{increase_pct:.1f}%)', 
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir}/overall_comparison.png")
    plt.close()


def plot_per_aspect_comparison(df_before, df_after, output_dir):
    """Plot per-aspect sentiment distribution comparison"""
    aspects = sorted(df_before['aspect'].unique())
    
    # Create subplots grid
    n_aspects = len(aspects)
    n_cols = 3
    n_rows = (n_aspects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    sentiments = ['Negative', 'Neutral', 'Positive']
    
    for idx, aspect in enumerate(aspects):
        ax = axes[idx]
        
        # Get data for this aspect
        aspect_before = df_before[df_before['aspect'] == aspect]['sentiment'].value_counts().sort_index()
        aspect_after = df_after[df_after['aspect'] == aspect]['sentiment'].value_counts().sort_index()
        
        # Ensure all sentiments are present (even if count is 0)
        before_counts = [aspect_before.get(s, 0) for s in sentiments]
        after_counts = [aspect_after.get(s, 0) for s in sentiments]
        
        # Bar positions
        x = np.arange(len(sentiments))
        width = 0.35
        
        # Create grouped bars
        bars1 = ax.bar(x - width/2, before_counts, width, label='Before', 
                      color=colors, alpha=0.6, edgecolor='black')
        bars2 = ax.bar(x + width/2, after_counts, width, label='After', 
                      color=colors, alpha=0.9, edgecolor='black')
        
        # Customize
        ax.set_title(f'{aspect}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(sentiments, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Add total count
        total_before = sum(before_counts)
        total_after = sum(after_counts)
        increase = total_after - total_before
        ax.text(0.98, 0.98, f'{total_before}→{total_after}\n(+{increase})',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Hide extra subplots
    for idx in range(n_aspects, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Per-Aspect Sentiment Distribution: Before vs After Oversampling', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_aspect_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir}/per_aspect_comparison.png")
    plt.close()


def plot_stacked_comparison(df_before, df_after, output_dir):
    """Plot stacked bar chart comparison"""
    aspects = sorted(df_before['aspect'].unique())
    sentiments = ['Negative', 'Neutral', 'Positive']
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    
    # Prepare data
    before_data = {}
    after_data = {}
    
    for sentiment in sentiments:
        before_data[sentiment] = []
        after_data[sentiment] = []
        
        for aspect in aspects:
            before_count = len(df_before[(df_before['aspect'] == aspect) & 
                                        (df_before['sentiment'] == sentiment)])
            after_count = len(df_after[(df_after['aspect'] == aspect) & 
                                      (df_after['sentiment'] == sentiment)])
            before_data[sentiment].append(before_count)
            after_data[sentiment].append(after_count)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    x = np.arange(len(aspects))
    width = 0.6
    
    # Before oversampling
    bottom_before = np.zeros(len(aspects))
    for sentiment, color in zip(sentiments, colors):
        ax1.bar(x, before_data[sentiment], width, label=sentiment, 
               color=color, alpha=0.8, edgecolor='black', bottom=bottom_before)
        bottom_before += before_data[sentiment]
    
    ax1.set_title('Before Oversampling - Stacked Distribution per Aspect', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(aspects, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add total labels
    for i, aspect in enumerate(aspects):
        total = sum(before_data[s][i] for s in sentiments)
        ax1.text(i, total + 20, f'{total}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # After oversampling
    bottom_after = np.zeros(len(aspects))
    for sentiment, color in zip(sentiments, colors):
        ax2.bar(x, after_data[sentiment], width, label=sentiment, 
               color=color, alpha=0.8, edgecolor='black', bottom=bottom_after)
        bottom_after += after_data[sentiment]
    
    ax2.set_title('After Oversampling - Stacked Distribution per Aspect', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlabel('Aspect', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(aspects, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add total labels
    for i, aspect in enumerate(aspects):
        total = sum(after_data[s][i] for s in sentiments)
        ax2.text(i, total + 20, f'{total}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stacked_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir}/stacked_comparison.png")
    plt.close()


def plot_balance_heatmap(df_before, df_after, output_dir):
    """Plot heatmap showing balance across aspects"""
    aspects = sorted(df_before['aspect'].unique())
    sentiments = ['Negative', 'Neutral', 'Positive']
    
    # Create matrices for percentages
    before_matrix = []
    after_matrix = []
    
    for aspect in aspects:
        aspect_before = df_before[df_before['aspect'] == aspect]
        aspect_after = df_after[df_after['aspect'] == aspect]
        
        before_row = []
        after_row = []
        
        for sentiment in sentiments:
            before_pct = (len(aspect_before[aspect_before['sentiment'] == sentiment]) / 
                         len(aspect_before) * 100) if len(aspect_before) > 0 else 0
            after_pct = (len(aspect_after[aspect_after['sentiment'] == sentiment]) / 
                        len(aspect_after) * 100) if len(aspect_after) > 0 else 0
            before_row.append(before_pct)
            after_row.append(after_pct)
        
        before_matrix.append(before_row)
        after_matrix.append(after_row)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before heatmap
    im1 = ax1.imshow(before_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(sentiments)))
    ax1.set_yticks(np.arange(len(aspects)))
    ax1.set_xticklabels(sentiments)
    ax1.set_yticklabels(aspects)
    ax1.set_title('Before Oversampling - Percentage Distribution', 
                 fontsize=14, fontweight='bold')
    
    # Add percentage text
    for i in range(len(aspects)):
        for j in range(len(sentiments)):
            text = ax1.text(j, i, f'{before_matrix[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Percentage (%)')
    
    # After heatmap
    im2 = ax2.imshow(after_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(sentiments)))
    ax2.set_yticks(np.arange(len(aspects)))
    ax2.set_xticklabels(sentiments)
    ax2.set_yticklabels(aspects)
    ax2.set_title('After Oversampling - Percentage Distribution', 
                 fontsize=14, fontweight='bold')
    
    # Add percentage text
    for i in range(len(aspects)):
        for j in range(len(sentiments)):
            text = ax2.text(j, i, f'{after_matrix[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=ax2, label='Percentage (%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/balance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir}/balance_heatmap.png")
    plt.close()


def main():
    """Main visualization function"""
    print("=" * 80)
    print("PhoBERT ABSA - Oversampling Visualization")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df_before = pd.read_csv('data/train.csv')
    df_after = pd.read_csv('data/train_oversampled.csv')
    print(f"  Before: {len(df_before):,} samples")
    print(f"  After:  {len(df_after):,} samples")
    print()
    
    # Create output directory
    output_dir = 'visualizations'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    print("1. Overall comparison...")
    plot_overall_comparison(df_before, df_after, output_dir)
    
    print("2. Per-aspect comparison...")
    plot_per_aspect_comparison(df_before, df_after, output_dir)
    
    print("3. Stacked comparison...")
    plot_stacked_comparison(df_before, df_after, output_dir)
    
    print("4. Balance heatmap...")
    plot_balance_heatmap(df_before, df_after, output_dir)
    
    print()
    print("=" * 80)
    print(f"[OK] All visualizations saved to: {output_dir}/")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - overall_comparison.png      (Overall sentiment distribution)")
    print("  - per_aspect_comparison.png   (Detailed per-aspect bars)")
    print("  - stacked_comparison.png      (Stacked bars for all aspects)")
    print("  - balance_heatmap.png         (Percentage heatmap)")


if __name__ == '__main__':
    main()
