import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_revenue_distribution(data):
    revenue_counts = data['Revenue'].value_counts()
    plt.figure(figsize=(8, 6))
    bars = revenue_counts.plot(kind='bar', color=['blue', 'green'])
    plt.title('Bar Plot of Revenue')
    plt.xlabel('Revenue (Purchase Made)')
    plt.ylabel('Number of Sessions')
    plt.xticks(rotation=0)
    for bar in bars.patches:
        plt.annotate(format(bar.get_height(), '.0f'),
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    plt.savefig('results/revenue_distribution.png')
    plt.close()

def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=0.5, linecolor='black')
    plt.title('Correlation Matrix Heatmap of Features')
    plt.savefig('results/correlation_matrix.png')
    plt.close()

def plot_numerical_stats(df):
    num_columns = df.select_dtypes(include=['number']).columns
    for col in num_columns:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=False, color='skyblue', bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 2)
        sns.kdeplot(df[col], color='green')
        plt.title(f'Density Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')

        plt.subplot(1, 3, 3)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Value')

        plt.tight_layout()
        plt.savefig(f'results/{col}_stats.png')
        plt.close()
