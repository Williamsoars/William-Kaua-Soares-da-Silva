import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import gcd, lcm
from itertools import combinations
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
import pandas as pd
from sympy import isprime

class ModularGraphDiversityAnalyzer:
    def __init__(self, prime_set):
        """
        Analyzes diversity of modular graphs generated from a prime set
        """
        self.P = sorted(prime_set)
        self.n = len(self.P)
        self.verify_primes()
        
        # Calculate universal exponent A*
        self.A_star = self.compute_universal_exponent()
        self.divisors = self.get_all_divisors()
        self.orders = self.precompute_orders()
        
    def verify_primes(self):
        """Verify all numbers in the set are primes"""
        for p in self.P:
            if not isprime(p):
                raise ValueError(f"{p} is not a prime number!")
    
    def compute_universal_exponent(self):
        """Calculate A* = lcm(p1-1, p2-1, ..., pn-1)"""
        values = [p - 1 for p in self.P]
        result = 1
        for val in values:
            result = lcm(result, val)
        return result
    
    def multiplicative_order(self, a, p):
        """Calculate multiplicative order of a modulo p"""
        if gcd(a, p) != 1:
            return None
        
        order = 1
        power = a % p
        while power != 1:
            power = (power * a) % p
            order += 1
            if order > p:
                return None
        return order
    
    def precompute_orders(self):
        """Precompute all multiplicative orders"""
        orders = {}
        for i, p_i in enumerate(self.P):
            for j, p_j in enumerate(self.P):
                if i != j:
                    order = self.multiplicative_order(p_i, p_j)
                    orders[(i, j)] = order
        return orders
    
    def get_all_divisors(self):
        """Return all divisors of A*"""
        def get_divisors(n):
            divisors = []
            i = 1
            while i * i <= n:
                if n % i == 0:
                    divisors.append(i)
                    if i != n // i:
                        divisors.append(n // i)
                i += 1
            return sorted(divisors)
        return get_divisors(self.A_star)
    
    def build_modular_graph(self, A):
        """Build modular graph G_A(P)"""
        n = self.n
        adj_matrix = np.zeros((n, n), dtype=int)
        
        for (i, j), order in self.orders.items():
            if order is not None and A % order == 0:
                adj_matrix[i, j] = 1
        
        return adj_matrix
    
    def graph_to_vector(self, adj_matrix):
        """Convert adjacency matrix to binary vector"""
        return adj_matrix.flatten()
    
    def calculate_graph_metrics(self, adj_matrix):
        """Calculate structural metrics of the graph"""
        n = self.n
        G = nx.DiGraph(adj_matrix)
        
        # Degrees
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        # Density
        edge_count = np.sum(adj_matrix)
        max_edges = n * (n - 1)
        density = edge_count / max_edges if max_edges > 0 else 0
        
        # Asymmetry
        symmetric_edges = np.sum(adj_matrix * adj_matrix.T)
        asymmetry = 1 - (symmetric_edges / edge_count) if edge_count > 0 else 0
        
        # Directed clustering coefficient
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0
        
        metrics = {
            'edge_count': edge_count,
            'density': density,
            'avg_in_degree': np.mean(in_degrees),
            'avg_out_degree': np.mean(out_degrees),
            'in_degree_std': np.std(in_degrees),
            'out_degree_std': np.std(out_degrees),
            'asymmetry': asymmetry,
            'clustering': clustering,
            'weakly_connected': nx.is_weakly_connected(G),
            'strongly_connected': nx.is_strongly_connected(G)
        }
        
        return metrics
    
    def analyze_diversity(self):
        """Analyze diversity of all modular graphs"""
        print(f"Analyzing diversity for prime set: {self.P}")
        print(f"Universal exponent A* = {self.A_star}")
        print(f"Number of possible exponents (graphs): {len(self.divisors)}")
        
        graphs = []
        vectors = []
        metrics_list = []
        
        # Generate all graphs
        for A in self.divisors:
            adj_matrix = self.build_modular_graph(A)
            graph_vector = self.graph_to_vector(adj_matrix)
            metrics = self.calculate_graph_metrics(adj_matrix)
            metrics['exponent'] = A
            
            graphs.append(adj_matrix)
            vectors.append(graph_vector)
            metrics_list.append(metrics)
        
        # Convert to arrays
        graph_vectors = np.array(vectors)
        metrics_df = pd.DataFrame(metrics_list)
        
        # Diversity analysis
        diversity_results = self.calculate_diversity_metrics(graphs, graph_vectors, metrics_df)
        
        return graphs, metrics_df, diversity_results
    
    def calculate_diversity_metrics(self, graphs, graph_vectors, metrics_df):
        """Calculate diversity metrics"""
        n_graphs = len(graphs)
        
        # 1. Count unique graphs
        unique_graphs = len(set(tuple(v) for v in graph_vectors))
        uniqueness_ratio = unique_graphs / n_graphs
        
        # 2. Density distribution
        density_entropy = entropy(np.histogram(metrics_df['density'], bins=20)[0])
        
        # 3. Structural variability
        structural_diversity = self.calculate_structural_diversity(metrics_df)
        
        # 4. Distance between graphs
        distance_metrics = self.calculate_graph_distances(graph_vectors)
        
        # 5. Search space coverage
        coverage_metrics = self.calculate_search_space_coverage(graph_vectors)
        
        diversity_results = {
            'total_graphs': n_graphs,
            'unique_graphs': unique_graphs,
            'uniqueness_ratio': uniqueness_ratio,
            'density_range': (metrics_df['density'].min(), metrics_df['density'].max()),
            'density_entropy': density_entropy,
            'structural_diversity': structural_diversity,
            'distance_metrics': distance_metrics,
            'coverage_metrics': coverage_metrics,
            'degree_variation': metrics_df[['avg_in_degree', 'avg_out_degree']].std().to_dict(),
            'connectivity_stats': {
                'weakly_connected': metrics_df['weakly_connected'].mean(),
                'strongly_connected': metrics_df['strongly_connected'].mean()
            }
        }
        
        return diversity_results
    
    def calculate_structural_diversity(self, metrics_df):
        """Calculate structural diversity based on multiple metrics"""
        structural_features = ['density', 'asymmetry', 'clustering', 
                              'avg_in_degree', 'avg_out_degree']
        
        feature_matrix = metrics_df[structural_features].values
        feature_std = np.std(feature_matrix, axis=0)
        
        return {
            'feature_std': dict(zip(structural_features, feature_std)),
            'overall_variation': np.mean(feature_std),
            'correlation_diversity': np.linalg.det(np.corrcoef(feature_matrix.T))
        }
    
    def calculate_graph_distances(self, graph_vectors):
        """Calculate distances between graphs"""
        # Hamming distance
        hamming_distances = pairwise_distances(graph_vectors, metric='hamming')
        avg_hamming = np.mean(hamming_distances)
        
        # Euclidean distance
        euclidean_distances = pairwise_distances(graph_vectors, metric='euclidean')
        avg_euclidean = np.mean(euclidean_distances)
        
        return {
            'avg_hamming_distance': avg_hamming,
            'avg_euclidean_distance': avg_euclidean,
            'hamming_std': np.std(hamming_distances),
            'min_distance': np.min(hamming_distances[hamming_distances > 0]),
            'max_distance': np.max(hamming_distances)
        }
    
    def calculate_search_space_coverage(self, graph_vectors):
        """Analyze search space coverage"""
        n_graphs, n_features = graph_vectors.shape
        
        # Percentage of binary space covered
        total_possible_graphs = 2 ** n_features
        coverage_ratio = n_graphs / total_possible_graphs
        
        # Edge patterns diversity
        edge_patterns = len(set(tuple(row) for row in graph_vectors.T))
        
        return {
            'total_possible_graphs': total_possible_graphs,
            'coverage_ratio': coverage_ratio,
            'unique_edge_patterns': edge_patterns,
            'edge_pattern_diversity': edge_patterns / n_features
        }
    
    def generate_diversity_report(self, diversity_results):
        """Generate complete diversity report"""
        print("\n" + "="*70)
        print("DIVERSITY REPORT - MODULAR GRAPHS")
        print("="*70)
        
        dr = diversity_results
        
        print(f"\n1. QUANTITATIVE DIVERSITY:")
        print(f"   • Total graphs generated: {dr['total_graphs']}")
        print(f"   • Unique graphs: {dr['unique_graphs']}")
        print(f"   • Uniqueness ratio: {dr['uniqueness_ratio']:.3f}")
        
        print(f"\n2. STRUCTURAL DIVERSITY:")
        print(f"   • Overall variation: {dr['structural_diversity']['overall_variation']:.4f}")
        print(f"   • Density range: [{dr['density_range'][0]:.3f}, {dr['density_range'][1]:.3f}]")
        print(f"   • Density entropy: {dr['density_entropy']:.3f}")
        
        print(f"\n3. CONNECTIVITY:")
        print(f"   • Weakly connected graphs: {dr['connectivity_stats']['weakly_connected']:.1%}")
        print(f"   • Strongly connected graphs: {dr['connectivity_stats']['strongly_connected']:.1%}")
        
        print(f"\n4. GRAPH DISTANCES:")
        dm = dr['distance_metrics']
        print(f"   • Average Hamming distance: {dm['avg_hamming_distance']:.4f}")
        print(f"   • Minimum distance: {dm['min_distance']:.4f}")
        print(f"   • Maximum distance: {dm['max_distance']:.4f}")
        
        print(f"\n5. SEARCH SPACE COVERAGE:")
        cm = dr['coverage_metrics']
        print(f"   • Total possible space: {cm['total_possible_graphs']}")
        print(f"   • Coverage ratio: {cm['coverage_ratio']:.6f}")
        print(f"   • Unique edge patterns: {cm['unique_edge_patterns']}/{self.n * (self.n-1)}")
        
        # Qualitative assessment
        uniqueness = dr['uniqueness_ratio']
        coverage = cm['coverage_ratio']
        variation = dr['structural_diversity']['overall_variation']
        
        diversity_score = (uniqueness + min(coverage * 100, 1) + variation) / 3
        
        print(f"\n6. FINAL DIVERSITY SCORE: {diversity_score:.3f}/1.000")
        
        if diversity_score > 0.7:
            print("   ✅ HIGH DIVERSITY: Approach generates wide structural variety")
        elif diversity_score > 0.4:
            print("   ⚠️  MODERATE DIVERSITY: Good variation, but space not fully explored")
        else:
            print("   ❌ LOW DIVERSITY: Limitations in generating diverse structures")
    
    def plot_diversity_analysis(self, metrics_df, diversity_results):
        """Generate diversity analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Density distribution
        axes[0, 0].hist(metrics_df['density'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Graph Density')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Density Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. In-degree vs Out-degree
        axes[0, 1].scatter(metrics_df['avg_in_degree'], metrics_df['avg_out_degree'], 
                          alpha=0.6, s=50, color='green')
        axes[0, 1].set_xlabel('Average In-degree')
        axes[0, 1].set_ylabel('Average Out-degree')
        axes[0, 1].set_title('In-degree vs Out-degree Relationship')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Asymmetry vs Clustering
        axes[0, 2].scatter(metrics_df['asymmetry'], metrics_df['clustering'], 
                          alpha=0.6, s=50, color='red')
        axes[0, 2].set_xlabel('Asymmetry')
        axes[0, 2].set_ylabel('Clustering Coefficient')
        axes[0, 2].set_title('Asymmetry vs Clustering')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Evolution with exponent
        axes[1, 0].scatter(metrics_df['exponent'], metrics_df['density'], 
                          alpha=0.6, s=30, color='purple')
        axes[1, 0].set_xlabel('Exponent A')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Density vs Exponent')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Connectivity
        connectivity_data = [metrics_df['weakly_connected'].mean(), 
                           metrics_df['strongly_connected'].mean()]
        axes[1, 1].bar(['Weakly', 'Strongly'], connectivity_data, 
                      color=['orange', 'brown'], alpha=0.7)
        axes[1, 1].set_ylabel('Graph Proportion')
        axes[1, 1].set_title('Graph Connectivity')
        
        # 6. Diversity summary
        diversity_metrics = ['Uniqueness', 'Variation', 'Coverage']
        diversity_values = [
            diversity_results['uniqueness_ratio'],
            diversity_results['structural_diversity']['overall_variation'],
            min(diversity_results['coverage_metrics']['coverage_ratio'] * 100, 1)
        ]
        axes[1, 2].bar(diversity_metrics, diversity_values, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Diversity Metrics')
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Diversity analysis example"""
    
    # Prime set from your paper
    prime_set = [191, 317, 577, 17, 541]
    
    # Analyze diversity
    analyzer = ModularGraphDiversityAnalyzer(prime_set)
    graphs, metrics_df, diversity_results = analyzer.analyze_diversity()
    
    # Generate report
    analyzer.generate_diversity_report(diversity_results)
    
    # Generate visualizations
    analyzer.plot_diversity_analysis(metrics_df, diversity_results)
    
    # Save results
    metrics_df.to_csv('graph_diversity_metrics.csv', index=False)
    
    return analyzer, metrics_df, diversity_results

if __name__ == "__main__":
    analyzer, metrics_df, diversity_results = main()
