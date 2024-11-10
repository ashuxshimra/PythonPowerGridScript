import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import pywt
import warnings
import os
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_style("darkgrid")
plt.style.use('default')

class PowerQualityAnalyzer:
    def __init__(self, sampling_rate=10000, signal_duration=1.0):
        self.sampling_rate = sampling_rate
        self.signal_duration = signal_duration
        self.t = np.linspace(0, signal_duration, int(sampling_rate * signal_duration))
        self.frequency = 50  # Base frequency (50 Hz)
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic power quality signals"""
        signals = []
        labels = []
        
        for _ in range(n_samples):
            signal_type = np.random.randint(0, 3)
            
            # Base signal
            signal_data = np.sin(2 * np.pi * self.frequency * self.t)
            
            if signal_type == 0:  # High quality
                noise = np.random.normal(0, 0.01, len(self.t))
                signal_data = signal_data + noise
                
            elif signal_type == 1:  # Medium quality (harmonics)
                third_harmonic = 0.15 * np.sin(2 * np.pi * 3 * self.frequency * self.t)
                fifth_harmonic = 0.1 * np.sin(2 * np.pi * 5 * self.frequency * self.t)
                signal_data = signal_data + third_harmonic + fifth_harmonic
                
            else:  # Low quality (dips and transients)
                # Add voltage dip
                dip_start = np.random.randint(1000, 8000)
                dip_duration = np.random.randint(500, 1500)
                signal_data[dip_start:dip_start+dip_duration] *= 0.7
                
                # Add transient
                transient_start = np.random.randint(1000, 8000)
                transient_duration = 50
                signal_data[transient_start:transient_start+transient_duration] *= 1.5
            
            signals.append(signal_data)
            labels.append(signal_type)
            
        return np.array(signals), np.array(labels)
    
    def extract_features(self, signals):
        """Extract comprehensive feature set from signals"""
        features = []
        
        for signal_data in signals:
            # Time domain features
            features_dict = {
                'mean': np.mean(signal_data),
                'std': np.std(signal_data),
                'rms': np.sqrt(np.mean(signal_data**2)),
                'peak': np.max(np.abs(signal_data)),
                'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
                'kurtosis': stats.kurtosis(signal_data),
                'skewness': stats.skew(signal_data)
            }
            
            # Frequency domain features
            freqs, psd = signal.welch(signal_data, self.sampling_rate)
            features_dict.update({
                'dominant_freq': freqs[np.argmax(psd)],
                'spectral_entropy': -np.sum(psd * np.log2(psd + 1e-10)),
                'spectral_rolloff': np.percentile(psd, 85)
            })
            
            # Wavelet features
            coeffs = pywt.wavedec(signal_data, 'db4', level=4)
            for i, coeff in enumerate(coeffs):
                features_dict[f'wavelet_energy_{i}'] = np.sum(coeff**2)
            
            features.append(list(features_dict.values()))
            
        return np.array(features)
    
    def build_advanced_model(self):
        """Build an advanced stacking ensemble model"""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=200, random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=200, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42))
        ]
        
        meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
    
    def calculate_tariffs(self, predictions, base_rate=0.15):
        """Calculate tariffs based on power quality predictions"""
        tariffs = []
        for pred in predictions:
            if pred == 0:  # High quality
                tariff = base_rate * 1.0
            elif pred == 1:  # Medium quality
                tariff = base_rate * 0.8
            else:  # Low quality
                tariff = base_rate * 0.6
            tariffs.append(tariff)
        return np.array(tariffs)

class PowerQualityVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.qualities = ['High Quality', 'Medium Quality', 'Low Quality']
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        
    def plot_signal_examples(self, signals, labels):
        """Plot example signals for each quality category with detailed analysis"""
        fig = plt.figure(figsize=(15, 12))
        
        for i, quality in enumerate(self.qualities):
            idx = np.where(labels == i)[0][0]
            signal_data = signals[idx]
            
            # Time domain plot
            plt.subplot(3, 3, i*3 + 1)
            plt.plot(self.analyzer.t[:500], signal_data[:500], 'b-', linewidth=2)
            plt.title(f'{quality} - Time Domain')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Frequency domain plot
            plt.subplot(3, 3, i*3 + 2)
            freqs = fftfreq(len(signal_data), 1/self.analyzer.sampling_rate)
            fft_vals = np.abs(fft(signal_data))
            plt.plot(freqs[:len(freqs)//2], fft_vals[:len(freqs)//2], 'r-')
            plt.title(f'{quality} - Frequency Domain')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            
            # Wavelet analysis
            plt.subplot(3, 3, i*3 + 3)
            coeffs, freqs = pywt.cwt(signal_data, np.arange(1, 128), 'morl')
            plt.imshow(abs(coeffs), extent=[0, 1, 1, 128], aspect='auto', cmap='jet')
            plt.title(f'{quality} - Wavelet Transform')
            plt.xlabel('Time')
            plt.ylabel('Scale')
            plt.colorbar(label='Magnitude')
        
        plt.tight_layout()
        return fig

    def plot_performance_metrics(self, model, X_train, X_test, y_train, y_test, predictions):
        """Generate comprehensive performance metrics visualization"""
        fig = plt.figure(figsize=(20, 15))
        
        # Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.qualities,
                   yticklabels=self.qualities)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # ROC Curves
        plt.subplot(2, 3, 2)
        for i, quality in enumerate(self.qualities):
            proba = model.predict_proba(X_test)[:, i]
            fpr, tpr, _ = roc_curve(y_test == i, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{quality} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Learning Curves
        plt.subplot(2, 3, 3)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        
        # Tariff Analysis
        plt.subplot(2, 3, 4)
        tariffs = self.analyzer.calculate_tariffs(predictions)
        sns.boxplot(x=predictions, y=tariffs)
        plt.xticks(range(3), self.qualities)
        plt.title('Tariff Distribution by Quality Class')
        plt.xlabel('Power Quality Category')
        plt.ylabel('Tariff Rate ($/kWh)')
        
        # Feature Importance
        plt.subplot(2, 3, 5)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            plt.bar(range(10), importances[indices])
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance Score')
        
        plt.tight_layout()
        return fig

    def plot_quality_metrics(self, signals, labels):
        """Plot detailed quality metrics for each category"""
        fig = plt.figure(figsize=(15, 10))
        
        metrics = {
            'THD': [],
            'Crest Factor': [],
            'RMS': [],
            'Quality': []
        }
        
        for signal_data, label in zip(signals, labels):
            # Calculate Total Harmonic Distortion
            freqs = fftfreq(len(signal_data), 1/self.analyzer.sampling_rate)
            fft_vals = np.abs(fft(signal_data))
            fundamental_idx = np.argmax(fft_vals[:len(freqs)//2])
            harmonics = fft_vals[fundamental_idx*2:fundamental_idx*10:fundamental_idx]
            thd = np.sqrt(np.sum(harmonics**2)) / fft_vals[fundamental_idx] * 100
            
            metrics['THD'].append(thd)
            metrics['Crest Factor'].append(np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)))
            metrics['RMS'].append(np.sqrt(np.mean(signal_data**2)))
            metrics['Quality'].append(self.qualities[label])
        
        # Convert to DataFrame for easier plotting
        df_metrics = pd.DataFrame(metrics)
        
        # Plot distributions
        plt.subplot(2, 2, 1)
        sns.boxplot(x='Quality', y='THD', data=df_metrics)
        plt.title('Total Harmonic Distortion by Quality')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Quality', y='Crest Factor', data=df_metrics)
        plt.title('Crest Factor by Quality')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x='Quality', y='RMS', data=df_metrics)
        plt.title('RMS Values by Quality')
        plt.xticks(rotation=45)
        
        # 3D scatter plot
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        for i, quality in enumerate(self.qualities):
            mask = df_metrics['Quality'] == quality
            ax.scatter(df_metrics[mask]['THD'], 
                      df_metrics[mask]['Crest Factor'],
                      df_metrics[mask]['RMS'],
                      label=quality)
        ax.set_xlabel('THD')
        ax.set_ylabel('Crest Factor')
        ax.set_zlabel('RMS')
        ax.legend()
        plt.title('3D Quality Metrics Distribution')
        
        plt.tight_layout()
        return fig

def create_output_directory():
    """Create directory for output files"""
    output_dir = 'power_quality_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    # Create output directory
    output_dir = create_output_directory()
    
    # Initialize analyzer and visualizer
    print("Initializing analyzer and visualizer...")
    pqa = PowerQualityAnalyzer()
    visualizer = PowerQualityVisualizer(pqa)
    
    # Generate and process data
    print("Generating synthetic data...")
    signals, labels = pqa.generate_synthetic_data(n_samples=5000)
    
    print("Extracting features...")
    features = pqa.extract_features(signals)
    
    # Split and scale data
    print("Preparing training and test datasets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training the model (this may take a few minutes)...")
    model = pqa.build_advanced_model()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    # Generate and save all visualizations
    print("\nGenerating visualizations...")
    
    # 1. Signal Examples
    print("- Generating signal examples plot...")
    fig1 = visualizer.plot_signal_examples(signals, labels)
    fig1.savefig(os.path.join(output_dir, 'signal_examples.png'), 
                 dpi=300, bbox_inches='tight')
    
    # 2. Performance Metrics
    print("- Generating performance metrics plot...")
    fig2 = visualizer.plot_performance_metrics(
        model, X_train_scaled, X_test_scaled, y_train, y_test, predictions
    )
    fig2.savefig(os.path.join(output_dir, 'performance_metrics.png'), 
                 dpi=300, bbox_inches='tight')
    
    # 3. Quality Metrics
    print("- Generating quality metrics plot...")
    fig3 = visualizer.plot_quality_metrics(signals, labels)
    fig3.savefig(os.path.join(output_dir, 'quality_metrics.png'), 
                 dpi=300, bbox_inches='tight')
    
    # Save classification report
    print("\nGenerating classification report...")
    report = classification_report(y_test, predictions, 
                                 target_names=visualizer.qualities,
                                 output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Print results location
    print("\nResults have been saved to the '{}' directory:".format(output_dir))
    print("1. signal_examples.png - Signal analysis visualizations")
    print("2. performance_metrics.png - Model performance metrics")
    print("3. quality_metrics.png - Power quality metrics analysis")
    print("4. classification_report.csv - Detailed classification metrics")
    
    # Display plots (optional)
    plt.show()

if __name__ == "__main__":
    main()