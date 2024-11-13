import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import ttest_ind
import pywt
import warnings
import os

# Configure plotting and warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.style.use('default')

class AdvancedPowerQualityAnalyzer:
    def __init__(self, sampling_rate=10000, signal_duration=1.0):
        """Initialize the analyzer with given sampling rate and signal duration"""
        self.sampling_rate = sampling_rate
        self.signal_duration = signal_duration
        self.t = np.linspace(0, signal_duration, int(sampling_rate * signal_duration))
        self.frequency = 50  # Base frequency (50 Hz)
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic power quality signals with different characteristics"""
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

    def calculate_snr(self, signal):
        """Calculate Signal-to-Noise Ratio in dB"""
        noise = signal - np.mean(signal)
        return 10 * np.log10(np.sum(signal**2) / np.sum(noise**2))
    
    def calculate_power_quality_indices(self, signal):
        """Calculate comprehensive power quality indices"""
        # FFT analysis
        freqs = fftfreq(len(signal), 1/self.sampling_rate)
        fft_vals = np.abs(fft(signal))
        
        # Find fundamental and harmonics
        fundamental_idx = np.argmax(fft_vals[:len(freqs)//2])
        fundamental = fft_vals[fundamental_idx]
        harmonics = fft_vals[fundamental_idx*2:fundamental_idx*10:fundamental_idx]
        
        # Calculate indices
        thd = np.sqrt(np.sum(harmonics**2)) / fundamental * 100
        cf = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
        ff = np.sum(np.abs(np.diff(signal))) / (len(signal) * np.sqrt(np.mean(signal**2)))
        
        return {
            'THD': thd,
            'Crest_Factor': cf,
            'Form_Factor': ff,
            'SNR': self.calculate_snr(signal)
        }

    def extract_enhanced_features(self, signals):
        """Extract comprehensive feature set from signals with enhanced metrics"""
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
            
            # Additional enhanced metrics
            quality_indices = self.calculate_power_quality_indices(signal_data)
            features_dict.update({
                'snr': quality_indices['SNR'],
                'form_factor': quality_indices['Form_Factor'],
                'enhanced_thd': quality_indices['THD']
            })
            
            features.append(list(features_dict.values()))
            
        return np.array(features)
    def build_deep_model(self):
        """Build a deep learning model using scikit-learn's MLPClassifier with PCA"""
        return Pipeline([
            ('pca', PCA(n_components=0.95)),  # Preserve 95% of variance
            ('deep_net', MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=42,
                verbose=True
            ))
        ])
    
    def build_advanced_model(self):
        """Build an advanced stacking ensemble model"""
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            ))
        ]
        
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
    
    def perform_comparative_analysis(self, X_train, X_test, y_train, y_test):
        """Perform comprehensive comparative analysis with traditional methods"""
        # Traditional method (Single Random Forest)
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=-1)
        
        # Advanced method (Our ensemble)
        ensemble_model = self.build_advanced_model()
        ensemble_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, n_jobs=-1)
        
        # Statistical significance test
        t_stat, p_value = ttest_ind(rf_scores, ensemble_scores)
        
        return {
            'traditional_scores': rf_scores,
            'ensemble_scores': ensemble_scores,
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_traditional': np.mean(rf_scores),
            'mean_ensemble': np.mean(ensemble_scores),
            'std_traditional': np.std(rf_scores),
            'std_ensemble': np.std(ensemble_scores)
        }
    
    def calculate_tariffs(self, predictions, base_rate=0.15):
        """Calculate tariffs based on power quality predictions with uncertainty"""
        tariffs = []
        uncertainties = []
        
        for pred in predictions:
            if pred == 0:  # High quality
                tariff = base_rate * 1.0
                uncertainty = 0.02  # 2% uncertainty for high quality
            elif pred == 1:  # Medium quality
                tariff = base_rate * 0.8
                uncertainty = 0.05  # 5% uncertainty for medium quality
            else:  # Low quality
                tariff = base_rate * 0.6
                uncertainty = 0.08  # 8% uncertainty for low quality
            
            # Add random variation within uncertainty bounds
            tariff_with_uncertainty = tariff * (1 + np.random.uniform(-uncertainty, uncertainty))
            tariffs.append(tariff_with_uncertainty)
            uncertainties.append(uncertainty)
            
        return np.array(tariffs), np.array(uncertainties)

class EnhancedPowerQualityVisualizer:
    def __init__(self, analyzer):
        """Initialize the visualizer with analyzer instance"""
        self.analyzer = analyzer
        self.qualities = ['High Quality', 'Medium Quality', 'Low Quality']
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['figure.dpi'] = 100
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
        tariffs, uncertainties = self.analyzer.calculate_tariffs(predictions)
        sns.boxplot(x=predictions, y=tariffs)
        plt.xticks(range(3), self.qualities)
        plt.title('Tariff Distribution by Quality Class')
        plt.xlabel('Power Quality Category')
        plt.ylabel('Tariff Rate ($/kWh)')
        
        # Feature Importance
        plt.subplot(2, 3, 5)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(10), importances[indices])
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance Score')
        
        plt.tight_layout()
        return fig
    
    def plot_quality_metrics(self, signals, labels):
        """Plot detailed quality metrics for each category"""
        # Create two separate figures
        fig_2d = plt.figure(figsize=(15, 10))
        
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
        
        # Plot 2D distributions
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
        
        plt.tight_layout()
        
        # Create separate figure for 3D plot
        fig_3d = plt.figure(figsize=(10, 10))
        ax = fig_3d.add_subplot(111, projection='3d')
        
        colors = ['blue', 'green', 'red']
        markers = ['o', 's', '^']
        
        for i, quality in enumerate(self.qualities):
            mask = df_metrics['Quality'] == quality
            scatter = ax.scatter(df_metrics[mask]['THD'],
                               df_metrics[mask]['Crest Factor'],
                               df_metrics[mask]['RMS'],
                               c=colors[i],
                               marker=markers[i],
                               s=100,
                               alpha=0.6,
                               label=quality)
        
        # Enhance 3D plot appearance
        ax.set_xlabel('THD')
        ax.set_ylabel('Crest Factor')
        ax.set_zlabel('RMS')
        ax.view_init(elev=20, azim=45)
        
        # Add grid
        ax.grid(True)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.15, 0.5))
        plt.title('3D Quality Metrics Distribution\n(Click and drag to rotate)')
        
        # Make the plot interactive
        def on_key(event):
            if event.key == 'r':
                ax.view_init(elev=20, azim=45)
                plt.draw()
        
        fig_3d.canvas.mpl_connect('key_press_event', on_key)
        
        plt.tight_layout()
        
        return fig_2d, fig_3d

    def plot_advanced_metrics(self, signals, labels, comparative_results):
        """Plot advanced metrics and comparative analysis"""
        fig = plt.figure(figsize=(20, 15))
        
        # SNR Distribution
        plt.subplot(3, 2, 1)
        snr_values = [self.analyzer.calculate_snr(signal) for signal in signals]
        sns.boxplot(x=[self.qualities[l] for l in labels], y=snr_values)
        plt.title('Signal-to-Noise Ratio Distribution')
        plt.xticks(rotation=45)
        
        # Comparative Analysis
        plt.subplot(3, 2, 2)
        data = {
            'Traditional': comparative_results['traditional_scores'],
            'Ensemble': comparative_results['ensemble_scores']
        }
        sns.boxplot(data=data)
        plt.title(f'Model Comparison\np-value: {comparative_results["p_value"]:.4f}')
        
        # Power Quality Indices
        plt.subplot(3, 2, 3)
        quality_indices = [self.analyzer.calculate_power_quality_indices(signal) 
                         for signal in signals]
        pd.DataFrame(quality_indices).boxplot()
        plt.title('Power Quality Indices Distribution')
        plt.xticks(rotation=45)
        
        # Cross-validation Scores
        plt.subplot(3, 2, 4)
        cv_scores = comparative_results['ensemble_scores']
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
        plt.title(f'Cross-validation Scores\nMean: {np.mean(cv_scores):.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        
        # Model Performance Summary
        plt.subplot(3, 2, 5)
        performance_data = {
            'Model': ['Traditional', 'Ensemble'],
            'Mean Score': [comparative_results['mean_traditional'], 
                         comparative_results['mean_ensemble']],
            'Std Dev': [comparative_results['std_traditional'],
                       comparative_results['std_ensemble']]
        }
        df_performance = pd.DataFrame(performance_data)
        df_performance.plot(kind='bar', x='Model', y='Mean Score', yerr='Std Dev')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    def plot_uncertainty_analysis(self, model, X_test, predictions):
        """Plot uncertainty analysis using prediction probabilities"""
        fig = plt.figure(figsize=(15, 10))
        
        # Get prediction probabilities
        proba = model.predict_proba(X_test)
        
        # Uncertainty Distribution
        plt.subplot(2, 2, 1)
        max_proba = np.max(proba, axis=1)
        sns.histplot(max_proba, bins=30)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Maximum Probability')
        
        # Uncertainty by Class
        plt.subplot(2, 2, 2)
        uncertainty_by_class = []
        for i, quality in enumerate(self.qualities):
            mask = predictions == i
            uncertainty_by_class.append(1 - max_proba[mask])
        
        plt.boxplot(uncertainty_by_class, labels=self.qualities)
        plt.title('Uncertainty by Class')
        plt.xticks(rotation=45)
        
        # Prediction Probability Distribution
        plt.subplot(2, 2, 3)
        for i, quality in enumerate(self.qualities):
            sns.kdeplot(proba[:, i], label=quality)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability')
        plt.legend()
        
        # Cumulative Confidence Distribution
        plt.subplot(2, 2, 4)
        confidence_thresh = np.linspace(0, 1, 100)
        confident_predictions = [np.mean(max_proba >= thresh) for thresh in confidence_thresh]
        plt.plot(confidence_thresh, confident_predictions)
        plt.title('Cumulative Confidence Distribution')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Fraction of Predictions')
        
        plt.tight_layout()
        return fig


def create_output_directory():
    """Create directory for output files"""
    output_dir = 'power_quality_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    try:
        # Create output directory
        output_dir = create_output_directory()
        
        # Initialize enhanced analyzer and visualizer
        print("Initializing enhanced analyzer and visualizer...")
        pqa = AdvancedPowerQualityAnalyzer()
        visualizer = EnhancedPowerQualityVisualizer(pqa)
        
        # Generate and process data
        print("Generating synthetic data...")
        signals, labels = pqa.generate_synthetic_data(n_samples=5000)
        
        print("Extracting enhanced features...")
        features = pqa.extract_enhanced_features(signals)
        
        # Split and scale data
        print("Preparing training and test datasets...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Perform comparative analysis
        print("Performing comparative analysis...")
        comparative_results = pqa.perform_comparative_analysis(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Train models
        print("Training models...")
        # Traditional ensemble model
        ensemble_model = pqa.build_advanced_model()
        ensemble_model.fit(X_train_scaled, y_train)
        ensemble_predictions = ensemble_model.predict(X_test_scaled)
        
        # Deep learning model
        print("Training deep learning model...")
        deep_model = pqa.build_deep_model()
        deep_model.fit(X_train_scaled, y_train)
        deep_predictions = deep_model.predict(X_test_scaled)
        
        # Generate and save all visualizations
        print("\nGenerating enhanced visualizations...")
        
        # Original visualizations
        print("- Generating signal examples...")
        visualizer.plot_signal_examples(signals, labels).savefig(
            os.path.join(output_dir, 'signal_examples.png'), dpi=300, bbox_inches='tight')
        
        print("- Generating performance metrics...")
        visualizer.plot_performance_metrics(
            ensemble_model, X_train_scaled, X_test_scaled, y_train, y_test, ensemble_predictions
        ).savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        
        print("- Generating quality metrics...")
        fig_2d, fig_3d = visualizer.plot_quality_metrics(signals, labels)
        fig_2d.savefig(os.path.join(output_dir, 'quality_metrics_2d.png'), dpi=300, bbox_inches='tight')
        fig_3d.savefig(os.path.join(output_dir, 'quality_metrics_3d.png'), dpi=300, bbox_inches='tight')
        
        # New enhanced visualizations
        print("- Generating advanced metrics...")
        visualizer.plot_advanced_metrics(
            signals, labels, comparative_results
        ).savefig(os.path.join(output_dir, 'advanced_metrics.png'), dpi=300, bbox_inches='tight')
        
        print("- Generating uncertainty analysis...")
        visualizer.plot_uncertainty_analysis(
            ensemble_model, X_test_scaled, ensemble_predictions
        ).savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
        
        # Save classification reports
        print("\nGenerating classification reports...")
        ensemble_report = classification_report(y_test, ensemble_predictions, 
                                             target_names=visualizer.qualities,
                                             output_dict=True)
        pd.DataFrame(ensemble_report).transpose().to_csv(
            os.path.join(output_dir, 'ensemble_classification_report.csv'))
        
        deep_report = classification_report(y_test, deep_predictions, 
                                         target_names=visualizer.qualities,
                                         output_dict=True)
        pd.DataFrame(deep_report).transpose().to_csv(
            os.path.join(output_dir, 'deep_learning_report.csv'))
        
        # Save comparative analysis results
        pd.DataFrame(comparative_results).to_csv(
            os.path.join(output_dir, 'comparative_analysis.csv'))
        
        # Print results location
        print("\nEnhanced results have been saved to the '{}' directory:".format(output_dir))
        print("1. signal_examples.png - Signal analysis visualizations")
        print("2. performance_metrics.png - Model performance metrics")
        print("3. quality_metrics.png - Power quality metrics analysis")
        print("4. advanced_metrics.png - Advanced metrics and comparative analysis")
        print("5. uncertainty_analysis.png - Uncertainty quantification analysis")
        print("6. ensemble_classification_report.csv - Detailed classification metrics")
        print("7. deep_learning_report.csv - Deep learning model metrics")
        print("8. comparative_analysis.csv - Comparative analysis results")
        
        # Display plots (optional)
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()