import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def make_final_plots(save_dir, train_losses, val_losses, metric_name, train_metrics, val_metrics, 
                     grad_norms, model, activations, predictions, targets):
    """
    Generates and saves the requested plots.

    Args:
        save_dir (str): Directory to save plots.
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        metric_name (str): Name of the metric (e.g., 'MAE').
        train_metrics (list): List of training metrics per epoch.
        val_metrics (list): List of validation metrics per epoch.
        grad_norms (list): List of average gradient norms per epoch.
        model (torch.nn.Module): The trained model.
        activations (dict): Dictionary of {layer_name: numpy_array} for activation distributions.
        predictions (numpy.ndarray): Predictions array of shape [N, D].
        targets (numpy.ndarray): Targets array of shape [N, D].
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Training and validation loss and metric vs epoch
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    
    if train_metrics and val_metrics:
        # Plot metrics on a secondary y-axis or normalized? 
        # For simplicity, plotting on same axis if range is compatible, 
        # otherwise consider subplots or secondary axis. 
        # Assuming similar scale or just plotting them.
        plt.plot(epochs, train_metrics, label=f'Train {metric_name}', linestyle='--')
        plt.plot(epochs, val_metrics, label=f'Val {metric_name}', linestyle='--')
        
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training/Validation Loss and Metric vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, '1_loss_metric_vs_epoch.png'))
    plt.close()

    # 2. Gradient norm averaged per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norms, label='Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Average Gradient Norm per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, '2_grad_norm_vs_epoch.png'))
    plt.close()

    # 3. Final weight histograms
    # Create a subdirectory for weights to avoid clutter
    weights_dir = os.path.join(save_dir, '3_weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure(figsize=(8, 5))
            data = param.data.cpu().numpy().flatten()
            plt.hist(data, bins=50, alpha=0.7)
            plt.title(f'Weight Distribution - {name}')
            clean_name = name.replace('.', '_')
            plt.grid(True)
            plt.savefig(os.path.join(weights_dir, f'{clean_name}.png'))
            plt.close()

    # 4. Activation distributions
    # Create a subdirectory for activations
    acts_dir = os.path.join(save_dir, '4_activations')
    os.makedirs(acts_dir, exist_ok=True)
    
    for name, acts in activations.items():
        plt.figure(figsize=(8, 5))
        # Using simple histogram instead of seaborn kde
        plt.hist(acts.flatten(), bins=50, alpha=0.7)
        plt.title(f'Activation Distribution - {name}')
        clean_name = name.replace('.', '_')
        plt.grid(True)
        plt.savefig(os.path.join(acts_dir, f'{clean_name}.png'))
        plt.close()

    # 5. Predicted vs. true scatter plot
    # Handle both 2D arrays and lists of arrays
    if isinstance(predictions, list):
        # If list, assume each element is a feature channel [N_samples]
        num_features = len(predictions)
        feature_labels = [f'Feature {i}' for i in range(num_features)]
        if num_features == 4:
            feature_labels = ['Vel X', 'Vel Y', 'Vel Z', 'Stress']
            
        iter_predictions = predictions
        iter_targets = targets
    else:
        # If 2D array [N, D]
        num_features = predictions.shape[1]
        feature_labels = [f'Feature {i}' for i in range(num_features)]
        if num_features == 4:
            feature_labels = ['Vel X', 'Vel Y', 'Vel Z', 'Stress']
            
        iter_predictions = [predictions[:, i] for i in range(num_features)]
        iter_targets = [targets[:, i] for i in range(num_features)]

    pred_dir = os.path.join(save_dir, '5_pred_vs_true')
    os.makedirs(pred_dir, exist_ok=True)

    for i in range(num_features):
        plt.figure(figsize=(8, 8))
        y_true = iter_targets[i]
        y_pred = iter_predictions[i]
        
        # Scatter with subsampling if too many points
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            plt.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=5)
        else:
            plt.scatter(y_true, y_pred, alpha=0.5, s=10)
            
        # Diagonal line
        if len(y_true) > 0:
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Predicted vs True - {feature_labels[i]}')
        plt.grid(True)
        plt.savefig(os.path.join(pred_dir, f'feat_{i}.png'))
        plt.close()

    # 6. Residuals vs. true value
    res_dir = os.path.join(save_dir, '6_residuals')
    os.makedirs(res_dir, exist_ok=True)
    
    for i in range(num_features):
        plt.figure(figsize=(10, 6))
        y_true = iter_targets[i]
        y_pred = iter_predictions[i]
        res = y_pred - y_true
        
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            plt.scatter(y_true[idx], res[idx], alpha=0.3, s=5)
        else:
            plt.scatter(y_true, res, alpha=0.5, s=10)
            
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('True Value')
        plt.ylabel('Residual (Pred - True)')
        plt.title(f'Residuals vs True - {feature_labels[i]}')
        plt.grid(True)
        plt.savefig(os.path.join(res_dir, f'feat_{i}.png'))
        plt.close()
