#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Logger:
    def __init__(self, metric_names):
        """
        Initialization.

        Args:
            metric_names (list of str): List of metric names to be logged
        """
        # Initialize empty lists to store the loss and metric data
        self.train_loss = []  # For TrLoss
        self.val_loss = []    # For ValLoss
        self.metrics = [[] for _ in range(len(metric_names))]
        self.metric_names = metric_names  # List of metric names

    def log(self,
            tr_loss,
            val_loss,
            metrics):
        """
        Log the loss and metrics for a single epoch.

        Args:
            tr_loss (float): Training loss for the current epoch
            val_loss (float): Validation loss for the current epoch
            metrics (list of float): 
                List of metric values for the current epoch
        """
        # Append the loss data
        self.train_loss.append(tr_loss)
        self.val_loss.append(val_loss)

        # log
        assert len(self.metric_names) == len(metrics), "Wrong #metrics to log"
        # Append the corresponding metrics for the current epoch
        for i, metric_value in enumerate(metrics):
            self.metrics[i].append(metric_value)

    def save(self, log_save_path):
        """
        Save the logged data (loss and metrics) into a CSV file.

        Args:
            log_save_path (str): Path to save the CSV log file
        """
        # Prepare header and data rows
        header = ['Epoch', 'TrLoss', 'ValLoss'] + self.metric_names
        rows = []
        
        # Create rows for each epoch
        for epoch in range(len(self.train_loss)):
            row = [epoch + 1, self.train_loss[epoch], self.val_loss[epoch]]
            row.extend(self.metrics[i][epoch] for i in range(len(self.metrics)))
            rows.append(row)
        
        # Write the data to a CSV file
        with open(log_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def plot(self,
             fig_save_path,
             fig2_ylabel):
        """
        Plot the loss curves and validation 
        metrics in one figure with two subplots.

        Args:
            fig_save_path (str): Path to save the figure
            fig2_ylabel (str): Y-axis label for the validation metrics subplot
        """
        
        epochs = range(1, len(self.train_loss) + 1)

        # Create a figure with two subplots (1x2 grid)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot Loss Curve (Figure 1)
        ax1.plot(epochs, self.train_loss, label='TrLoss', color='blue')
        ax1.plot(epochs, self.val_loss, label='ValLoss', color='red')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot Validation Metrics (Figure 2)
        nb_metrics = len(self.metric_names)
        color_map = cm.get_cmap('tab20', nb_metrics)
        for i, metric in enumerate(self.metrics):
            ax2.plot(epochs, metric, label=self.metric_names[i], color=color_map(i))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel(fig2_ylabel)
        ax2.set_title('Validation Metrics')
        ax2.legend(ncol=2)
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=400)
        plt.cla()
