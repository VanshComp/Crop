import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
crops = pd.read_csv('../data/crops.csv')
plots = pd.read_csv('../data/plots.csv')
time_periods = pd.read_csv('../data/time_periods.csv')
crop_rotations = pd.read_csv('../data/crop_rotations.csv')

# Processing data
def get_crop_rotation_schedule():
    # Merge dataframes to get a complete view of the crop rotation schedule
    schedule = crop_rotations.merge(crops, on='CropID').merge(plots, on='PlotID').merge(time_periods, on='TimePeriodID')
    return schedule

schedule = get_crop_rotation_schedule()

# Function to check crop rotation criteria
def is_valid_rotation(schedule):
    plot_schedules = schedule.groupby('PlotName')
    for plot_name, plot_schedule in plot_schedules:
        last_crop_family = None
        for index, row in plot_schedule.iterrows():
            current_family = row['BotanicFamily']
            if last_crop_family == current_family:
                return False, f"Invalid rotation in {plot_name} due to successive planting of the same family."
            last_crop_family = current_family
    return True, "Valid rotation."

valid, message = is_valid_rotation(schedule)
print(message)

# Updated visualization function with alternating column colors, row lines, and two-line text
def visualize_crop_rotation(schedule):
    # Pivot the data for heatmap
    pivot_table = schedule.pivot_table(index='PlotName', columns='Month', values='CropName', aggfunc='first', fill_value='')

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Create alternating column colors
    cols = pivot_table.columns
    colors = ['#d4e157' if i % 2 == 0 else '#aed581' for i in range(len(cols))]

    # Draw a heatmap with no color information
    sns.heatmap(pivot_table.isnull(), cmap='viridis', cbar=False, xticklabels=False, yticklabels=False, annot=False, linewidths=1, linecolor='black')

    # Overlay the colored blocks
    for idx, col in enumerate(pivot_table.columns):
        plt.gca().add_patch(plt.Rectangle((idx, 0), 1, pivot_table.shape[0], fill=True, color=colors[idx], linewidth=0))

    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = pivot_table.iloc[i, j]
            if isinstance(text, str) and ' ' in text:
                text = '\n'.join(text.split())
            plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black')

    # Set axis labels and ticks
    plt.title('Crop Rotation Schedule')
    plt.xlabel('Month')
    plt.ylabel('Plot')
    plt.xticks(ticks=range(len(pivot_table.columns)), labels=pivot_table.columns, rotation=90)
    plt.yticks(ticks=range(len(pivot_table.index)), labels=pivot_table.index, rotation=0)

    plt.show()

visualize_crop_rotation(schedule)
