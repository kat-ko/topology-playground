# Observation Shift Analysis

This analysis examines the actual perturbation values added to observations across different levels in continual learning experiments.

## Overview

The observation shift analysis plots the shift values for each observation dimension across levels, with confidence intervals calculated across seeds. This helps visualize how the noise/shift changes across levels for each observation dimension.

## Key Features

- **X-axis**: Levels (0, 1, 2, 3, ... up to 15)
- **Y-axis**: The actual shift/noise values added to observations
- **Data source**: All experiments starting with `HYB_*` in specified folder
- **Plotting style**: Same as Figure 6 plots with grey dotted lines at level transitions
- **Aggregation**: Average across seeds with bootstrap confidence intervals
- **Purpose**: Show how the noise/shift changes across levels for each observation dimension

## Files

### Python Script
- **`observation_shift_analysis.py`**: Standalone script for generating observation shift plots

### Usage
```bash
# Run from test_experiments directory
python observation_shift_analysis.py
```

### Output
- **`observation_shifts_cartpole_N0002_S256.png`**: High-resolution PNG plot
- **`observation_shifts_cartpole_N0002_S256.pdf`**: Vector PDF plot

## Data Structure

The script reads from `shift_data.csv` files in each experiment directory, which contain:
- `shift_step`: Timestep when shift occurred
- `shift_id`: Level number (1, 2, 3, ...)
- `offset_repr`: String representation of shift values for each observation dimension
- `seed`: Experiment seed
- `env`: Environment name
- `topology`: Network topology type

## Example Results

The script successfully analyzed:
- **14 seeds** from `cartpole/N0002/S256`
- **14 levels** per seed
- **4 observation dimensions** (Cartpole has 4 observation features)

## Plot Features

- **Multi-line plot**: One line per observation dimension
- **Confidence bands**: Bootstrap confidence intervals around each line
- **Grey dotted lines**: Vertical lines at level transitions
- **Consistent styling**: Matches Figure 6 plot aesthetics
- **Color coding**: Each observation dimension has a distinct color

## Customization

To analyze different subfolders, modify the `data_folder` variable in the script:

```python
# Different noise levels
data_folder = "cartpole/N0001/S256"  # N0001 noise level
data_folder = "cartpole/N0003/S256"  # N0003 noise level

# Different network sizes
data_folder = "cartpole/N0002/S128"  # S128 network size
data_folder = "cartpole/N0002/S384"  # S384 network size

# Different tasks
data_folder = "acrobot/N0002/S256"   # Acrobot task
data_folder = "lunarlander/N0002/S256"  # Lunarlander task
```

## Technical Details

- **Bootstrap confidence intervals**: 1000 bootstrap samples, 95% confidence level
- **Data parsing**: Uses `ast.literal_eval()` to parse string representations of lists
- **Error handling**: Graceful handling of missing or malformed data
- **Plot styling**: White background, green grid, black labels, consistent with Figure 6 plots
