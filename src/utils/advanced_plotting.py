"""
Advanced plotting utilities for topology network analysis.
Handles the complexity of different training phases and task orders.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
import wandb


# Color schemes for consistent visualization
TOPOLOGY_COLORS = {
    'small_world': '#1f77b4',
    'modular': '#ff7f0e', 
    'hybrid': '#2ca02c',
    'fully_connected': '#d62728'
}

TASK_COLORS = {
    'CartPole-v1': '#1f77b4',
    'Acrobot-v1': '#ff7f0e',
    'LunarLander-v2': '#2ca02c'
}

# Task order combinations
DOUBLE_TASK_ORDERS = [
    'CartPole-v1_Acrobot-v1', 'CartPole-v1_LunarLander-v2',
    'Acrobot-v1_CartPole-v1', 'Acrobot-v1_LunarLander-v2',
    'LunarLander-v2_CartPole-v1', 'LunarLander-v2_Acrobot-v1'
]

TRIPLE_TASK_ORDERS = [
    'CartPole-v1_Acrobot-v1_LunarLander-v2', 'CartPole-v1_LunarLander-v2_Acrobot-v1',
    'Acrobot-v1_CartPole-v1_LunarLander-v2', 'Acrobot-v1_LunarLander-v2_CartPole-v1',
    'LunarLander-v2_CartPole-v1_Acrobot-v1', 'LunarLander-v2_Acrobot-v1_CartPole-v1'
]

ALL_TASKS = ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
ALL_TOPOLOGIES = ['small_world', 'modular', 'hybrid', 'fully_connected']


def parse_task_sequence(task_sequence: str) -> Tuple[List[str], int]:
    """
    Parse task sequence to extract trained tasks and number of phases.
    
    Args:
        task_sequence: Task sequence string (e.g., 'CartPole-v1_Acrobot-v1')
    
    Returns:
        Tuple of (trained_tasks, num_phases)
    """
    if '_' not in task_sequence:
        return [task_sequence], 1
    
    trained_tasks = task_sequence.split('_')
    num_phases = len(trained_tasks)
    
    return trained_tasks, num_phases


def get_trained_task_for_phase(task_sequence: str, phase: int) -> str:
    """
    Get the task that was trained in a specific phase.
    
    Args:
        task_sequence: Task sequence string
        phase: Phase number (1-indexed)
    
    Returns:
        Task name that was trained in this phase
    """
    trained_tasks, _ = parse_task_sequence(task_sequence)
    if 1 <= phase <= len(trained_tasks):
        return trained_tasks[phase - 1]
    return "Unknown"


def create_multi_phase_learning_curves(
    phase_results: Dict, 
    topology_type: str, 
    task_sequence: str
) -> go.Figure:
    """
    Create learning curves showing performance evolution across all phases.
    This is the MOST important plot for topology comparison.
    
    Args:
        phase_results: Dictionary containing phase results
        topology_type: Type of topology
        task_sequence: Task sequence string
    
    Returns:
        Plotly figure showing learning progression
    """
    
    trained_tasks, num_phases = parse_task_sequence(task_sequence)
    
    fig = go.Figure()
    
    # Add performance lines for each task
    for task in ALL_TASKS:
        task_performance = []
        phases = []
        
        for phase_num in range(1, num_phases + 1):
            phase_key = f'phase{phase_num}'
            metric_key = f'{topology_type}/{task_sequence}/{phase_key}/testing/{task}/mean_reward'
            
            if metric_key in phase_results:
                task_performance.append(phase_results[metric_key])
                phases.append(f'Phase {phase_num}')
        
        if task_performance:  # Only add if we have data
            fig.add_trace(go.Scatter(
                x=phases,
                y=task_performance,
                name=task,
                line=dict(color=TASK_COLORS[task], width=3),
                mode='lines+markers',
                marker=dict(size=8),
                hovertemplate=f"<b>{task}</b><br>" +
                             "Phase: %{x}<br>" +
                             "Reward: %{y:.1f}<br>" +
                             "<extra></extra>"
            ))
    
    # Add vertical lines to separate training phases
    for phase_num in range(1, num_phases):
        trained_task = get_trained_task_for_phase(task_sequence, phase_num)
        fig.add_vline(
            x=phase_num - 0.5, 
            line_dash="dash", 
            line_color="gray", 
            line_width=2,
            annotation_text=f"Trained on: {trained_task}",
            annotation_position="top right",
            annotation_font_size=10
        )
    
    fig.update_layout(
        title=f"Learning Progression: {topology_type} - {task_sequence}",
        xaxis_title="Training Phase",
        yaxis_title="Mean Reward",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        width=800
    )
    
    return fig


def create_transfer_comparison_for_task_order(
    all_topology_results: Dict, 
    task_sequence: str
) -> go.Figure:
    """
    Create transfer learning comparison for ONE specific task order.
    Shows which topologies are best at transfer learning for this sequence.
    
    Args:
        all_topology_results: Results from all topologies
        task_sequence: Task sequence string
    
    Returns:
        Plotly figure showing transfer learning comparison
    """
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Forward Transfer", "Backward Transfer (Retention)", 
                       "Catastrophic Forgetting", "Overall Transfer Score"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Extract transfer metrics for THIS specific task order
    forward_scores = []
    backward_scores = []
    forgetting_scores = []
    overall_scores = []
    
    for topology in ALL_TOPOLOGIES:
        # Forward transfer
        forward_key = f'{topology}/{task_sequence}/transfer/forward_transfer_score'
        forward = all_topology_results.get(forward_key, 0)
        forward_scores.append(forward)
        
        # Backward transfer (retention)
        backward_key = f'{topology}/{task_sequence}/transfer/backward_transfer_score'
        backward = all_topology_results.get(backward_key, 0)
        backward_scores.append(backward)
        
        # Catastrophic forgetting
        forgetting_key = f'{topology}/{task_sequence}/transfer/catastrophic_forgetting'
        forgetting = all_topology_results.get(forgetting_key, 0)
        forgetting_scores.append(forgetting)
        
        # Overall transfer score (normalized combination)
        overall = (forward / 100) * 0.5 + backward * 0.5
        overall_scores.append(overall)
    
    # Forward Transfer
    fig.add_trace(go.Bar(
        x=ALL_TOPOLOGIES,
        y=forward_scores,
        name="Forward Transfer",
        marker_color=[TOPOLOGY_COLORS[t] for t in ALL_TOPOLOGIES],
        text=[f"{score:.1f}" for score in forward_scores],
        textposition='auto',
        showlegend=False
    ), row=1, col=1)
    
    # Backward Transfer (Retention)
    fig.add_trace(go.Bar(
        x=ALL_TOPOLOGIES,
        y=backward_scores,
        name="Retention",
        marker_color=[TOPOLOGY_COLORS[t] for t in ALL_TOPOLOGIES],
        text=[f"{score:.3f}" for score in backward_scores],
        textposition='auto',
        showlegend=False
    ), row=1, col=2)
    
    # Catastrophic Forgetting
    fig.add_trace(go.Bar(
        x=ALL_TOPOLOGIES,
        y=forgetting_scores,
        name="Forgetting",
        marker_color=[TOPOLOGY_COLORS[t] for t in ALL_TOPOLOGIES],
        text=[f"{score:.3f}" for score in forgetting_scores],
        textposition='auto',
        showlegend=False
    ), row=2, col=1)
    
    # Overall Transfer Score
    fig.add_trace(go.Bar(
        x=ALL_TOPOLOGIES,
        y=overall_scores,
        name="Overall Score",
        marker_color=[TOPOLOGY_COLORS[t] for t in ALL_TOPOLOGIES],
        text=[f"{score:.3f}" for score in overall_scores],
        textposition='auto',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title=f"Transfer Learning Comparison: {task_sequence}",
        height=600,
        width=1000,
        showlegend=False
    )
    
    return fig


def create_performance_matrix_for_task_order(
    sweep_results: Dict, 
    task_sequence: str
) -> go.Figure:
    """
    Create performance matrix for ONE specific task order.
    Shows topology performance across all tasks and phases.
    
    Args:
        sweep_results: Results from the sweep
        task_sequence: Task sequence string
    
    Returns:
        Plotly figure showing performance matrix
    """
    
    trained_tasks, num_phases = parse_task_sequence(task_sequence)
    
    # Create subplots for each phase
    fig = make_subplots(
        rows=1, cols=num_phases,
        subplot_titles=[f"Phase {i+1}" for i in range(num_phases)],
        specs=[[{"type": "heatmap"} for _ in range(num_phases)]]
    )
    
    for phase_idx, phase_num in enumerate(range(1, num_phases + 1)):
        phase_key = f'phase{phase_num}'
        
        # Create data matrix for this phase
        data_matrix = []
        for topology in ALL_TOPOLOGIES:
            topology_data = []
            for task in ALL_TASKS:
                metric_key = f'{topology}/{task_sequence}/{phase_key}/testing/{task}/mean_reward'
                value = sweep_results.get(metric_key, 0)
                topology_data.append(value)
            data_matrix.append(topology_data)
        
        # Add heatmap for this phase
        fig.add_trace(go.Heatmap(
            z=data_matrix,
            x=ALL_TASKS,
            y=ALL_TOPOLOGIES,
            colorscale='Viridis',
            text=data_matrix,
            texttemplate="%{text:.0f}",
            textfont={"size": 12},
            colorbar=dict(title="Mean Reward", len=0.3, y=0.5),
            name=f"Phase {phase_num}",
            showscale=(phase_idx == num_phases - 1)  # Only show colorbar for last subplot
        ), row=1, col=phase_idx+1)
    
    fig.update_layout(
        title=f"Topology Performance Matrix: {task_sequence}",
        height=400,
        width=300 * num_phases
    )
    
    return fig


def create_capacity_scaling_for_task_order(
    sweep_results: Dict, 
    topology_type: str, 
    task_sequence: str
) -> go.Figure:
    """
    Show how topology performance scales with different capacities for ONE task order.
    
    Args:
        sweep_results: Results from the sweep
        topology_type: Type of topology
        task_sequence: Task sequence string
    
    Returns:
        Plotly figure showing capacity scaling
    """
    
    trained_tasks, num_phases = parse_task_sequence(task_sequence)
    capacities = [1000, 5000, 10000, 50000]
    
    fig = go.Figure()
    
    for task in ALL_TASKS:
        task_performance = []
        for capacity in capacities:
            # Aggregate across all phases for this capacity
            phase_performances = []
            for phase_num in range(1, num_phases + 1):
                phase_key = f'phase{phase_num}'
                metric_key = f'{topology_type}/{task_sequence}/{phase_key}/testing/{task}/mean_reward'
                
                # Filter by capacity (would need capacity info in results)
                # For now, use the metric if it exists
                if metric_key in sweep_results:
                    phase_performances.append(sweep_results[metric_key])
            
            # Use average performance across phases
            avg_performance = np.mean(phase_performances) if phase_performances else 0
            task_performance.append(avg_performance)
        
        fig.add_trace(go.Scatter(
            x=capacities,
            y=task_performance,
            name=task,
            line=dict(color=TASK_COLORS[task], width=3),
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate=f"<b>{task}</b><br>" +
                         "Capacity: %{x:,} params<br>" +
                         "Performance: %{y:.1f}<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Capacity Scaling: {topology_type} - {task_sequence}",
        xaxis_title="Parameter Capacity",
        yaxis_title="Mean Reward",
        xaxis_type="log",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        width=800
    )
    
    return fig


def create_sequential_task_performance_plot(
    sweep_results: Dict, 
    task_sequence: str,
    training_type: str = 'double_task'
) -> go.Figure:
    """
    Create sequential performance plot showing how each task's performance evolves
    throughout the training sequence. This is CRUCIAL for continual learning analysis.
    
    X-axis: Training phases (which task was trained)
    Y-axis: Performance on a specific task
    Multiple lines: One for each tested task
    
    This directly shows:
    - Forward transfer (how training on earlier tasks helps later tasks)
    - Backward transfer/retention (how well earlier tasks are maintained)
    - Catastrophic forgetting (if performance on earlier tasks degrades)
    
    Args:
        sweep_results: Dictionary containing results from all topologies
        task_sequence: Task sequence string (e.g., 'CartPole-v1_Acrobot-v1')
        training_type: Type of training ('double_task' or 'triple_task')
    
    Returns:
        Plotly figure showing sequential performance evolution
    """
    
    trained_tasks, num_phases = parse_task_sequence(task_sequence)
    
    # Create subplots: one for each topology
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{topology.replace("_", " ").title()}' for topology in ALL_TOPOLOGIES],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # X-axis labels: training phases
    x_labels = []
    for i, task in enumerate(trained_tasks):
        if i == 0:
            x_labels.append(f"Initial\n{task}")
        else:
            x_labels.append(f"After {task}")
    
    # For each topology
    for topology_idx, topology_type in enumerate(ALL_TOPOLOGIES):
        row = (topology_idx // 2) + 1
        col = (topology_idx % 2) + 1
        
        # Get results for this topology and task sequence
        topology_key = f"{topology_type}/{task_sequence}"
        
        # For each tested task
        for task_idx, tested_task in enumerate(ALL_TASKS):
            performance_data = []
            
            # Collect performance data for each phase
            for phase_num in range(1, num_phases + 1):
                metric_key = f"{topology_key}/phase{phase_num}/testing/{tested_task}/mean_reward"
                
                if metric_key in sweep_results:
                    performance_data.append(sweep_results[metric_key])
                else:
                    # If no data, use None to create gaps in the line
                    performance_data.append(None)
            
            # Add line for this task
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=performance_data,
                    mode='lines+markers',
                    name=f'{tested_task} ({topology_type})',
                    line=dict(
                        color=TASK_COLORS[tested_task],
                        width=2,
                        dash='solid' if topology_type == 'fully_connected' else 'dash' if topology_type == 'small_world' else 'dot' if topology_type == 'modular' else 'dashdot'
                    ),
                    marker=dict(
                        size=8,
                        symbol='circle' if topology_type == 'fully_connected' else 'square' if topology_type == 'small_world' else 'diamond' if topology_type == 'modular' else 'triangle-up'
                    ),
                    showlegend=False,  # We'll add legend separately
                    hovertemplate=f'<b>{tested_task}</b><br>' +
                                f'Topology: {topology_type}<br>' +
                                f'Phase: %{{x}}<br>' +
                                f'Performance: %{{y:.2f}}<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
    
    # Add a separate legend
    legend_fig = go.Figure()
    for tested_task in ALL_TASKS:
        legend_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines+markers',
                name=tested_task,
                line=dict(color=TASK_COLORS[tested_task], width=3),
                marker=dict(size=10),
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Sequential Task Performance Evolution<br><sub>Task Sequence: {task_sequence} | Training Type: {training_type}</sub>",
        title_x=0.5,
        title_font_size=16,
        height=800,
        width=1200,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes for each subplot
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                title_text="Training Phase",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                row=i, col=j
            )
            fig.update_yaxes(
                title_text="Mean Reward",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                row=i, col=j
            )
    
    return fig


def create_task_specific_topology_comparison_plot(
    sweep_results: Dict, 
    task_sequence: str,
    tested_task: str,
    training_type: str = 'double_task'
) -> go.Figure:
    """
    Create topology comparison plot for a specific tested task.
    Shows how all 4 topologies perform on one specific task throughout training.
    
    Args:
        sweep_results: Dictionary containing results from all topologies
        task_sequence: Task sequence string (e.g., 'LunarLander-v2_CartPole-v1')
        tested_task: The specific task being tested (e.g., 'CartPole-v1')
        training_type: Type of training ('double_task' or 'triple_task')
    
    Returns:
        Plotly figure showing topology comparison for one specific task
    """
    
    trained_tasks, num_phases = parse_task_sequence(task_sequence)
    
    fig = go.Figure()
    
    # X-axis labels: training phases
    x_labels = []
    for i, task in enumerate(trained_tasks):
        if i == 0:
            x_labels.append(f"Initial\n{task}")
        else:
            x_labels.append(f"After {task}")
    
    # For each topology
    for topology_type in ALL_TOPOLOGIES:
        performance_data = []
        
        # Get results for this topology and task sequence
        topology_key = f"{topology_type}/{task_sequence}"
        
        # Collect performance data for each phase
        for phase_num in range(1, num_phases + 1):
            metric_key = f"{topology_key}/phase{phase_num}/testing/{tested_task}/mean_reward"
            
            if metric_key in sweep_results:
                performance_data.append(sweep_results[metric_key])
            else:
                performance_data.append(None)
        
        # Add line for this topology
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=performance_data,
                mode='lines+markers',
                name=topology_type.replace('_', ' ').title(),
                line=dict(
                    color=TOPOLOGY_COLORS[topology_type],
                    width=3
                ),
                marker=dict(
                    size=10,
                    symbol='circle' if topology_type == 'fully_connected' else 'square' if topology_type == 'small_world' else 'diamond' if topology_type == 'modular' else 'triangle-up'
                ),
                hovertemplate=f'<b>{topology_type.replace("_", " ").title()}</b><br>' +
                            f'Task: {tested_task}<br>' +
                            f'Phase: %{{x}}<br>' +
                            f'Performance: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Topology Comparison: Performance on {tested_task}<br><sub>Task Sequence: {task_sequence} | Training Type: {training_type}</sub>",
        title_x=0.5,
        title_font_size=16,
        height=600,
        width=800,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Training Phase",
        yaxis_title="Mean Reward",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig


def create_single_topology_sequential_plot(
    sweep_results: Dict, 
    topology_type: str,
    task_sequence: str,
    training_type: str = 'double_task'
) -> go.Figure:
    """
    Create sequential performance plot for a single topology.
    This provides a clearer view of continual learning patterns.
    
    Args:
        sweep_results: Dictionary containing results
        topology_type: Type of topology to plot
        task_sequence: Task sequence string
        training_type: Type of training
    
    Returns:
        Plotly figure showing sequential performance for one topology
    """
    
    trained_tasks, num_phases = parse_task_sequence(task_sequence)
    
    fig = go.Figure()
    
    # X-axis labels: training phases
    x_labels = []
    for i, task in enumerate(trained_tasks):
        if i == 0:
            x_labels.append(f"Initial\n{task}")
        else:
            x_labels.append(f"After {task}")
    
    # Get results for this topology and task sequence
    topology_key = f"{topology_type}/{task_sequence}"
    
    # For each tested task
    for tested_task in ALL_TASKS:
        performance_data = []
        
        # Collect performance data for each phase
        for phase_num in range(1, num_phases + 1):
            metric_key = f"{topology_key}/phase{phase_num}/testing/{tested_task}/mean_reward"
            
            if metric_key in sweep_results:
                performance_data.append(sweep_results[metric_key])
            else:
                performance_data.append(None)
        
        # Add line for this task
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=performance_data,
                mode='lines+markers',
                name=tested_task,
                line=dict(
                    color=TASK_COLORS[tested_task],
                    width=3
                ),
                marker=dict(
                    size=10,
                    symbol='circle'
                ),
                hovertemplate=f'<b>{tested_task}</b><br>' +
                            f'Topology: {topology_type}<br>' +
                            f'Phase: %{{x}}<br>' +
                            f'Performance: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Sequential Task Performance: {topology_type.replace('_', ' ').title()}<br><sub>Task Sequence: {task_sequence} | Training Type: {training_type}</sub>",
        title_x=0.5,
        title_font_size=16,
        height=600,
        width=800,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Training Phase",
        yaxis_title="Mean Reward",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig


def create_task_order_effects_plot(
    sweep_results: Dict, 
    topology_type: str,
    training_type: str = 'double_task'
) -> go.Figure:
    """
    Analyze how different task orders affect topology performance.
    
    Args:
        sweep_results: Results from the sweep
        topology_type: Type of topology
        training_type: 'double_task' or 'triple_task'
    
    Returns:
        Plotly figure showing task order effects
    """
    
    if training_type == 'double_task':
        task_orders = DOUBLE_TASK_ORDERS
        phases = ['phase1', 'phase2']
    else:  # triple_task
        task_orders = TRIPLE_TASK_ORDERS
        phases = ['phase1', 'phase2', 'phase3']
    
    fig = go.Figure()
    
    for task_seq in task_orders:
        # Calculate overall performance for this task sequence
        overall_performance = []
        for phase in phases:
            phase_performance = []
            for task in ALL_TASKS:
                metric_key = f'{topology_type}/{task_seq}/{phase}/testing/{task}/mean_reward'
                if metric_key in sweep_results:
                    phase_performance.append(sweep_results[metric_key])
            
            if phase_performance:
                overall_performance.append(np.mean(phase_performance))
        
        # Plot performance for this task sequence
        fig.add_trace(go.Scatter(
            x=[f'Phase {i+1}' for i in range(len(phases))],
            y=overall_performance,
            name=task_seq,
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate=f"<b>{task_seq}</b><br>" +
                         "Phase: %{x}<br>" +
                         "Performance: %{y:.1f}<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Task Order Effects: {topology_type} ({training_type})",
        xaxis_title="Training Phase",
        yaxis_title="Overall Performance",
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        height=500,
        width=1000
    )
    
    return fig


def generate_all_plots_for_sweep(sweep_results: Dict) -> Dict:
    """
    Generate ALL plots for ALL task orders.
    
    Args:
        sweep_results: Results from the sweep
    
    Returns:
        Dictionary containing all plots organized by task order
    """
    
    all_plots = {}
    
    # Generate plots for each double-task order
    for task_order in DOUBLE_TASK_ORDERS:
        plots = {}
        
        # Learning curves for each topology
        for topology in ALL_TOPOLOGIES:
            plots[f'learning_curves_{topology}'] = create_multi_phase_learning_curves(
                sweep_results, topology, task_order
            )
        
        # Transfer comparison
        plots['transfer_comparison'] = create_transfer_comparison_for_task_order(
            sweep_results, task_order
        )
        
        # Performance matrix
        plots['performance_matrix'] = create_performance_matrix_for_task_order(
            sweep_results, task_order
        )
        
        # Capacity scaling for each topology
        for topology in ALL_TOPOLOGIES:
            plots[f'capacity_scaling_{topology}'] = create_capacity_scaling_for_task_order(
                sweep_results, topology, task_order
            )
        
        # Sequential performance plot (all topologies comparison)
        plots['sequential_performance_comparison'] = create_sequential_task_performance_plot(
            sweep_results, task_order, 'double_task'
        )
        
        # Individual sequential performance plots for each topology
        for topology in ALL_TOPOLOGIES:
            plots[f'sequential_performance_{topology}'] = create_single_topology_sequential_plot(
                sweep_results, topology, task_order, 'double_task'
            )
        
        # Task-specific topology comparison plots (one for each tested task)
        for tested_task in ALL_TASKS:
            plots[f'topology_comparison_{tested_task}'] = create_task_specific_topology_comparison_plot(
                sweep_results, task_order, tested_task, 'double_task'
            )
        
        all_plots[f'double_task_{task_order}'] = plots
    
    # Generate plots for each triple-task order  
    for task_order in TRIPLE_TASK_ORDERS:
        plots = {}
        
        # Learning curves for each topology
        for topology in ALL_TOPOLOGIES:
            plots[f'learning_curves_{topology}'] = create_multi_phase_learning_curves(
                sweep_results, topology, task_order
            )
        
        # Transfer comparison
        plots['transfer_comparison'] = create_transfer_comparison_for_task_order(
            sweep_results, task_order
        )
        
        # Performance matrix
        plots['performance_matrix'] = create_performance_matrix_for_task_order(
            sweep_results, task_order
        )
        
        # Capacity scaling for each topology
        for topology in ALL_TOPOLOGIES:
            plots[f'capacity_scaling_{topology}'] = create_capacity_scaling_for_task_order(
                sweep_results, topology, task_order
            )
        
        # Sequential performance plot (all topologies comparison)
        plots['sequential_performance_comparison'] = create_sequential_task_performance_plot(
            sweep_results, task_order, 'triple_task'
        )
        
        # Individual sequential performance plots for each topology
        for topology in ALL_TOPOLOGIES:
            plots[f'sequential_performance_{topology}'] = create_single_topology_sequential_plot(
                sweep_results, topology, task_order, 'triple_task'
            )
        
        # Task-specific topology comparison plots (one for each tested task)
        for tested_task in ALL_TASKS:
            plots[f'topology_comparison_{tested_task}'] = create_task_specific_topology_comparison_plot(
                sweep_results, task_order, tested_task, 'triple_task'
            )
        
        all_plots[f'triple_task_{task_order}'] = plots
    
    # Generate task order effects plots for each topology
    for topology in ALL_TOPOLOGIES:
        all_plots[f'task_order_effects_{topology}_double'] = create_task_order_effects_plot(
            sweep_results, topology, 'double_task'
        )
        all_plots[f'task_order_effects_{topology}_triple'] = create_task_order_effects_plot(
            sweep_results, topology, 'triple_task'
        )
    
    return all_plots


def log_all_plots_to_wandb(wandb_run, all_plots: Dict):
    """
    Log all plots to wandb with clear task order distinction.
    
    Args:
        wandb_run: Wandb run object
        all_plots: Dictionary containing all plots
    """
    
    for plot_key, plot in all_plots.items():
        if isinstance(plot, dict):
            # Nested plots (e.g., all plots for a task order)
            for subplot_key, subplot in plot.items():
                wandb_run.log({
                    f"plots/{plot_key}/{subplot_key}": subplot
                })
        else:
            # Single plot
            wandb_run.log({
                f"plots/{plot_key}": plot
            })


def log_comprehensive_plots_for_run(
    wandb_run, 
    phase_results: Dict, 
    transfer_metrics: Dict, 
    topology_type: str, 
    task_sequence: str,
    sweep_results: Optional[Dict] = None
):
    """
    Log comprehensive plots for a single run.
    
    Args:
        wandb_run: Wandb run object
        phase_results: Results from this run
        transfer_metrics: Transfer learning metrics
        topology_type: Type of topology
        task_sequence: Task sequence string
        sweep_results: Optional sweep results for comparison
    """
    
    # 1. Multi-phase learning curves (MOST IMPORTANT)
    learning_curves = create_multi_phase_learning_curves(phase_results, topology_type, task_sequence)
    wandb_run.log({f"{topology_type}/{task_sequence}/plots/learning_progression": learning_curves})
    
    # 2. Transfer learning comparison (if we have data from other topologies)
    if sweep_results:
        transfer_plot = create_transfer_comparison_for_task_order(sweep_results, task_sequence)
        wandb_run.log({f"{topology_type}/{task_sequence}/plots/transfer_analysis": transfer_plot})
    
    # 3. Performance matrix (if sweep results available)
    if sweep_results:
        performance_matrix = create_performance_matrix_for_task_order(sweep_results, task_sequence)
        wandb_run.log({f"sweep_analysis/performance_matrix_{task_sequence}": performance_matrix})
    
    # 4. Capacity scaling (if capacity data available)
    if sweep_results:
        capacity_plot = create_capacity_scaling_for_task_order(sweep_results, topology_type, task_sequence)
        wandb_run.log({f"{topology_type}/{task_sequence}/plots/capacity_scaling": capacity_plot})
    
    # 5. Task order effects (if multiple task sequences available)
    if sweep_results:
        task_order_plot = create_task_order_effects_plot(sweep_results, topology_type)
        wandb_run.log({f"{topology_type}/plots/task_order_effects": task_order_plot})
    
    # 6. Sequential performance plot (CRUCIAL for continual learning analysis)
    if sweep_results:
        sequential_plot = create_sequential_task_performance_plot(sweep_results, task_sequence)
        wandb_run.log({f"{topology_type}/{task_sequence}/plots/sequential_performance": sequential_plot})
    
    # 7. Single topology sequential plot (clearer view for this topology)
    if sweep_results:
        single_topology_plot = create_single_topology_sequential_plot(sweep_results, topology_type, task_sequence)
        wandb_run.log({f"{topology_type}/{task_sequence}/plots/sequential_performance_single": single_topology_plot})
    
    # 8. Task-specific topology comparison plots (one for each tested task)
    if sweep_results:
        for tested_task in ALL_TASKS:
            task_comparison_plot = create_task_specific_topology_comparison_plot(
                sweep_results, task_sequence, tested_task
            )
            wandb_run.log({f"{topology_type}/{task_sequence}/plots/topology_comparison_{tested_task}": task_comparison_plot}) 