from f_cam import FrontCameraSimulation
from sensor import SensorSimulation

import pandas as pd
import numpy as np
import pytest

import os
from pathlib import Path


@pytest.fixture
def simulation():
    """Fixture providing a default simulation instance"""
    return FrontCameraSimulation()

@pytest.fixture
def generated_data(simulation):
    """Fixture with generated data"""
    simulation.generate_data()
    return simulation.data

def test_generate_data_structure(generated_data):
    """Test the structure of generated data"""
    assert isinstance(generated_data, pd.DataFrame)
    assert list(generated_data.columns) == ["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"]
    assert len(generated_data) == 2000
    assert all(generated_data["FrameID"].astype(int) == np.arange(100, 2100))

def test_timestamp_behavior(generated_data):
    """Test timestamp starts at 100s and increments with proper variation"""
    timestamps = generated_data["Timestamp"].astype(float).values
    assert pytest.approx(timestamps[0], abs=1e-6) == 100000000.0
    
    increments = np.diff(timestamps)
    expected_increment = 27.7 * 1000  # ms to μs
    variation = 0.05 * 1000  # ms to μs
    
    # Check increments are within specified range
    assert np.all(increments >= expected_increment - variation)
    assert np.all(increments <= expected_increment + variation)

def test_speed_behavior(generated_data):
    """Test speed starts at 60 and increases until 120, then stays around 120"""
    speeds = generated_data["Speed"].astype(float).values
    assert pytest.approx(speeds[0], abs=1e-2) == 60.0
    
    # Check increasing phase
    increasing_phase = speeds[speeds < 119.95]
    if len(increasing_phase) > 1:
        diffs = np.diff(increasing_phase)
        assert np.allclose(diffs, 0.08, atol=1e-2)
    
    # Check stabilized phase
    stabilized_phase = speeds[speeds >= 120]
    assert np.all(stabilized_phase <= 120 + 0.05)

def test_yaw_rate_behavior(generated_data):
    """Test yaw rate stays within ±1 deg/sec"""
    yaw_rates = generated_data["YawRate"].astype(float).values
    assert yaw_rates[0] == 0.0
    assert np.all(yaw_rates >= -1.0)
    assert np.all(yaw_rates <= 1.0)

def test_signal1_behavior(generated_data):
    """Test Signal1 starts at 0, changes after frame 200, and remains constant"""
    signal1 = generated_data["Signal1"].values
    frame_ids = generated_data["FrameID"].values
    
    # First 200 frames should be 0
    assert np.all(signal1[frame_ids <= 200] == 0)
    
    # After frame 200, should be between 1-15 and constant
    post_200 = signal1[frame_ids > 200]
    if len(post_200) > 0:
        assert post_200[0] in range(1, 16)
        assert np.all(post_200 == post_200[0])

def test_signal2_behavior(generated_data):
    """Test Signal2 is 0 when Signal1 <5, otherwise 80±10"""
    signal1 = generated_data["Signal1"].values
    signal2 = generated_data["Signal2"].astype(float).values
    
    # Cases where Signal1 < 5
    low_signal_mask = (signal1 < 5) & (signal1 > 0)
    assert np.all(signal2[low_signal_mask] == 0)
    
    # Cases where Signal1 >= 5
    high_signal_mask = signal1 >= 5
    if np.any(high_signal_mask):
        assert np.all(signal2[high_signal_mask] >= 70)
        assert np.all(signal2[high_signal_mask] <= 90)

def test_format_data(simulation, generated_data):
    """Test data formatting"""
    simulation.data = generated_data.copy()
    formatted = simulation.format_data()
    
    # Check integer columns
    assert formatted["FrameID"].dtype == 'int64'
    assert formatted["Signal1"].dtype == 'int64'
    
    # Check float formatting
    assert all(formatted["Timestamp"].str.match(r'^\d+\.\d{6}$'))
    assert all(formatted["Speed"].str.match(r'^\d+\.\d{2}$'))
    assert all(formatted["YawRate"].str.match(r'^-?\d+\.\d{2}$'))
    assert all(formatted["Signal2"].str.match(r'^-?\d+\.\d{2}$'))

def test_to_csv(tmp_path, simulation):
    """Test CSV file generation"""
    output_dir = tmp_path / "output"
    simulation.to_csv(str(output_dir))
    
    # Check file was created
    output_file = output_dir / "f_cam_out.csv"
    assert output_file.exists()
    
    # Verify file content
    df = pd.read_csv(output_file)
    assert len(df) == 2000
    assert list(df.columns) == ["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"]