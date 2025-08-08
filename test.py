import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from f_cam import FrontCameraSimulation
from sensor import SensorSimulation

# Common fixtures and parameters
@pytest.fixture(params=[FrontCameraSimulation, SensorSimulation])
def simulation_class(request):
    return request.param

@pytest.fixture
def simulation(simulation_class):
    if simulation_class == FrontCameraSimulation:
        return simulation_class(from_id=100, frames=2000)
    return simulation_class(from_s=100, to_s=160)

@pytest.fixture
def generated_data(simulation):
    simulation.generate_data()
    return simulation.data

def test_data_structure(generated_data, simulation):
    """Test the structure of generated data"""
    assert isinstance(generated_data, pd.DataFrame)
    
    if isinstance(simulation, FrontCameraSimulation):
        expected_columns = ["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"]
    else:
        expected_columns = ["Timestamp", "Speed"]
        
    assert list(generated_data.columns) == expected_columns

def test_timestamp_behavior(generated_data, simulation):
    """Test timestamp behavior"""
    timestamps = generated_data["Timestamp"].astype(float).values
    
    if isinstance(simulation, FrontCameraSimulation):
        # Camera specific timestamp tests
        assert pytest.approx(timestamps[0], abs=1e-6) == 100 * 10**6
        increments = np.diff(timestamps)
        expected_increment = 27.7 * 1000
        variation = 0.05 * 1000
    else:
        # Sensor specific timestamp tests
        assert pytest.approx(timestamps[0], abs=1e-6) == 100 * 10**6
        assert timestamps[-1] <= 160 * 10**6
        increments = np.diff(timestamps)
        expected_increment = 200 * 1000
        variation = 10 * 1000
    
    # Common timestamp increment checks
    assert np.all(increments >= expected_increment - variation)
    assert np.all(increments <= expected_increment + variation)

def test_speed_behavior(generated_data, simulation):
    """Test speed behavior"""
    speeds = generated_data["Speed"].astype(float).values
    assert pytest.approx(speeds[0], abs=1e-2) == 60.0
    
    if isinstance(simulation, FrontCameraSimulation):
        # Camera specific speed tests
        increasing_phase = speeds[speeds < 119.95]
        if len(increasing_phase) > 1:
            assert np.allclose(np.diff(increasing_phase), 0.08, atol=1e-2)
        
        stabilized_phase = speeds[speeds >= 120]
        assert np.all(stabilized_phase <= 120 + 0.05)
    else:      
        increasing_phase = speeds[speeds < 119.9]
        if len(increasing_phase) > 1:
            assert np.allclose(np.diff(increasing_phase), 0.56, atol=1e-2)
        
        stabilized_phase = speeds[speeds >= 120]
        assert np.all(stabilized_phase <= 120 + 0.1)

def test_format_data(generated_data, simulation):
    """Test data formatting"""
    simulation.data = generated_data.copy()
    formatted = simulation.format_data()
    
    # Common format checks
    assert all(formatted["Timestamp"].str.match(r'^\d+\.\d{6}$'))
    assert all(formatted["Speed"].str.match(r'^\d+\.\d{2}$'))
    
    # Camera specific format checks
    if isinstance(simulation, FrontCameraSimulation):
        assert formatted["FrameID"].dtype == 'int64'
        assert formatted["Signal1"].dtype == 'int64'
        assert all(formatted["YawRate"].str.match(r'^-?\d+\.\d{2}$'))
        assert all(formatted["Signal2"].str.match(r'^-?\d+\.\d{2}$'))

def test_to_csv(tmp_path, simulation):
    """Test CSV file generation"""
    output_dir = tmp_path / "output"
    simulation.to_csv(str(output_dir))
    
    # Check file was created with correct name
    if isinstance(simulation, FrontCameraSimulation):
        output_file = output_dir / "f_cam_out.csv"
    else:
        output_file = output_dir / "sensor_out.csv"
    
    assert output_file.exists()
    
    # Verify file content
    df = pd.read_csv(output_file)
    assert len(df) > 0

# Camera-specific tests
@pytest.mark.parametrize("simulation_class", [FrontCameraSimulation], indirect=True)
def test_frame_id_sequence(generated_data):
    """Test FrameID starts at 100 and increments by 1 (Camera only)"""
    frame_ids = generated_data["FrameID"].values
    assert frame_ids[0] == 100
    assert np.all(np.diff(frame_ids) == 1)
    assert frame_ids[-1] == 100 + 2000 - 1
    assert len(frame_ids) == 2000

@pytest.mark.parametrize("simulation_class", [FrontCameraSimulation], indirect=True)
def test_camera_specific_columns(generated_data):
    """Test camera-specific columns behavior"""
    # Test YawRate range
    yaw_rates = generated_data["YawRate"].astype(float).values
    assert yaw_rates[0] == 0.0
    assert np.all(yaw_rates >= -1.0)
    assert np.all(yaw_rates <= 1.0)
    
    # Test Signal1 behavior
    signal1 = generated_data["Signal1"].values
    frame_ids = generated_data["FrameID"].values
    assert np.all(signal1[frame_ids <= 200] == 0)
    post_200 = signal1[frame_ids > 200]
    if len(post_200) > 0:
        assert post_200[0] in range(1, 16)
        assert np.all(post_200 == post_200[0])
    
    # Test Signal2 behavior
    signal2 = generated_data["Signal2"].astype(float).values
    low_signal_mask = (signal1 < 5) & (signal1 > 0)
    assert np.all(signal2[low_signal_mask] == 0)
    high_signal_mask = signal1 >= 5
    if np.any(high_signal_mask):
        assert np.all(signal2[high_signal_mask] >= 70)
        assert np.all(signal2[high_signal_mask] <= 90)

# Sensor-specific tests
@pytest.mark.parametrize("simulation_class", [SensorSimulation], indirect=True)
def test_sensor_timestamp_cutoff(generated_data):
    """Test that data generation stops at 160s (Sensor only)"""
    timestamps = generated_data["Timestamp"].astype(float).values
    assert timestamps[-1] <= 160 * 10**6
    assert timestamps[-1] + (200 + 10) * 1000 > 160 * 10**6