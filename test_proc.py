import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
from resim import Reprocessor  # assuming the file is named reprocessor.py

# Fixtures for test data
@pytest.fixture
def create_test_files(tmp_path):
    """Create test input files for reprocessing"""
    # Create test camera data
    camera_data = pd.DataFrame({
        "Timestamp": [100000000.0, 100027700.0, 100055400.0],
        "FrameID": [100, 101, 102],
        "Speed": [60.0, 60.08, 60.16],
        "YawRate": [0.1, -0.2, 0.3],
        "Signal1": [0, 0, 0],
        "Signal2": [0.0, 0.0, 0.0]
    })
    
    # Create test sensor data (with slightly different timestamps)
    sensor_data = pd.DataFrame({
        "Timestamp": [99900000.0, 100025000.0, 100052000.0, 100080000.0],
        "Speed": [59.0, 59.56, 60.12, 60.68]
    })
    
    # Create input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Save files
    camera_path = input_dir / "f_cam_out.csv"
    sensor_path = input_dir / "sensor_out.csv"
    camera_data.to_csv(camera_path, index=False)
    sensor_data.to_csv(sensor_path, index=False)
    
    return camera_path, sensor_path

def test_try_load_data_success(create_test_files):
    """Test successful data loading"""
    camera_path, sensor_path = create_test_files
    processor = Reprocessor(camera_path, sensor_path)
    assert processor.try_load_data() is True
    assert isinstance(processor.sensor_data, pd.DataFrame)
    assert isinstance(processor.camera_data, pd.DataFrame)

def test_try_load_data_failure(tmp_path):
    """Test data loading failure with missing files"""
    processor = Reprocessor("missing_camera.csv", "missing_sensor.csv")
    assert processor.try_load_data() is False

def test_reprocess_data(create_test_files):
    """Test data reprocessing logic"""
    camera_path, sensor_path = create_test_files
    processor = Reprocessor(camera_path, sensor_path)
    result = processor.reprocess_data()
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"]
    
    # Verify speed averaging
    expected_speeds = [
        (60.0 + 59.0)/2,    # First camera timestamp matches last sensor timestamp before it
        (60.08 + 59.56)/2,  # Matches sensor timestamp 100025000.0
        (60.16 + 60.12)/2   # Matches sensor timestamp 100052000.0
    ]
    assert np.allclose(result["Speed"].values, expected_speeds)
    
    # Verify other columns are copied unchanged
    assert (result["FrameID"].values == [100, 101, 102]).all()
    assert np.allclose(result["YawRate"].values, [0.1, -0.2, 0.3])

def test_format_data(create_test_files):
    """Test data formatting"""
    camera_path, sensor_path = create_test_files
    processor = Reprocessor(camera_path, sensor_path)
    processor.reprocess_data()
    formatted = processor.format_data()
    
    # Check integer columns
    assert formatted["FrameID"].dtype == 'int64'
    assert formatted["Signal1"].dtype == 'int64'
    
    # Check float formatting
    assert all(formatted["Timestamp"].str.match(r'^\d+\.\d{6}$'))
    assert all(formatted["Speed"].str.match(r'^\d+\.\d{2}$'))
    assert all(formatted["YawRate"].str.match(r'^-?\d+\.\d{2}$'))
    assert all(formatted["Signal2"].str.match(r'^-?\d+\.\d{2}$'))

def test_to_csv(create_test_files, tmp_path):
    """Test CSV file generation"""
    camera_path, sensor_path = create_test_files
    output_dir = tmp_path / "output"
    
    processor = Reprocessor(camera_path, sensor_path)
    processor.to_csv(output_dir)
    
    # Check file was created
    output_file = output_dir / "resim_out.csv"
    assert output_file.exists()
    
    # Verify file content
    df = pd.read_csv(output_file)
    assert len(df) == 3
    assert list(df.columns) == ["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"]

def test_to_csv_with_missing_inputs(tmp_path):
    """Test CSV generation with missing input files"""
    output_dir = tmp_path / "output"
    
    processor = Reprocessor("missing_camera.csv", "missing_sensor.csv")
    processor.to_csv(output_dir)
    
    # Verify no output file was created
    output_file = output_dir / "resim_out.csv"
    assert not output_file.exists()

def test_timestamp_matching_logic(create_test_files):
    """Test the sensor timestamp matching logic"""
    camera_path, sensor_path = create_test_files
    processor = Reprocessor(camera_path, sensor_path)
    processor.try_load_data()
    
    # Test timestamp matching
    test_cases = [
        (99950000.0, 0),   # Should match first sensor point
        (100026000.0, 1),  # Should match second sensor point
        (100060000.0, 2),  # Should match third sensor point
        (100090000.0, 3)   # Should match last sensor point
    ]
    
    for timestamp, expected_idx in test_cases:
        sensor_index = 0
        while (sensor_index + 1 < len(processor.sensor_data) and 
               processor.sensor_data.loc[sensor_index + 1, "Timestamp"] <= timestamp):
            sensor_index += 1
        assert sensor_index == expected_idx

def test_edge_case_empty_files(tmp_path):
    """Test behavior with empty input files"""
    empty_camera = tmp_path / "empty_cam.csv"
    empty_sensor = tmp_path / "empty_sensor.csv"
    
    pd.DataFrame().to_csv(empty_camera, index=False)
    pd.DataFrame().to_csv(empty_sensor, index=False)
    
    processor = Reprocessor(empty_camera, empty_sensor)
    assert processor.reprocess_data() is None

def test_edge_case_empty_df(tmp_path):
    """Test behavior with empty input files"""
    empty_camera = tmp_path / "head_empty_cam.csv"
    empty_sensor = tmp_path / "head_empty_sensor.csv"
    
    pd.DataFrame(columns=["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"]).to_csv(empty_camera, index=False)
    pd.DataFrame(columns=["Timestamp", "Speed"]).to_csv(empty_sensor, index=False)
    
    processor = Reprocessor(empty_camera, empty_sensor)
    assert processor.reprocess_data().empty == True

def test_column_preservation(create_test_files):
    """Verify all non-speed columns are preserved exactly"""
    camera_path, sensor_path = create_test_files
    processor = Reprocessor(camera_path, sensor_path)
    result = processor.reprocess_data()
    
    # Load original camera data
    camera_data = pd.read_csv(camera_path)
    
    # Check all non-speed columns match exactly
    for col in ["Timestamp", "FrameID", "YawRate", "Signal1", "Signal2"]:
        if col in camera_data.columns:
            assert (result[col] == camera_data[col]).all()