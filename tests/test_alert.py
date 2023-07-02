import pytest
import pandas as pd
import sys
sys.path.append("../")
sys.path.append(".")
from utils import seconds_to_slurm_time_format

@pytest.fixture(autouse=True)
def setUp():
    print()
    print("setUp")
    print("tearDown")

def test_seconds_to_slurm_time_format():
    assert seconds_to_slurm_time_format(100) == "00:01:40"
    assert seconds_to_slurm_time_format(1000) == "00:16:40"
    assert seconds_to_slurm_time_format(10000) == "02:46:40"
    assert seconds_to_slurm_time_format(100000) == "1-03:46:40"
