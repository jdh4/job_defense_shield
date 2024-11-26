from utils import seconds_to_slurm_time_format
from utils import add_dividers

def test_seconds_to_slurm_time_format():
    assert seconds_to_slurm_time_format(100) == "00:01:40"
    assert seconds_to_slurm_time_format(1000) == "00:16:40"
    assert seconds_to_slurm_time_format(10000) == "02:46:40"
    assert seconds_to_slurm_time_format(100000) == "1-03:46:40"

def test_add_dividers():
    df_str = "x y\n1 1\n2 2\n3 3"
    title = "points"
    actual = add_dividers(df_str, title, pre="", post="")
    expected = "points\n------\nx y\n------\n1 1\n2 2\n3 3\n"
    assert actual == expected
