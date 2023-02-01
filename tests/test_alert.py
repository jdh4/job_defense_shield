import pytest
import pandas as pd
import sys
sys.path.append("../")
from job_defense_shield import Alert

@pytest.fixture(autouse=True)
def setUp():
    print()
    print("setUp")
    print("tearDown")

def test_answer():
    df = pd.DataFrame({"netid":["jdh4", "jdh4", "jdh4", "miz"],
                       "partition":["gpu", "mig", "gpu", "mig"],
                       "use":range(4)})
    A = Alert(df, 13, "N/A", "N/A", "Subject")
    assert A.is_work_day("2023-01-01") == False
    assert A.is_work_day("2023-07-04") == False
    assert A.is_work_day("2023-07-07") == True
    assert A.is_work_day("2023-11-14") == True
