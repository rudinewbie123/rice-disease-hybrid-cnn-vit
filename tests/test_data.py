import pandas as pd
import pytest
from rice_disease.data import RiceLeafDataset


def test_manifest_requires_image_path_and_label(tmp_path):
    manifest=tmp_path/"bad.csv"; pd.DataFrame({"file":["x.jpg"],"class":["a"]}).to_csv(manifest,index=False)
    with pytest.raises(ValueError,match="columns"):
        RiceLeafDataset(manifest,tmp_path,["a"])
