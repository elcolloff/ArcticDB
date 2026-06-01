"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""

import numpy as np
import pandas as pd
import pytest

from arcticdb.util.test import assert_frame_equal

pytestmark = pytest.mark.pipeline


def test_basic(in_memory_store_factory):
    lib = in_memory_store_factory()
    sym = "test_basic"
    df_0 = pd.DataFrame({"col": np.arange(20)})
    lib.write(sym, df_0)
    df_1 = pd.DataFrame({"col": np.arange(20, 30)})
    lib.append(sym, df_1, compact_data_inline=True)
    expected = pd.concat([df_0, df_1]).reset_index(drop=True)
    received = lib.read(sym).data
    assert_frame_equal(expected, received)
    assert len(lib.read_index(sym)) == 1


# TODO: Tests
#  - appending an empty df with compact_data_inline=True defrags existing data
