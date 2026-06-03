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


@pytest.mark.parametrize("index", [None, "ts"])
def test_basic(in_memory_store_factory, index):
    lib = in_memory_store_factory()
    sym = "test_basic"
    df_0 = pd.DataFrame(
        {"col": np.arange(20)}, index=None if index is None else pd.date_range("2026-01-01", periods=20)
    )
    lib.write(sym, df_0)
    df_1 = pd.DataFrame(
        {"col": np.arange(20, 30)}, index=None if index is None else pd.date_range("2026-01-21", periods=10)
    )
    lib.append(sym, df_1, compact_data_inline=True)
    expected = pd.concat([df_0, df_1])
    if index is None:
        expected = expected.reset_index(drop=True)
    received = lib.read(sym).data
    assert_frame_equal(expected, received)
    assert len(lib.read_index(sym)) == 1


@pytest.mark.parametrize("index", [None, "ts"])
def test_defrag_whole_symbol(in_memory_store_factory, index):
    lib = in_memory_store_factory(segment_row_size=10)
    sym = "test_defrag_whole_symbol"
    df = pd.DataFrame({"col": np.arange(20)}, index=None if index is None else pd.date_range("2026-01-01", periods=20))
    lib.write(sym, df[:5])
    lib.append(sym, df[5:10])
    lib.append(sym, df[10:15])
    lib.append(sym, df[15:], compact_data_inline=True)
    received = lib.read(sym).data
    assert_frame_equal(df, received)
    assert len(lib.read_index(sym)) == 2


@pytest.mark.parametrize("index", [None, "ts"])
def test_basic_dynamic_schema(in_memory_store_factory, index):
    lib = in_memory_store_factory(dynamic_schema=True)
    sym = "test_basic_dynamic_schema"
    df_0 = pd.DataFrame(
        {
            "col_0": np.arange(20, dtype=np.float64),
            "col_1": np.arange(20, 40, dtype=np.float64),
            "col_2": np.arange(40, 60, dtype=np.float64),
        },
        index=None if index is None else pd.date_range("2026-01-01", periods=20),
    )
    lib.write(sym, df_0)
    df_1 = pd.DataFrame(
        {
            "col_3": np.arange(100, 110, dtype=np.float64),
            "col_2": np.arange(60, 70, dtype=np.float64),
            "col_1": np.arange(40, 50, dtype=np.float64),
        },
        index=None if index is None else pd.date_range("2026-01-21", periods=10),
    )
    lib.append(sym, df_1, compact_data_inline=True)
    expected = pd.concat([df_0, df_1])
    if index is None:
        expected = expected.reset_index(drop=True)
    received = lib.read(sym).data
    assert_frame_equal(expected, received)
    assert len(lib.read_index(sym)) == 1


@pytest.mark.parametrize("index", [None, "ts"])
def test_column_slicing(in_memory_store_factory, index):
    lib = in_memory_store_factory(column_group_size=2)
    sym = "test_column_slicing"
    df_0 = pd.DataFrame(
        {f"col_{idx}": np.arange(20) for idx in range(5)},
        index=None if index is None else pd.date_range("2026-01-01", periods=20),
    )
    lib.write(sym, df_0)
    df_1 = pd.DataFrame(
        {f"col_{idx}": np.arange(20, 30) for idx in range(5)},
        index=None if index is None else pd.date_range("2026-01-21", periods=10),
    )
    lib.append(sym, df_1, compact_data_inline=True)
    expected = pd.concat([df_0, df_1])
    if index is None:
        expected = expected.reset_index(drop=True)
    received = lib.read(sym).data
    assert_frame_equal(expected, received)
    assert len(lib.read_index(sym)) == 3


def test_schema_mismatch_static(in_memory_store_factory):
    lib = in_memory_store_factory()
    sym = "test_schema_mismatch_static"
    df_0 = pd.DataFrame({"col_0": [0]})
    lib.write(sym, df_0)
    # Different column sets
    df_1 = pd.DataFrame({"col_1": [0]})
    with pytest.raises(Exception) as e_without_arg:
        lib.append(sym, df_1)
    with pytest.raises(Exception) as e_with_arg:
        lib.append(sym, df_1, compact_data_inline=True)
    assert e_with_arg.type == e_without_arg.type
    assert e_with_arg.typename == e_without_arg.typename
    assert e_with_arg.value.args[0] == e_without_arg.value.args[0]
    # Different column type
    df_1 = pd.DataFrame({"col_0": ["hello"]})
    with pytest.raises(Exception) as e_without_arg:
        lib.append(sym, df_1)
    with pytest.raises(Exception) as e_with_arg:
        lib.append(sym, df_1, compact_data_inline=True)
    assert e_with_arg.type == e_without_arg.type
    assert e_with_arg.typename == e_without_arg.typename
    assert e_with_arg.value.args[0] == e_without_arg.value.args[0]


def test_schema_mismatch_dynamic(in_memory_store_factory):
    lib = in_memory_store_factory(dynamic_schema=True)
    sym = "test_schema_mismatch_dynamic"
    df_0 = pd.DataFrame({"col_0": [0]})
    lib.write(sym, df_0)
    df_1 = pd.DataFrame({"col_0": ["hello"]})
    with pytest.raises(Exception) as e_without_arg:
        lib.append(sym, df_1)
    with pytest.raises(Exception) as e_with_arg:
        lib.append(sym, df_1, compact_data_inline=True)
    assert e_with_arg.type == e_without_arg.type
    assert e_with_arg.typename == e_without_arg.typename
    assert e_with_arg.value.args[0] == e_without_arg.value.args[0]


# TODO: Tests
# - with timeseries index
# - appending an empty df with compact_data_inline=True defrags existing data
# - appending with row slices NOT attached to the new frame need compacting
# - with column slicing
# - with dynamic schema
# - with data that needs writing (and slicing) after what gets combined with existing data
# - with fortran-style data
# - That static/dynamic schema reject appends with mismatching/incompatible schemas
# - Newly provided metadata is persisted correctly, and old metadata not maintained to match normal append behaviour
# - Hypothesis
