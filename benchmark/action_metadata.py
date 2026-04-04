"""Shared action metadata for the Mario benchmark."""

ACTION_NAMES = {
    0: "inspect_schema",
    1: "view_error_log",
    2: "sample_data",
    3: "fill_mean",
    4: "fill_median",
    5: "fill_forward",
    6: "drop_nulls",
    7: "cast_to_int",
    8: "cast_to_float",
    9: "cast_to_string",
    10: "remove_duplicates",
    11: "drop_outliers",
    12: "rename_column",
    13: "reorder_columns",
    14: "validate_schema",
    15: "commit_changes",
    16: "scale_resources_up",
    17: "scale_resources_down",
    18: "prioritize_incremental_batch",
    19: "refresh_pipeline_outputs",
}

PARAMETER_ACTIONS = {3, 4, 5, 6, 7, 8, 9, 11, 12}

