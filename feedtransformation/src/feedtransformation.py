import polars as pl
from typing import List, Union
from polars import ColumnNotFoundError
import warnings


def rename_cols(feed: pl.DataFrame, rename_dict: dict) -> pl.DataFrame:
    """
    Rename columns of the feed

    Parameters:
        rename_dict (dict): A dictionary with the columns to rename

    Returns:
        self
    """
    if not isinstance(rename_dict, dict):
        raise TypeError("rename_dict must be a dictionary")

    return feed.rename(rename_dict)


def format_cols(feed: pl.DataFrame) -> pl.DataFrame:
    """
    Format columns of the feed

    Parameters:
        None

    Returns:
        self
    """
    return feed.select(
        pl.all().map_alias(lambda col: col.strip().lower().replace(" ", "_"))
    )


def filter_products(feed: pl.DataFrame, col: str, filter_dict: dict):
    pass


def create_metadata(
    feed: pl.DataFrame, cols: list, meta_name: str = "metadata"
) -> pl.DataFrame:
    """
    Create metadata column as a struct of the columns in cols

    Parameters:
        cols (list): A list of columns to be included in the metadata column
        meta_name (str): The name of the metadata column

    Returns:
        self
    """
    _validate_existing_columns(feed, cols)

    _overwrite_metadata(feed, meta_name)
    return feed.with_columns(pl.struct(cols).alias(meta_name))


def all_combinations_metadata(
    feed: pl.DataFrame,
    cols: list,
    on_col: Union[str, List[str]],
    meta_name: str = "metadata",
) -> pl.DataFrame:
    """
    Create metadata column as a struct of the columns in cols for all combinations of on_col

    Parameters:
        cols (list): A list of columns to be included in the metadata column
        on_col (str or list): The column(s) to group by
        col_name (str): The name of the metadata column

    Returns:
        self
    """
    _validate_existing_columns(feed, cols)

    _overwrite_metadata(feed, meta_name)

    current_metadata = meta_name
    comb_metadata = (
        feed.with_columns(pl.struct(cols).alias(meta_name))
        .select(pl.col(meta_name, on_col))
        .group_by(on_col)
        .agg(pl.col(meta_name))
    )
    return feed.join(comb_metadata, on=on_col, how="left")


def group_metadata(
    feed: pl.DataFrame,
    group_cols: Union[List[str], str],
    metadata: str = "metadata",
    order: bool = False,
) -> pl.DataFrame:
    """
    Group metadata column by group_cols and keep the first value of the other columns

    Parameters:
        group_cols (list or str): A list of columns to group by (or a single column name)
        order (bool): Whether to maintain the order of the feed

    Returns:
        self
    """
    try:
        feed.select(pl.col(group_cols))
    except ColumnNotFoundError:
        raise ColumnNotFoundError("One or more columns in cols not found in feed")

    cols = [col for col in feed.columns if col not in group_cols and col != metadata]
    return feed.group_by(group_cols, maintain_order=order).agg(
        pl.col(cols).first(), pl.col(metadata)
    )


def rename_column_value(
    feed: pl.DataFrame, col: str, old: str, new: str, regex=False, ow: str = None
) -> pl.DataFrame:
    """
    Rename column value from old to new in column col

    Parameters:
        col (str): The name of the column
        old (str): The old value
        new (str): The new value
        regex (bool): Whether to use regex
        ow (str): The value to replace with if old is not found

    Returns:
        self
    """
    _validate_existing_columns(feed, col)

    if ow is None:
        ow = old
        warnings.warn(f"ow not specified. Using old value {old} as ow")

    if regex:
        feed = feed.with_columns(pl.col(col).str.replace(rf"{old}", new))
    else:
        feed = feed.with_columns(
            pl.when(pl.col(col) == old)
            .then(pl.lit(new))
            .otherwise(pl.lit(ow))
            .alias(col)
        )
    return feed


def export_json(
    feed: pl.DataFrame, path: str, finalize: bool = True
) -> pl.DataFrame | None:
    """
    Export feed to json

    Parameters:
        path (str): The path to export the feed
        finalize (bool): Whether to finalize the feed (i.e. return None)
    Returns:
        self
    """
    feed.write_json(path, row_oriented=True, pretty=True)

    if finalize:
        return None

    return feed


def export_csv(
    feed: pl.DataFrame, path: str, finalize: bool = False
) -> pl.DataFrame | None:
    """
    Export feed to csv

    Parameters:
        path (str): The path to export the feed
        finalize (bool): Whether to finalize the feed (i.e. return None)

    Returns:
        self
    """
    feed.write_csv(path)
    if finalize:
        return None
    return feed


def _validate_existing_columns(df: pl.DataFrame, cols: list[str] | str) -> None:
    if isinstance(cols, list):
        try:
            df.select(pl.col(cols))
        except ColumnNotFoundError:
            raise ColumnNotFoundError("One or more columns in cols not found in feed.")
    else:
        try:
            df.select(pl.col(cols))
        except ColumnNotFoundError:
            raise ColumnNotFoundError(f"Column {cols} not found in feed")


def _overwrite_metadata(df: pl.DataFrame, metadata: str) -> None:
    if metadata in df.columns:
        warnings.warn(
            f"Column {metadata} already exists in feed. It will be overwritten"
        )
