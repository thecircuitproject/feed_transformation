import pandas as pd
import polars as pl
import numpy as np
from typing import List, Union
from polars import ColumnNotFoundError
import warnings

class FeedTransformation:
    """
    This class is used to transform a feed.

    To visualize the dataframe, call the feed attribute after each method.
    
    Attributes:
        feed (pl.DataFrame): A polars DataFrame
        current_metadata (str): The name of the current metadata column

    Methods:
        rename_cols(rename_dict:dict): Rename columns of the feed
        format_cols(): Format columns of the feed
        filter_products(col:str, filter_dict:dict): Filter products of the feed
        create_metadata(cols:list,col_name:str='metadata'): Create metadata column
        all_combinations_metadata(cols:list, on_col:Union[str, List[str]], col_name:str='metadata'): Create metadata column for all combinations of on_col
        group_metadata(group_cols:list, order:bool=False): Group metadata column
        rename_column_value(col:str, old:str, new:str, regex=False, ow:str=None): Rename column value
        export_json(path:str): Export feed to json
        export_csv(path:str): Export feed to csv
    """
    def __init__(self, feed:pl.DataFrame):
        """
        Parameters:
            feed (pl.DataFrame): A polars DataFrame

        Returns:
            None
        """

        if not isinstance(feed, pl.DataFrame):
            raise TypeError("Feed must be a polars DataFrame")

        self.feed = feed
        self.current_metadata = None
    def rename_cols(self, rename_dict:dict):
        """
        Rename columns of the feed

        Parameters:
            rename_dict (dict): A dictionary with the columns to rename

        Returns:
            self
        """
        if not isinstance(rename_dict, dict):
            raise TypeError("rename_dict must be a dictionary")
        
        self.feed = self.feed.rename(rename_dict)
        return self
    def format_cols(self):
        """
        Format columns of the feed

        Parameters:
            None

        Returns:
            self
        """
        self.feed = self.feed.select(
                            pl.all().map_alias(lambda col: col.strip().lower().replace(" ","_"))
                        )
        return self
    def filter_products(self, col:str, filter_dict:dict):
        pass
    def create_metadata(self, cols:list,meta_name:str='metadata'):
        """
        Create metadata column as a struct of the columns in cols

        Parameters:
            cols (list): A list of columns to be included in the metadata column
            meta_name (str): The name of the metadata column

        Returns:
            self
        """
        try:
            self.feed.select(pl.col(cols))
        except ColumnNotFoundError:
            raise ColumnNotFoundError("One or more columns in cols not found in feed")
        
        if meta_name in self.feed.columns:
            warnings.warn(f"Column {meta_name} already exists in feed. It will be overwritten")

        if self.current_metadata:
            drop_old_metadata = self.current_metadata
        else:
            drop_old_metadata = None
        
        self.current_metadata = meta_name
        self.feed = (self.feed
                     .with_columns(pl.struct(cols).alias(meta_name))
                    )
        if drop_old_metadata:
            self.feed = self.feed.drop(drop_old_metadata)
        return self
    
    def all_combinations_metadata(self, cols:list, on_col:Union[str, List[str]], meta_name:str='metadata'):
        """
        Create metadata column as a struct of the columns in cols for all combinations of on_col

        Parameters:
            cols (list): A list of columns to be included in the metadata column
            on_col (str or list): The column(s) to group by
            col_name (str): The name of the metadata column

        Returns:
            self
        """
        try:
            self.feed.select(pl.col(cols))
        except ColumnNotFoundError:
            raise ColumnNotFoundError("One or more columns in cols not found in feed")
        
        if meta_name in self.feed.columns:
            warnings.warn(f"Column {meta_name} already exists in feed. It will be overwritten")

        if self.current_metadata:
            drop_old_metadata = self.current_metadata
        else:
            drop_old_metadata = None

        self.current_metadata = meta_name
        comb_metadata = (self.feed
                        .with_columns(pl.struct(cols).alias(meta_name))
                        .select(pl.col(meta_name, on_col))
                        .groupby(on_col)
                        .agg(pl.col(meta_name))
                        )
        self.feed = (self.feed
                     .join(comb_metadata, on=on_col, how='left')
                    )
        
        if drop_old_metadata:
            self.feed = self.feed.drop(drop_old_metadata)
        return self
        

    def group_metadata(self,group_cols:Union[List[str], str], order:bool=False):
        """
        Group metadata column by group_cols and keep the first value of the other columns

        Parameters:
            group_cols (list or str): A list of columns to group by (or a single column name)
            order (bool): Whether to maintain the order of the feed

        Returns:
            self
        """
        try:
            self.feed.select(pl.col(group_cols))
        except ColumnNotFoundError:
            raise ColumnNotFoundError("One or more columns in cols not found in feed")
        
        if self.current_metadata is None:
            raise ValueError("Metadata column not found. Use create_metadata() first")


        cols = [col for col in self.feed.columns if col not in group_cols and col != self.current_metadata]
        self.feed = (self.feed
                     .groupby(group_cols, maintain_order=order)
                     .agg(pl.col(cols).first(), pl.col(self.current_metadata))
                    )
        return self
    def rename_column_value(self, col:str, old:str, new:str, regex=False, ow:str=None):
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
        try:
            self.feed.select(pl.col(col))
        except ColumnNotFoundError:
            raise ColumnNotFoundError(f"Column {col} not found in feed")
        
        if ow is None:
            ow = old
            warnings.warn(f"ow not specified. Using old value {old} as ow")

        if regex:
            self.feed = (self.feed
                         .with_columns(
                            pl.col(col).str.replace(rf"{old}", new))
                        )
        else:
            self.feed = (self.feed
                         .with_columns(
                            pl.when(pl.col(col) == old)
                                .then(pl.lit(new))
                                .otherwise(pl.lit(ow)).alias(col))
                        )
        return self
    
    def export_json(self, path:str, finalize:bool=False):
        """
        Export feed to json

        Parameters:
            path (str): The path to export the feed
            finalize (bool): Whether to finalize the feed (i.e. return None)
        Returns:
            self
        """
        self.feed.write_json(path, row_oriented=True)

        if finalize:
            return None
        
        return self
    
    def export_csv(self, path:str, finalize:bool=False):
        """
        Export feed to csv

        Parameters:
            path (str): The path to export the feed
            finalize (bool): Whether to finalize the feed (i.e. return None)

        Returns:
            self
        """
        self.feed.write_csv(path)
        if finalize:
            return None
        return self