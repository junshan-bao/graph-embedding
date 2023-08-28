import abc

import pandas as pd
import os
import sys
import math
import pathlib
import time
import json
import pickle
import datetime
import logging

from blocks.filesystem import GCSFileSystem

from block_cloud_auth.providers import GCPSecretProvider
from pysnowflake.session import Session


logger = logging.getLogger('ge')


class _DataBaseConnector:
    @abc.abstractmethod
    def execute(self, query, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, query, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def upload(self, df, table_name, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_table(self, table_name, n, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def download_table(self, table_name, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def count_table(self, table_name, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def create_table(self, table_name, table_query, *args, **kwargs):
        raise NotImplementedError


class _CloudStorageConnector:
    @staticmethod
    @abc.abstractmethod
    def ls(path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def copy(lpath, rpath, recursive=False):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def remove(path, recursive=False):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def isdir(path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def save_pickle(data, path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load_pickle(path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def save_text(data, path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load_text(path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def save_binary(data, path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load_binary(path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def save_json(data, path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load_json(path):
        raise NotImplementedError


class SnowflakeConnector(_DataBaseConnector):
    def __init__(self):
        super().__init__()
        self.secret_id = "ap_r_p_frdrisk"
        self.project = "ds-cash-production"
        self.auth = GCPSecretProvider(secret_id=self.secret_id, project=self.project)

        self.logger = logger

    @staticmethod
    def oneline_query(query, n=100):
        return ' '.join(query.split())[:n]

    @staticmethod
    def trans_sec(sec):
        return str(datetime.timedelta(seconds=sec))

    def execute(self, query, ignore_error=False, mode='query', verbose=True):
        queries = [q.strip() for q in query.split(';') if q.strip() != '']
        for i, query in enumerate(queries):
            start_time = time.time()
            success = False

            with Session(auth=self.auth) as sess:
                try:
                    res = sess.execute(query, mode=mode)
                    res = res.fetchall()
                    success = True
                except Exception as e:
                    self.logger.error(e)

            if verbose:
                self.logger.info(
                    f"{'Finish' if success else 'Failed'} [{i + 1}|{len(queries)}] "
                    f"(runtime: {self.trans_sec(time.time() - start_time)}) Query: {self.oneline_query(query)} ..."
                )
            if not success and not ignore_error:
                return False
        return True

    # def executes(self, query, return_result=False, mode='query'):
    #     with Session(auth=self.auth) as sess:
    #         res = sess.parameterized_execute(query=query, mode=mode)
    #         if return_result:
    #             return res.fetchall()
    #         return res

    def run(self, query, mode='query', verbose=True):
        with Session(auth=self.auth) as sess:
            start_time = time.time()

            res = sess.download(query, mode=mode)

            if verbose:
                self.logger.info(
                    f"{'Finish'} "
                    f"(runtime: {self.trans_sec(time.time() - start_time)}) Query: {self.oneline_query(query)} ..."
                )
            return res

    def upload(self, df, table_name, replace=True, verbose=True):
        with Session(auth=self.auth) as sess:
            start_time = time.time()

            if replace:
                sess.upload(df, table_name, replace=True)
            else:
                sess.upsert(df, table_name)

            if verbose:
                self.logger.info(
                    f"Finish {'upload' if replace else 'upsert'} with {str(df.shape)} data "
                    f"(runtime: {self.trans_sec(time.time() - start_time)}) Table: {table_name}"
                )

    def sample_table(self, table_name, n=1):
        query = f'select * from {table_name} limit {n}'
        res = self.run(query)
        return res

    def download_table(self, table_name):
        query = f'select * from {table_name}'
        res = self.run(query)
        return res

    def count_table(self, table_name):
        query = f'select count(1) as cnt from {table_name}'
        res = self.run(query)
        return res

    def create_table(self, table_name, table_query, verbose=True):
        query = """
            create or replace table {table_name} as
            {table_query}
        """.format(
            table_name=table_name,
            table_query=table_query
        )
        return self.execute(query=query, verbose=verbose)


class GCloudStorageConnector(_CloudStorageConnector):
    @staticmethod
    def ls(path):
        return GCSFileSystem().ls(path)

    @staticmethod
    def copy(lpath, rpath, recursive=False):
        GCSFileSystem().copy(lpath, rpath, recursive=recursive)

    @staticmethod
    def remove(path, recursive=False):
        GCSFileSystem().remove(path, recursive=recursive)

    @staticmethod
    def isdir(path):
        return GCSFileSystem().isdir(path)

    @staticmethod
    def save_pickle(data, path):
        with GCSFileSystem().open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(path):
        with GCSFileSystem().open(path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_text(data, path):
        with GCSFileSystem().open(path, "w") as f:
            f.write(data)

    @staticmethod
    def load_text(path):
        with GCSFileSystem().open(path, "r") as f:
            data = f.read()
        return data

    @staticmethod
    def save_binary(data, path):
        with GCSFileSystem().open(path, "wb") as f:
            f.write(data)

    @staticmethod
    def load_binary(path):
        with GCSFileSystem().open(path, "rb") as f:
            data = f.read()
        return data

    @staticmethod
    def save_json(data, path):
        with GCSFileSystem().open(path, "wb") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(path):
        with GCSFileSystem().open(path, "rb") as f:
            data = json.load(f)
        return data
