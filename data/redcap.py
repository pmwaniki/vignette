import os
from datetime import datetime
import numpy as np
import asyncio
from redcapdata.datasets import  get_metadata_async, Metadata, get_data_async
from typing import Optional
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
REDCAP_URL='https://redcap.datahubweb.com/api/'

async def get_benchmark_event(event,formated:bool=False):
    raw_data= await get_data_async(url=REDCAP_URL,id_var='record_id',variables=('record_id',),events=(event,),
                                   forms=('evaluation',),token=os.getenv('BENCHMARK_TOKEN'),
                                   convert_to_pandas=True)
    if formated:
        metadata = await get_metadata_async(url=REDCAP_URL, token=os.getenv('BENCHMARK_TOKEN'))
        metadata = Metadata(metadata)
        formated_data=raw_data.copy()
        for c in metadata.get_variables_without_description():
            try:
                formated_data[c]=metadata.format_column(c,formated_data[c])
            except:
                pass
        return formated_data
    return raw_data

async def get_benchmark_preliminary(formated:bool=False):
    raw_data= await get_data_async(url=REDCAP_URL,id_var='record_id',variables=('record_id','study_id'),events=('preliminary_arm_1',),
                                   forms=('form_1',),token=os.getenv('BENCHMARK_TOKEN'),
                                   convert_to_pandas=True)
    if formated:
        metadata = await get_metadata_async(url=REDCAP_URL, token=os.getenv('BENCHMARK_TOKEN'))
        metadata = Metadata(metadata)
        formated_data=raw_data.copy()
        for c in metadata.get_variables_without_description():
            try:
                formated_data[c]=metadata.format_column(c,formated_data[c])
            except:
                pass
        return formated_data
    return raw_data

async def get_benchmark_metadata():
    metadata = await get_metadata_async(url=REDCAP_URL, token=os.getenv('BENCHMARK_TOKEN'))
    metadata = Metadata(metadata)
    return metadata