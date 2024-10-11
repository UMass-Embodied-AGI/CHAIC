from __future__ import annotations

import os
from typing import List, Union
from tdw.controller import Controller
from requests import get
from pathlib import Path
import shutil
import hashlib
import boto3

private_bucket_prefix: str = "https://tdw-private.s3.amazonaws.com/"

class AssetCachedController(Controller):
    def __init__(self, cache_dir="transport_challenge_asset_bundles", **kwargs):
        self.cache_dir = None
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            self.cache_dir = Path(cache_dir)
        super().__init__(**kwargs)
    
    def communicate(self, commands: dict | List[dict]):
        '''
        override and calculate add_ons' commands in advance to make cache of asset
        '''
        if self.cache_dir is None:
            return super().communicate(commands)
        
        if isinstance(commands, dict):
            commands = [commands]
        add_ons_t = self.add_ons
        self.add_ons = [] # remove add_ons
        for m in add_ons_t:
            if not m.initialized:
                commands.extend(m.get_initialization_commands())
                m.initialized = True
            else:
                commands.extend(m.commands)
                m.commands.clear()
                
        # for cmd in commands:
        #     print(cmd)

        # print()
        for m in add_ons_t:
            m.before_send(commands)
        
        for cmd in commands:
            if "url" in cmd:
                cmd["url"] = self.get_asset(cmd["url"])

        
            
        resp = super().communicate(commands)
        for m in add_ons_t:
            m.on_send(resp)
        self.add_ons = add_ons_t # resume add_ons
        return resp
    
    def get_asset(self, url: str):
        if not url.startswith("http"):
            return url
        
        name = hashlib.md5(url.encode()).hexdigest()
        path = self.cache_dir.joinpath(name)
        if not os.path.exists(str(path.resolve())):
            print(f"downloading {url} to {str(path)}")
            if private_bucket_prefix in url:
                s3_key = url.split(private_bucket_prefix)[1]
                session = boto3.Session(profile_name="tdw")
                s3 = session.resource("s3")
                resp = s3.meta.client.get_object(Bucket='tdw-private', Key=s3_key)
                status_code = resp["ResponseMetadata"]["HTTPStatusCode"]
                assert status_code == 200, (url, status_code)
                path.write_bytes(resp["Body"].read())
            else:
                content = get(url).content
                path.write_bytes(content)
        return "file://" + str(path.resolve())
    
    def clear_cache(self):
        shutil.rmtree(self.cache_dir.resolve())