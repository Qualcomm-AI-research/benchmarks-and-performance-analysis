# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
import dataclasses 
import enum 
import gzip 
import logging 
import os 
import pathlib
from pathlib import Path
import re 
import shutil
import subprocess
import sys 
from typing import Dict, List, Optional

import numpy as np 
import tqdm 

from custom_logging import setup_logger
from io_utils import save_object, save_archive, human_bytes_str

parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Parser module for gem5 statistic output files.")
parser.add_argument("--input", type=str, required=True, help="Input file to parse.")
parser.add_argument("--no-backup", action="store_true", help="Run without backing up the source file.")
parser.add_argument("--remove-backup", action="store_true", help="Remove backup.")
parser.add_argument("--archive", action="store_true")
parser.add_argument("--stream", action="store_true", help="Run in streaming mode (helpful for very large statistics files)")
parser.add_argument("--num-blocks", type=int, default=50, help="Number of blocks to read in parallel (streaming mode).")

@dataclasses.dataclass
class ParserConfig: 
    enable_whitelist: Optional[bool]=False
    enable_blacklist: Optional[bool]=False
    whitelist: Optional[List[str]]=None
    blacklist: Optional[List[str]]=None
    no_backup: Optional[bool]=False
    stream: Optional[bool]=False
    num_blocks: Optional[int]=50

class PMUEvents(enum.Enum): 
    NUM_L1_INSTRUCTION_CACHE_MISSES="system.cpu.icache.overallMisses::total"
    NUM_L1_INSTRUCTION_CACHE_ACCESSES="system.cpu.icache.overallAccesses::total"

    NUM_L1_DATA_CACHE_MISSES="system.cpu.dcache.overallMisses::total"
    NUM_L1_DATA_CACHE_ACCESSES="system.cpu.dcache.overallAccesses::total"

    NUM_INSTRUCTIONS_RETIRED="system.cpu.exec_context.thread_0.numInsts" # TODO generalize to multiple threads 
    NUM_CYCLES="system.cpu.numCycles"
    NUM_BRANCHES="system.cpu.dcache.overallAccesses::total"
    NUM_MEMORY_ACCESSES="system.cpu.exec_context.thread_0.numMemRefs" # TODO this may not be exactly the number of accesses

    NUM_L1_DATA_CACHE_WRITEBACKS="system.cpu.dcache.WriteReq.misses::cpu.data"
    NUM_BUS_ACCESSES="system.l2bus.pktCount::total"

class VisibleEvents(enum.Enum): 
    NUM_L1_INSTRUCTION_CACHE_MISSES="system.cpu.icache.overallMisses::total"
    NUM_L1_INSTRUCTION_CACHE_ACCESSES="system.cpu.icache.overallAccesses::total"

    NUM_L1_DATA_CACHE_MISSES="system.cpu.dcache.overallMisses::total"
    NUM_L1_DATA_CACHE_ACCESSES="system.cpu.dcache.overallAccesses::total"

    NUM_INSTRUCTIONS_RETIRED="system.cpu.exec_context.thread_0.numInsts" # TODO generalize to multiple threads 
    NUM_CYCLES="system.cpu.numCycles"
    NUM_BRANCHES="system.cpu.dcache.overallAccesses::total"
    NUM_MEMORY_ACCESSES="system.cpu.exec_context.thread_0.numMemRefs" # TODO this may not be exactly the number of accesses

    NUM_L1_DATA_CACHE_WRITEBACKS="system.cpu.dcache.WriteReq.misses::cpu.data"
    NUM_BUS_ACCESSES="system.l2bus.pktCount::total"

    # TODO branches mispredicted, probably only applies for out of order microarchitectures 

class Parser: 
    """Basic parser for gem5 statistics outputs.
    """
    def __init__(self, config: ParserConfig, **kwargs): 
        self.log = kwargs.get("log", setup_logger(__name__))
        self.config = config
        self.source = None
        self.chunk_size: Optional[int] = 100 # [chars] 
        self.parsed_data: Dict[str, np.ndarray] = {}
        self.block_delimiter_pattern: str = r"Begin Simulation Statistics(.*?)End Simulation Statistics"

    def _parse_chunk(self, chunk: str): 
        matches = re.findall(self.block_delimiter_pattern, chunk, re.DOTALL)
        identifier_value_regex = re.compile(r'^(\w+)\s+(-?\d+(?:\.\d+)?)$')

        for match in matches:
            identifiers_in_block: set = set()
            values = {}
            lines = match.strip().split("\n")

            for line in lines:
                if len(line.split()) >= 2: 
                    identifier, value, *_ = line.split()
                    if identifier not in self.config.blacklist: 
                        identifiers_in_block.add(identifier)
                        value = float(value) 

                        if identifier in self.parsed_data: 
                            self.parsed_data[identifier].append(value)
                        else: 
                            self.parsed_data[identifier] = [value] 

            for identifier in self.parsed_data: 
                if identifier not in identifiers_in_block: 
                    self.parsed_data[identifier].append(0.)

    def linecount(self): 
        count = 0 
        with gzip.open(self.source, 'rt') as handle: 
            for line in handle: 
                count += 1
        return count 

    def parse(self, source: Path): 
        self.source = source

        self.log.info("Counting source lines")
        linecount = self.linecount()

        # backup the source 
        self._backup()

        try: 
            current_blocks: str = ""
            blocks_read: int = 0 
            started: bool = False
            processed_lines: int = 0 

            with gzip.open(self.source, 'rt') as handle: 
                for line in handle: 
                    processed_lines += 1 

                    if "Begin Simulation Statistics" in line: 
                        started = True 

                    if started: 
                        current_blocks += self._remove_comments_from_line(line)

                        if "End Simulation Statistics" in line: 
                            blocks_read += 1 

                        if blocks_read == self.config.num_blocks: 
                            self._parse_chunk(current_blocks)
                            self.log.info(f"Completed {(processed_lines / linecount) * 100.0:0.3f}%")
                            current_blocks = ""
                            blocks_read = 0 
                if current_blocks: 
                    self._parse_chunk(current_blocks)
        except: 
            if not self.config.no_backup: 
                self.log.info("Restoring from backup")
                self._restore_from_backup()
                sys.exit(1)

        # convert to ndarray 
        max_length: int = max([len(x) for x in self.parsed_data.values()])
        for key in self.parsed_data: 
            while len(self.parsed_data[key]) < max_length: 
                self.parsed_data[key].append(0.)

            self.parsed_data[key] = np.array(self.parsed_data[key])

        save_path: Path = Path(self.source.absolute().as_posix().split('.')[0] + ".custom")
        self.log.info(f"saved parsed data to: {save_path}.pkl")
        save_object(self.parsed_data, save_path)

        if not self.config.no_backup: 
            self._remove_backup()

    def _backup(self): 
        self.backup: pathlib.Path = pathlib.Path(self.source.absolute().as_posix() + ".backup")
        source_size_bytes: int = os.path.getsize(self.source.as_posix())

        if source_size_bytes > (10 * (2**30)): 
            if self.config.no_backup: 
                self.log.info(f"Source file is {human_bytes_str(source_size_bytes)}, not backing it up")
            else: 
                self.log.info(f"Source file is {human_bytes_str(source_size_bytes)}, which is too large to backup... run with --no-backup to run parser without backing up the source")
                raise ValueError
        else: 
            if not self.config.no_backup:
                self.log.info("Backing up input file")
                shutil.copy(self.source, self.backup)
                self.log.info("Backed up input file...")

    def _restore_from_backup(self): 
        shutil.copy(self.backup, self.source)
        self._remove_backup()

    def _remove_backup(self): 
        self.log.info("removing backup")
        os.remove(self.backup)

    def _remove_comments_from_line(self, line: str): 
        without_comments: str = line.split('#', 1)[0]
        if without_comments[-1] != '\n': without_comments += '\n' 
        return without_comments

    def _remove_comments(self): 
        with open(self.source, 'r') as source: 
            with open(self.workspace, 'w') as workspace: 
                for line in source: 
                    line_without_comments: str = line.split('#', 1)[0]
                    if line_without_comments[-1] != '\n': 
                        line_without_comments += '\n'
                    workspace.write(line_without_comments)

    def _parse(self): 
        self.parsed_data: Dict[str, np.ndarray] = {}
        pattern: str = r"Begin Simulation Statistics(.*?)End Simulation Statistics"

        with open(self.source, "r") as file:
            file_content = file.read()

        self.log.info("Extracting blocks...")
        matches = re.findall(pattern, file_content, re.DOTALL)
        self.log.info(f"Extracted {len(matches)} blocks...")

        self.log.info("Processing blocks...")
        for match in tqdm.tqdm(matches):
            values = {}
            lines = match.strip().split("\n")
            for line in lines:
                key_value = line.strip().split()
                if len(key_value) >= 2:
                    key, value, *_ = key_value
                    if key not in self.config.blacklist:
                        values[key] = float(value)
            
            for key in self.parsed_data:
                if key not in values:
                    values[key] = 0
            
            for key, value in values.items():
                if key in self.parsed_data:
                    self.parsed_data[key].append(value)
                else:
                    self.parsed_data[key] = [value]
    
        for key in self.parsed_data:
            while len(self.parsed_data[key]) < len(matches):
                self.parsed_data[key].append(0)
            self.parsed_data[key] = np.array(self.parsed_data[key], dtype=float)

    def __call__(self, source: pathlib.Path, remove_backup: Optional[bool]=False): 
        if not source.exists(): 
            raise FileNotFoundError(f"Could not find source file: {source}")

        self.source = source 
        self._backup()

        try: 
            self.workspace: pathlib.Path = pathlib.Path(self.source.absolute().as_posix() + ".tmp")
#            self._remove_comments()
#            shutil.copy(self.workspace, self.source)
            self._parse()
            save_path: pathlib.Path = pathlib.Path(self.source.absolute().as_posix().split('.')[0] + ".pkl")
            target: os.PathLike = save_object(self.parsed_data, save_path)
            self.log.info(f"Saved parsed data to: {save_path}")

            if remove_backup and (not self.config.no_backup):
                self._remove_backup()
                self.log.info("removed backup...")
                os.remove(self.workspace)
                self.log.info("removed workspace...")
        except Exception as e: 
            self.log.error(e)
            if not self.config.no_backup:
                self.log.info(f"Restoring from backup: {self.backup}")
                self._restore_from_backup()

def main(args: argparse.Namespace): 
    PMU_whitelist: List[str] = [
            PMUEvents.NUM_L1_INSTRUCTION_CACHE_MISSES.value, 
            PMUEvents.NUM_L1_INSTRUCTION_CACHE_ACCESSES.value, 
            PMUEvents.NUM_L1_DATA_CACHE_MISSES.value, 
            PMUEvents.NUM_L1_DATA_CACHE_ACCESSES.value, 
            PMUEvents.NUM_INSTRUCTIONS_RETIRED.value, 
            PMUEvents.NUM_CYCLES.value, 
            PMUEvents.NUM_BRANCHES.value, 
            PMUEvents.NUM_MEMORY_ACCESSES.value, 
            PMUEvents.NUM_L1_DATA_CACHE_WRITEBACKS.value, 
            PMUEvents.NUM_BUS_ACCESSES.value
    ]
    blacklist: List[str] = [
        "simSeconds", 
        "simTicks", 
        "finalTick", 
        "simFreq", 
        "hostSeconds", 
        "hostTickRate", 
        "hostMemory", 
        "simInsts", 
        "simOps", 
        "hostInstRate", 
        "hostOpRate", 
        "system.clk_domain.clock", 
        "system.clk_domain.voltage_domain.voltage", 
        "system.cpu.numWorkItemsStarted", 
        "system.cpu.numWorkItemsCompleted", 
    ]

    config = ParserConfig(
        enable_blacklist=True, 
        blacklist=blacklist, 
        no_backup=args.no_backup, 
        stream=args.stream, 
        num_blocks=args.num_blocks
    )
    parser = Parser(config)
    input_file: pathlib.Path = pathlib.Path(args.input)
    parser.parse(input_file)

    if args.archive: 
        save_archive(input_file)
        os.remove(input_file)

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
