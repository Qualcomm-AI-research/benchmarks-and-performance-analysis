Copyright (c) 2016-2017 ARM Limited
All rights reserved.

The license below extends only to copyright in the software and shall
not be construed as granting a license to any other intellectual
property including but not limited to intellectual property relating
to a hardware implementation of the functionality of the software
licensed hereunder.  You may use the software subject to the license
terms below provided that you ensure that this notice is replicated
unmodified and in its entirety in all distributions of the software,
modified or unmodified, in source code or in binary form.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer;
redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution;
neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse 
import dataclasses 
import os 
import pathlib 
from pathlib import Path 
import time 
from typing import Optional, List

from common import ex5_big, ex5_LITTLE
from custom_logging import setup_logger
from io_utils import human_seconds_str

@dataclasses.dataclass 
class ArmSystemConfig: 
    cache_line_size: Optional[int] = 64 # [B]
    core_type: Optional[str] = "ex5"
    num_cores: Optional[int] = 4
    system_voltage: Optional[str] = "3.3V"
    cpu_voltage: Optional[str] = "1.2V"
    cpu_clock_rate: Optional[str] = ""
    system_clock_rate: Optional[str] = "1GHz"
    memory_size: Optinal[str] = "2GB"

parser = argparse.ArgumentParser(description="Arm OOO multi-processor configuration setup for gem5.")

parser.add_argument_group(title="Core Configuration")
parser.add_argument("--core-type", default="ex5", type=str, help="CPU core model." )
parser.add_argument("--num-cores", default=4, type=int, help="Number of cores to simulate.")
parser.add_argument("--cache-line-size", default=64, type=int, help="Cache line size in bytes.")

parser.add_argument_group(title="Power Distribution")
parser.add_argument("--system-voltage", default="3.3V", type=str, help="System-wide voltage.")
parser.add_argument("--cpu-voltage", default="1.2V", type=str, help="CPU voltage.")

parser.add_argument_group(title="Clock Distribution")
parser.add_argument("--cpu-clock-rate", default="4GHz", type=str, help="CPU clock rate.")
parser.add_argument("--system-clock-rate", default="1GHz", type=str, help="System-wide clock rate.")

parser.add_argument_group(title="Memory System")
parser.add_argument("--mem-size", default="2GB", type=str, help="Physical memory size.")
parser.add_argument("--mem-ranks", default=None, type=str, help="Number of memory ranks (per channel).")
parser.add_argument("--mem-channels", default=2, type=str, help="Number of memory channels.")
parser.add_argument("--mem-type", default="DDR3_1600_8x8", type=str, help="Memory technology.")

parser.add_argument("--binary", type=str, required=True, help="Binary to run.")

class ArmSystem(System): 
    def __init__(self, config: ArmSystemConfig, **kwargs): 
        super(ArmSystem, self).__init__(**kwargs)
        self.config = config

        # required for compatibility with devices module 
        self._clusters: List = [] 
        self._num_cpus: int = 0 

        self.initialize_power_distribution()
        self.initialize_clock_distribution()

        self.membus: SystemXBar = SystemXBar()
        self.system_port = self.membus.cpu_side_ports

        self.initialize_cpus()


    def initialize_power_distribution(self): 
        self.voltage_domain: VoltageDomain = VoltageDomain(voltage=self.config.system_voltage)

    def initialize_clock_distribution(self): 
        self.clk_domain: SrcClockDomain = SrcClockDomain(clock=self.config.system_clock_rate, voltage_domain=self.voltage_domain)

    def initialize_cpus(self): 
        breakpoint() 
        # TODO *cpu_types[self.config.core_type]
        self.cpu_cluster = devices.CpuCluster(
                                self, 
                                self.config.num_cores, 
                                self.config.cpu_clock_rate, 
                                self.config.cpu_voltage, 
                                )

        self.cpu_cluster.addL1()
        self.cpu_cluster.addL2(self.cpu_cluster.clk_domain)
        self.cpu_cluster.connectMemSide(self.membus)
        self.mem_mode = self.cpu_cluster.memoryMode()

    def numCpuClusters(self): 
        return len(self._clusters)

    def addCpuCluster(self, cpu_cluster, num_cpus): 
        assert cpu_cluster not in self._clusters 
        assert num_cpus > 0 
        self._clusters.append(cpu_cluster)
        self._num_cpus += num_cpus 

    def numCpus(self): 
        return self._num_cpus

def main(args: argparse.ArgumentParser): 
    log = setup_logger(__name__)

    # configure and instantiate system
    system_config: ArmSystemConfig = ArmSystemConfig(
        cache_line_size=args.cache_line_size, 
        core_type=args.core_type, 
        num_cores=args.num_cores, 
        system_voltage=args.system_voltage, 
        cpu_voltage=args.cpu_voltage, 
        cpu_clock_rate=args.cpu_clock_rate, 
        system_clock_rate=args.system_clock_rate, 
        memory_size=args.memory_size
    )
    system: ArmSystem = ArmSystem(system_config)
    log.info(f"Configured core system.")

    # configure off-chip memory
    system.mem_ranges = [AddrRange(start=0, size=args.mem_size)]
    MemConfig.config_mem(args, system)
    log.info(f"Configured memory system.")

    # configure workload
    processes: List[]




    root = Root(full_system=False)


    m5.instantiate()

    start: float = time.perf_counter()
    results = m5.simulate()
    runtime: float = time.perf_counter() - start

    log.info(f"Simulation completed in {human_seconds_str(runtime)} seconds.")
    log.info(f"Simulation exited with code {results.getCode()} because of {results.getCause()} at {m5.curTick()}")
    pass 

if __name__=="__m5_main__": 
    args = parser.parse_args()
    main(args)
