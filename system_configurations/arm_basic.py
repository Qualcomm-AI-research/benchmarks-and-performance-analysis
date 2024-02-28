# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
Gem5 System configuration script for a multi-processor Arm based system inspired 
by Gem5's provided starter_se configuration for Arm. 

By default, the system is comprised of 4 out-of-order Arm cores with private 
L1/L2 caches, and a shared L3 cache. 

Usage: 
    gem5 arm_basic.py --binaries [binary path]

Author: nichrich@qualcomm.qti.com 
"""
import argparse 
import os 
import sys 
import time 
from typing import List

from io_utils import GEM5_CONFIGS_DIRECTORY, GEM5_ARM_BINARY_DIRECTORY, GEM5_TEST_PROGRAM_DIRECTORY, human_seconds_str

sys.path.insert(0, GEM5_CONFIGS_DIRECTORY)
sys.path.insert(0, os.path.join(GEM5_CONFIGS_DIRECTORY, "learning_gem5/part1"))
sys.path.insert(0, os.path.join(GEM5_CONFIGS_DIRECTORY, "example/arm"))

import m5
from m5.objects import *
from m5.stats import periodicStatDump
#from gem5.runtime import get_runtime_isa
from common import SimpleOpts, ObjectList, MemConfig
from common.cores.arm import HPI
from common.Options import addCommonOptions
from common.cpu2000 import gcc

from caches import L1DCache, L1ICache, L2Cache
import devices 

from custom_logging import setup_logger

cpu_types = {
    "atomic": (AtomicSimpleCPU, None, None, None),
    "minor": (MinorCPU, devices.L1I, devices.L1D, devices.L2),
    "hpi": (HPI.HPI, HPI.HPI_ICache, HPI.HPI_DCache, HPI.HPI_L2),
    "O3Cpu": (O3CPU, devices.L1I, devices.L1D, devices.L2),
}

parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Base simulation configuration for a simple Arm system with two level cache hierarchy.")

parser.add_argument_group(title="Workload")
parser.add_argument("--binaries", type=str, nargs="+", help="Path to binary(ies) whose execution should be simulated.")
parser.add_argument("--binary-args", type=str, nargs="*", default="", help="Optional arguments to pass to the binary to be simulated.")

parser.add_argument_group(title="System Configuration")
parser.add_argument("--cpu", type=str, default="O3Cpu", choices=list(cpu_types.keys()), help="CPU model.")
parser.add_argument("--cpu-freq", type=str, default="1GHz")
parser.add_argument("--num-cores", type=int, default=1, help="Number of CPU cores")
parser.add_argument("--mem-type", default="DDR3_1600_8x8", choices=ObjectList.mem_list.get_names(), help="type of memory to use")
parser.add_argument("--mem-channels", type=int, default=2, help="number of memory channels")
parser.add_argument("--mem-ranks", type=int, default=None, help="number of memory ranks per channel")
parser.add_argument("--mem-size", action="store", type=str, default="8GB", help="Specify the physical memory size")

parser.add_argument("--clock-rate", type=str, default="1GHz", help="System (global) clock rate.")
parser.add_argument("--address-range", type=str, default="8GB", help="Memory address range.")

parser.add_argument_group(title="Telemetry")
parser.add_argument("--disable-sampling", action="store_true", help="If invoked, disables periodic sampling and results in a single summary statistics output.")
parser.add_argument("--sampling-period", default=1_000_000, type=int, help="Sampling period for statistics (in units of 'ticks').")


class ArmA57Soc(System): 
    """Arm A57-based system on chip (SoC) configuration. 
    """
    cache_line_size = 64

    def __init__(self, args, **kwargs):
        super(ArmA57Soc, self).__init__(**kwargs)

        # book keeping to be able to use CpuClusters from the devices module.
        self._clusters = []
        self._num_cpus = 0
        self._my_num_cpus = args.num_cores

        # voltage and clock domain for system components
        self.voltage_domain = VoltageDomain(voltage="3.3V")
        self.clk_domain = SrcClockDomain(clock="1GHz", voltage_domain=self.voltage_domain)

        # off-chip memory bus.
        self.membus = SystemXBar()
        self._cluster_mem_bus = self.membus

        # technically unnecessary: this is the port used for loading the kernel 
        self.system_port = self.membus.cpu_side_ports

        self.cpu_cluster = devices.ArmCpuCluster(self, args.num_cores, args.cpu_freq, "1.2V", *cpu_types[args.cpu])
        self.configure_caches()

    def configure_caches(self): 
        if self.cpu_cluster.memory_mode() == "timing": 
            self.cpu_cluster.addL1()
            self.cpu_cluster.addL2(self.cpu_cluster.clk_domain)

            # add a (shared) L3 cache 
            max_clock_cluster = max(
                self._clusters, key=lambda c: c.clk_domain.clock[0]
            )
            self.l3 = devices.L3(clk_domain=max_clock_cluster.clk_domain)
            self.toL3Bus = L2XBar(width=64)
            self.toL3Bus.mem_side_ports = self.l3.cpu_side
            self.l3.mem_side = self.membus.cpu_side_ports
            self._cluster_mem_bus = self.toL3Bus

            self.cpu_cluster.connectMemSide(self._cluster_mem_bus)
            self.mem_mode = self.cpu_cluster.memory_mode()

    def numCpuClusters(self):
        return len(self._clusters)

    def addCpuCluster(self, cpu_cluster):
        assert cpu_cluster not in self._clusters
        assert self._my_num_cpus > 0
        self._clusters.append(cpu_cluster)
        self._num_cpus += self._my_num_cpus

    def numCpus(self):
        return self._num_cpus

def configure_system(args: argparse.Namespace) -> m5.objects.System: 
    system = ArmA57Soc(args)

    system.mem_ranges = [AddrRange(start=0, size=args.mem_size)]
    MemConfig.config_mem(args, system)

    return system

def configure_cache_hierarchy(system: m5.objects.System, args: argparse.Namespace): 
    system.cpu.icache = L1ICache(args)
    system.cpu.dcache = L1DCache(args)
    system.cpu.icache.connectCPU(system.cpu)
    system.cpu.dcache.connectCPU(system.cpu)
    system.l2bus = L2XBar()
    system.cpu.icache.connectBus(system.l2bus)
    system.cpu.dcache.connectBus(system.l2bus)
    system.l2cache = L2Cache(args)
    system.l2cache.connectCPUSideBus(system.l2bus)
    system.membus = SystemXBar()
    system.l2cache.connectMemSideBus(system.membus)

def configure_memory_controller(system: m5.objects.System): 
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports

def configure_processes(binaries: List[str], binary_args: List[str], log=None) -> List[Process]: 
    if binary_args == "": 
        no_args = True
    else: 
        no_args = False

    cwd = os.getcwd()
    processes: List[Process] = [] 

    if no_args:
        for i, binary in enumerate(binaries): 
            process = Process(pid=100 + i, cwd=cwd, cmd=[binary], executable=binary)
            process.gid = os.getgid()
            if log is not None: 
                log.info(f"Command: {process.cmd}")
            else: 
                print("info: %d. command and arguments: %s" % (i + 1, process.cmd))
            processes.append(process)
    else: 
        for i, (binary, binary_args) in enumerate(zip(binaries, binary_args)): 
            command = [binary] + binary_args.split()
            process = Process(pid=100 + i, cwd=cwd, cmd=command, executable=binary)
            process.gid = os.getgid()
            if log is not None: 
                log.info(f"Command: {process.cmd}")
            else: 
                print("info: %d. command and arguments: %s" % (i + 1, process.cmd))
            processes.append(process)

    return processes


def main(args: argparse.Namespace): 
    log = setup_logger(__name__)

    setup_time_start: float = time.perf_counter()

    system = configure_system(args)

    # configure workload 
    processes: List[Process] = configure_processes(args.binaries, args.binary_args)

    if len(processes) != args.num_cores: 
        raise ValueError(f"Cannot map {len(processes)} processes onto {args.num_cores} CPU(s)")

    # system.workload = SEWorkload.init_compatible(args.binary)
    system.workload = SEWorkload.init_compatible(processes[0].executable)

    for core, thread in zip(system.cpu_cluster.cpus, processes): 
        core.workload = thread 

    root = Root(full_system=False, system=system)

    setup_runtime: float = time.perf_counter() - setup_time_start
    log.info(f"Finished system setup in: {human_seconds_str(setup_runtime)}")

    m5.instantiate()
    if not args.disable_sampling: 
        periodicStatDump(args.sampling_period)

    log.info("Starting simulation!")
    simulation_start: float = time.perf_counter()
    exit_event = m5.simulate()
    simulation_runtime: float = time.perf_counter() - simulation_start
    log.info(f"Simulation took: {human_seconds_str(simulation_runtime)}")
    log.info(f"Exiting at tick {m5.curTick()} because {exit_event.getCause()}")

if __name__=="__m5_main__": 
    args: argparse.Namespace = parser.parse_args()
    main(args) 
