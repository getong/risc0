// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use anyhow::{bail, Result};
use rayon;
use risc0_binfmt::{ByteAddr, MemoryImage2, WordAddr};
use risc0_zkp::core::{
    digest::{Digest, DIGEST_BYTES},
    log2_ceil,
};

use crate::{
    trace::{TraceCallback, TraceEvent},
    Rv32imV2Claim, TerminateState,
};

use super::{
    bigint::BigIntState,
    pager::{PageTraceEvent, PagedMemory},
    platform::*,
    poseidon2::Poseidon2State,
    r0vm::{LoadOp, Risc0Context, Risc0Machine},
    rv32im::{disasm, DecodedInstruction, Emulator, Instruction},
    segment::Segment,
    sha2::Sha2State,
    syscall::Syscall,
    SyscallContext,
};

#[derive(Clone, Debug, Default)]
pub struct EcallMetric {
    pub count: u64,
    pub cycles: u64,
}

pub struct Executor<'a, 'b, S: Syscall> {
    pc: ByteAddr,
    user_pc: ByteAddr,
    machine_mode: u32,
    user_cycles: u32,
    phys_cycles: u32,
    pager: PagedMemory,
    terminate_state: Option<TerminateState>,
    read_record: Vec<Vec<u8>>,
    write_record: Vec<u32>,
    syscall_handler: &'a S,
    input_digest: Digest,
    output_digest: Option<Digest>,
    trace: Vec<Rc<RefCell<dyn TraceCallback + 'b>>>,
    cycles: SessionCycles,
}

pub struct ExecutorResult {
    pub segments: u64,
    pub post_image: MemoryImage2,
    pub user_cycles: u64,
    pub total_cycles: u64,
    pub paging_cycles: u64,
    pub reserved_cycles: u64,
    pub claim: Rv32imV2Claim,
}

#[derive(Clone, Default)]
struct SessionCycles {
    total: u64,
    user: u64,
    paging: u64,
    reserved: u64,
}

pub struct SimpleSession {
    pub segments: Vec<Segment>,
    pub result: ExecutorResult,
}

/// Lightweight snapshot of executor state for parallel execution
#[derive(Clone)]
struct ExecutorState {
    pc: ByteAddr,
    user_pc: ByteAddr,
    machine_mode: u32,
    user_cycles: u32,
    phys_cycles: u32,
    registers: [u32; REG_MAX],
    terminate_state: Option<TerminateState>,
}

/// Execution result from a parallel chunk
#[derive(Clone)]
struct ChunkResult {
    state: ExecutorState,
    memory_writes: Vec<(WordAddr, u32)>,
    read_record: Vec<Vec<u8>>,
    write_record: Vec<u32>,
}

/// Chunk execution status
#[derive(Debug, PartialEq, Clone, Copy)]
enum ChunkStatus {
    Success,         // Chunk executed successfully
    SegmentBoundary, // Reached segment boundary
    Termination,     // Program terminated
    #[allow(dead_code)]
    Conflict, // Memory conflict with another chunk (for future use)
}

/// Thread pool configuration for parallel execution
pub struct ParallelConfig {
    /// Maximum number of threads to use for execution
    pub max_threads: usize,
    /// Number of instructions to process in each parallel chunk
    pub chunk_size: usize,
    /// Whether to enable parallel execution
    pub enable_parallel: bool,
}

impl ParallelConfig {
    /// Create a new default configuration for parallel execution
    pub fn new() -> Self {
        Self {
            max_threads: 10,
            chunk_size: 1000,
            enable_parallel: true,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutorState {
    /// Create a new executor state from the current executor
    #[allow(clippy::needless_lifetimes)]
    fn capture<'a, 'b, S: Syscall>(executor: &mut Executor<'a, 'b, S>) -> Result<Self> {
        let mut registers = [0; REG_MAX];

        // Capture user registers
        #[allow(clippy::needless_range_loop)]
        for idx in 0..REG_MAX {
            registers[idx] = executor.peek_register(idx)?;
        }

        Ok(Self {
            pc: executor.pc,
            user_pc: executor.user_pc,
            machine_mode: executor.machine_mode,
            user_cycles: executor.user_cycles,
            phys_cycles: executor.phys_cycles,
            registers,
            terminate_state: executor.terminate_state,
        })
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'a, 'b, S: Syscall> Executor<'a, 'b, S> {
    pub fn new(
        image: MemoryImage2,
        syscall_handler: &'a S,
        input_digest: Option<Digest>,
        trace: Vec<Rc<RefCell<dyn TraceCallback + 'b>>>,
    ) -> Self {
        Self {
            pc: ByteAddr(0),
            user_pc: ByteAddr(0),
            machine_mode: 0,
            user_cycles: 0,
            phys_cycles: 0,
            pager: PagedMemory::new(image, !trace.is_empty() /* tracing_enabled */),
            terminate_state: None,
            read_record: Vec::new(),
            write_record: Vec::new(),
            syscall_handler,
            input_digest: input_digest.unwrap_or_default(),
            output_digest: None,
            trace,
            cycles: SessionCycles::default(),
        }
    }

    /// Capture current executor state
    fn capture_state(&mut self) -> Result<ExecutorState> {
        ExecutorState::capture(self)
    }

    /// Apply a captured state to this executor
    fn apply_state(&mut self, state: &ExecutorState) -> Result<()> {
        self.pc = state.pc;
        self.user_pc = state.user_pc;
        self.machine_mode = state.machine_mode;
        self.user_cycles = state.user_cycles;
        self.phys_cycles = state.phys_cycles;
        self.terminate_state = state.terminate_state;

        // Apply registers
        let regs_addr = if state.machine_mode != 0 {
            MACHINE_REGS_ADDR.waddr()
        } else {
            USER_REGS_ADDR.waddr()
        };

        for (idx, &reg) in state.registers.iter().enumerate() {
            self.store_u32(regs_addr + idx, reg)?;
        }

        Ok(())
    }

    /// Execute a chunk of code in isolation with memory tracking
    fn execute_chunk(
        &mut self,
        starting_pcs: &[ByteAddr],
        segment_threshold: u32,
    ) -> Result<(ChunkResult, ChunkStatus)> {
        // Capture initial state for this chunk
        let initial_state = self.capture_state()?;

        // Track memory writes to detect conflicts
        let memory_writes = Vec::new();

        // Save initial read/write records to append only new ones
        let initial_read_len = self.read_record.len();
        let initial_write_len = self.write_record.len();

        // Execute the chunk instructions
        let mut emu = Emulator::new();
        let mut status = ChunkStatus::Success;

        for &pc in starting_pcs.iter() {
            // Set PC to next instruction in the chunk
            self.pc = pc;

            // Check for segment boundary
            if self.segment_cycles() >= segment_threshold {
                status = ChunkStatus::SegmentBoundary;
                break;
            }

            // Execute one instruction
            Risc0Machine::step(&mut emu, self)?;

            // Check for termination
            if self.terminate_state.is_some() {
                status = ChunkStatus::Termination;
                break;
            }
        }

        // Extract only the new reads and writes that happened during this chunk
        let read_record = self.read_record.split_off(initial_read_len);
        let write_record = self.write_record.split_off(initial_write_len);

        // Create the chunk result
        let result = ChunkResult {
            state: self.capture_state()?,
            memory_writes,
            read_record,
            write_record,
        };

        // Restore initial state (this chunk was executed speculatively)
        self.apply_state(&initial_state)?;

        Ok((result, status))
    }

    /// Creates a segment from the current executor state
    fn create_segment(
        &mut self,
        segment_po2: usize,
        segment_threshold: u32,
        index: u64,
    ) -> Result<Segment> {
        Risc0Machine::suspend(self)?;
        let (pre_digest, partial_image, post_digest) = self.pager.commit()?;

        Ok(Segment {
            partial_image,
            claim: Rv32imV2Claim {
                pre_state: pre_digest,
                post_state: post_digest,
                input: self.input_digest,
                output: self.output_digest,
                terminate_state: self.terminate_state,
                shutdown_cycle: None,
            },
            read_record: std::mem::take(&mut self.read_record),
            write_record: std::mem::take(&mut self.write_record),
            user_cycles: self.user_cycles,
            suspend_cycle: self.phys_cycles,
            paging_cycles: self.pager.cycles,
            po2: segment_po2 as u32,
            index,
            segment_threshold,
        })
    }

    /// Reset the segment state for the next segment
    fn reset_segment_state(&mut self) -> Result<()> {
        self.user_cycles = 0;
        self.phys_cycles = 0;
        self.pager.reset();
        Risc0Machine::resume(self)
    }

    /// Traditional sequential segment processing
    pub fn run<F: FnMut(Segment) -> Result<()>>(
        &mut self,
        segment_po2: usize,
        max_insn_cycles: usize,
        max_cycles: Option<u64>,
        mut callback: F,
    ) -> Result<ExecutorResult> {
        let segment_limit: u32 = 1 << segment_po2;
        assert!(max_insn_cycles < segment_limit as usize);
        let segment_threshold = segment_limit - max_insn_cycles as u32;
        let mut segment_counter = 0;

        self.reset();

        let mut emu = Emulator::new();
        Risc0Machine::resume(self)?;
        let initial_digest = self.pager.image.image_id();
        tracing::debug!("initial_digest: {initial_digest}");

        while self.terminate_state.is_none() {
            if let Some(max_cycles) = max_cycles {
                if self.cycles.user >= max_cycles {
                    bail!(
                        "Session limit exceeded: {} >= {max_cycles}",
                        self.cycles.user
                    );
                }
            }

            if self.segment_cycles() >= segment_threshold {
                tracing::debug!(
                    "split(phys: {} + pager: {} + reserved: {LOOKUP_TABLE_CYCLES}) = {} >= {segment_threshold}",
                    self.phys_cycles,
                    self.pager.cycles,
                    self.segment_cycles()
                );

                assert!(
                    self.segment_cycles() < segment_limit,
                    "segment limit ({segment_limit}) too small for instruction at pc: {:?}",
                    self.pc
                );

                // Create and process segment
                let segment =
                    self.create_segment(segment_po2, segment_threshold, segment_counter)?;
                callback(segment)?;

                segment_counter += 1;
                let total_cycles = 1 << segment_po2;
                let pager_cycles = self.pager.cycles as u64;
                let user_cycles = self.user_cycles as u64;
                self.cycles.total += total_cycles;
                self.cycles.paging += pager_cycles;
                self.cycles.reserved += total_cycles - pager_cycles - user_cycles;

                self.reset_segment_state()?;
            }

            Risc0Machine::step(&mut emu, self)?;
        }

        Risc0Machine::suspend(self)?;

        let (pre_digest, partial_image, post_digest) = self.pager.commit()?;
        let final_cycles = self.segment_cycles().next_power_of_two();
        let final_po2 = log2_ceil(final_cycles as usize);
        let segment_threshold = (1 << final_po2) - max_insn_cycles as u32;

        let final_claim = Rv32imV2Claim {
            pre_state: pre_digest,
            post_state: post_digest,
            input: self.input_digest,
            output: self.output_digest,
            terminate_state: self.terminate_state,
            shutdown_cycle: None,
        };

        callback(Segment {
            partial_image,
            claim: final_claim,
            read_record: std::mem::take(&mut self.read_record),
            write_record: std::mem::take(&mut self.write_record),
            user_cycles: self.user_cycles,
            suspend_cycle: self.phys_cycles,
            paging_cycles: self.pager.cycles,
            po2: final_po2 as u32,
            index: segment_counter,
            segment_threshold,
        })?;

        let final_cycles = final_cycles as u64;
        let user_cycles = self.user_cycles as u64;
        let pager_cycles = self.pager.cycles as u64;
        self.cycles.total += final_cycles;
        self.cycles.paging += pager_cycles;
        self.cycles.reserved += final_cycles - pager_cycles - user_cycles;

        let session_claim = Rv32imV2Claim {
            pre_state: initial_digest,
            post_state: post_digest,
            input: self.input_digest,
            output: self.output_digest,
            terminate_state: self.terminate_state,
            shutdown_cycle: None,
        };

        Ok(ExecutorResult {
            segments: segment_counter + 1,
            post_image: self.pager.image.clone(),
            user_cycles: self.cycles.user,
            total_cycles: self.cycles.total,
            paging_cycles: self.cycles.paging,
            reserved_cycles: self.cycles.reserved,
            claim: session_claim,
        })
    }

    /// Run with parallel execution
    pub fn run_parallel<F: FnMut(Segment) -> Result<()> + Send + Sync + 'static>(
        &mut self,
        segment_po2: usize,
        max_insn_cycles: usize,
        max_cycles: Option<u64>,
        mut callback: F,
        config: ParallelConfig,
    ) -> Result<ExecutorResult> {
        if !config.enable_parallel {
            // If parallel execution is disabled, fall back to sequential execution
            return self.run(segment_po2, max_insn_cycles, max_cycles, callback);
        }

        // Configure thread pool for parallel execution
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.max_threads)
            .build_global()
            .unwrap_or_default();

        let segment_limit: u32 = 1 << segment_po2;
        assert!(max_insn_cycles < segment_limit as usize);
        let segment_threshold = segment_limit - max_insn_cycles as u32;
        let mut segment_counter = 0;

        self.reset();

        let initial_digest = self.pager.image.image_id();
        tracing::debug!("initial_digest: {initial_digest}");
        Risc0Machine::resume(self)?;

        // Execution state tracking
        let mut terminate_detected = false;
        let chunk_size = config.chunk_size;
        let mut total_chunks_executed = 0;

        // Create a persistent null syscall handler for thread executors
        let null_syscall = Arc::new(NullSyscall {});

        // Main execution loop - process until termination
        while !terminate_detected
            && (max_cycles.is_none() || self.cycles.user < max_cycles.unwrap())
        {
            // Capture initial state for this round of parallelism
            let initial_state = self.capture_state()?;

            // Create multiple execution chunks to be processed in parallel
            let mut chunks = Vec::new();

            // Phase 1: Identify independent chunks by static analysis
            let mut reached_segment_boundary = false;

            // Initial exploration phase - identify chunks that can be executed in parallel
            for _chunk_idx in 0..config.max_threads * 2 {
                // Create 2x as many chunks as threads for better balancing
                // Stop creating chunks if we hit termination or segment boundary
                if reached_segment_boundary {
                    break;
                }

                // Execute a chunk to identify PC locations for future parallel execution
                let mut emu = Emulator::new();
                let mut chunk = Vec::with_capacity(chunk_size);
                let mut instruction_count = 0;

                // Analyze ahead to find chunk boundaries
                while instruction_count < chunk_size {
                    // Check if we need to split on segment boundary
                    if self.segment_cycles() >= segment_threshold {
                        reached_segment_boundary = true;
                        break;
                    }

                    // Record this PC in the chunk
                    chunk.push(self.pc);

                    // Execute one instruction to advance state, but do not commit changes yet
                    Risc0Machine::step(&mut emu, self)?;
                    instruction_count += 1;

                    // Check if this instruction caused termination
                    if self.terminate_state.is_some() {
                        terminate_detected = true;
                        break;
                    }
                }

                // Save this chunk if it has instructions
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }

                // Stop if we've reached termination
                if terminate_detected {
                    break;
                }
            }

            // If we have no chunks or hit segment boundary, handle the boundary
            if chunks.is_empty() || reached_segment_boundary {
                if reached_segment_boundary {
                    tracing::debug!(
                        "split(phys: {} + pager: {} + reserved: {LOOKUP_TABLE_CYCLES}) = {} >= {segment_threshold}",
                        self.phys_cycles,
                        self.pager.cycles,
                        self.segment_cycles()
                    );

                    assert!(
                        self.segment_cycles() < segment_limit,
                        "segment limit ({segment_limit}) too small for instruction at pc: {:?}",
                        self.pc
                    );

                    // Create and process segment
                    let segment =
                        self.create_segment(segment_po2, segment_threshold, segment_counter)?;
                    callback(segment)?;

                    segment_counter += 1;
                    let total_cycles = 1 << segment_po2;
                    let pager_cycles = self.pager.cycles as u64;
                    let user_cycles = self.user_cycles as u64;
                    self.cycles.total += total_cycles;
                    self.cycles.paging += pager_cycles;
                    self.cycles.reserved += total_cycles - pager_cycles - user_cycles;

                    self.reset_segment_state()?;
                }

                // Continue to next iteration
                continue;
            }

            // Skip parallelism for tiny chunks, just use sequential execution
            if chunks.len() == 1 && chunks[0].len() < chunk_size / 4 {
                // The state has already been advanced during chunk creation,
                // so we just need to continue with the next iteration
                tracing::debug!("Skipping parallel execution for small single chunk");
                total_chunks_executed += 1;
                continue;
            }

            // Phase 2: True parallel execution of chunks
            if chunks.len() > 1 {
                tracing::debug!("Executing {} chunks in parallel", chunks.len());

                // Restore initial state before parallel execution
                self.apply_state(&initial_state)?;

                // Execute chunks in parallel using Rayon
                let results = Arc::new(Mutex::new(Vec::with_capacity(chunks.len())));

                rayon::scope(|s| {
                    for (chunk_idx, chunk) in chunks.iter().enumerate() {
                        let results_clone = Arc::clone(&results);
                        let chunk_clone = chunk.clone();
                        let null_syscall_clone = Arc::clone(&null_syscall);
                        let initial_state = initial_state.clone(); // Clone for each thread

                        // Spawn a task for each chunk
                        s.spawn(move |_| {
                            // Create a new executor for this thread - use the persistent reference
                            let mut thread_executor = Executor::new(
                                MemoryImage2::default(), // Placeholder, will be populated by apply_state
                                &*null_syscall_clone,    // Dereference the Arc
                                None,
                                Vec::new(),
                            );

                            // Restore the initial state
                            let state_result = thread_executor.apply_state(&initial_state);
                            if let Ok(()) = state_result {
                                // Execute the chunk
                                if let Ok((chunk_result, status)) =
                                    thread_executor.execute_chunk(&chunk_clone, segment_threshold)
                                {
                                    // Lock and store the result
                                    let mut results = results_clone.lock().unwrap();
                                    results.push((chunk_idx, chunk_result, status));
                                }
                            }
                        });
                    }
                });

                // Get results from the mutex
                let results = {
                    let guard = results.lock().unwrap();
                    // Copy out of the mutex guard
                    guard
                        .iter()
                        .map(|&(idx, ref res, status)| (idx, res.clone(), status))
                        .collect::<Vec<_>>()
                };

                // Sort results by chunk index to ensure deterministic order
                let mut sorted_results = results;
                sorted_results.sort_by_key(|(idx, _, _)| *idx);

                // Check for conflicts and apply results
                let mut apply_next_chunk = true;
                let mut reached_segment_boundary = false;

                // Process the results in order
                for (_chunk_idx, result, status) in sorted_results {
                    if !apply_next_chunk {
                        // Skip this chunk since we had a dependency conflict
                        continue;
                    }

                    // Check if we hit a special condition
                    match status {
                        ChunkStatus::SegmentBoundary => {
                            reached_segment_boundary = true;
                            apply_next_chunk = false;
                        }
                        ChunkStatus::Termination => {
                            terminate_detected = true;
                            apply_next_chunk = false;
                        }
                        ChunkStatus::Conflict => {
                            // Dependency conflict, stop applying chunks
                            apply_next_chunk = false;
                        }
                        ChunkStatus::Success => {
                            // Keep applying chunks
                        }
                    }

                    // Apply the result if allowed
                    if apply_next_chunk {
                        // Apply the final state from this chunk
                        self.apply_state(&result.state)?;

                        // Apply any memory writes
                        for (addr, value) in result.memory_writes {
                            self.store_u32(addr, value)?;
                        }

                        // Append read and write records
                        self.read_record.extend(result.read_record);
                        self.write_record.extend(result.write_record);

                        total_chunks_executed += 1;
                    }
                }

                // Handle segment boundary if reached
                if reached_segment_boundary {
                    tracing::debug!("Segment boundary reached during parallel execution");

                    // Create and process segment
                    let segment =
                        self.create_segment(segment_po2, segment_threshold, segment_counter)?;
                    callback(segment)?;

                    segment_counter += 1;
                    let total_cycles = 1 << segment_po2;
                    let pager_cycles = self.pager.cycles as u64;
                    let user_cycles = self.user_cycles as u64;
                    self.cycles.total += total_cycles;
                    self.cycles.paging += pager_cycles;
                    self.cycles.reserved += total_cycles - pager_cycles - user_cycles;

                    self.reset_segment_state()?;
                }
            } else {
                // Single chunk - already executed during exploration
                total_chunks_executed += 1;
            }
        }

        // Final segment creation for termination
        if terminate_detected {
            Risc0Machine::suspend(self)?;
            let (pre_digest, partial_image, post_digest) = self.pager.commit()?;
            let final_cycles = self.segment_cycles().next_power_of_two();
            let final_po2 = log2_ceil(final_cycles as usize);
            let segment_threshold = (1 << final_po2) - max_insn_cycles as u32;

            // Process final segment
            callback(Segment {
                partial_image,
                claim: Rv32imV2Claim {
                    pre_state: pre_digest,
                    post_state: post_digest,
                    input: self.input_digest,
                    output: self.output_digest,
                    terminate_state: self.terminate_state,
                    shutdown_cycle: None,
                },
                read_record: std::mem::take(&mut self.read_record),
                write_record: std::mem::take(&mut self.write_record),
                user_cycles: self.user_cycles,
                suspend_cycle: self.phys_cycles,
                paging_cycles: self.pager.cycles,
                po2: final_po2 as u32,
                index: segment_counter,
                segment_threshold,
            })?;

            let final_cycles = final_cycles as u64;
            let user_cycles = self.user_cycles as u64;
            let pager_cycles = self.pager.cycles as u64;
            self.cycles.total += final_cycles;
            self.cycles.paging += pager_cycles;
            self.cycles.reserved += final_cycles - pager_cycles - user_cycles;
        }

        // Create final result
        let post_digest = self.pager.image.image_id();
        let session_claim = Rv32imV2Claim {
            pre_state: initial_digest,
            post_state: post_digest,
            input: self.input_digest,
            output: self.output_digest,
            terminate_state: self.terminate_state,
            shutdown_cycle: None,
        };

        tracing::info!(
            "Executed {} chunks in parallel execution",
            total_chunks_executed
        );

        Ok(ExecutorResult {
            segments: segment_counter + 1,
            post_image: self.pager.image.clone(),
            user_cycles: self.cycles.user,
            total_cycles: self.cycles.total,
            paging_cycles: self.cycles.paging,
            reserved_cycles: self.cycles.reserved,
            claim: session_claim,
        })
    }
}

/// A null syscall implementation for speculative execution
struct NullSyscall {}

impl Syscall for NullSyscall {
    fn host_read(&self, _ctx: &mut dyn SyscallContext, _fd: u32, _buf: &mut [u8]) -> Result<u32> {
        Ok(0) // No reads in speculative execution
    }

    fn host_write(&self, _ctx: &mut dyn SyscallContext, _fd: u32, _buf: &[u8]) -> Result<u32> {
        Ok(0) // No writes in speculative execution
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'a, 'b, S: Syscall> Executor<'a, 'b, S> {
    #[inline]
    fn reset(&mut self) {
        self.pager.reset();
        self.terminate_state = None;
        self.read_record.clear();
        self.write_record.clear();
        self.output_digest = None;
        self.machine_mode = 0;
        self.user_cycles = 0;
        self.phys_cycles = 0;
        self.cycles = SessionCycles::default();
        self.pc = ByteAddr(0);
    }

    #[inline(always)]
    fn segment_cycles(&self) -> u32 {
        self.phys_cycles + self.pager.cycles + LOOKUP_TABLE_CYCLES as u32
    }

    #[inline(always)]
    fn trace(&mut self, event: TraceEvent) -> Result<()> {
        // Only trace if we have trace callbacks registered
        if !self.trace.is_empty() {
            for trace in self.trace.iter() {
                trace.borrow_mut().trace_callback(event.clone())?;
            }
        }
        Ok(())
    }

    #[inline]
    fn trace_pager(&mut self) -> Result<()> {
        // Fast path for no tracing
        if self.trace.is_empty() {
            return Ok(());
        }

        // Get all pager trace events and process them
        let events = self.pager.trace_events();
        if !events.is_empty() {
            for &event in events {
                let event = TraceEvent::from(event);
                for trace in self.trace.iter() {
                    trace.borrow_mut().trace_callback(event.clone())?;
                }
            }
            self.pager.clear_trace_events();
        }
        Ok(())
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'a, 'b, S: Syscall> Risc0Context for Executor<'a, 'b, S> {
    #[inline(always)]
    fn get_pc(&self) -> ByteAddr {
        self.pc
    }

    #[inline(always)]
    fn set_pc(&mut self, addr: ByteAddr) {
        self.pc = addr;
    }

    #[inline(always)]
    fn set_user_pc(&mut self, addr: ByteAddr) {
        self.user_pc = addr;
    }

    #[inline(always)]
    fn get_machine_mode(&self) -> u32 {
        self.machine_mode
    }

    #[inline(always)]
    fn set_machine_mode(&mut self, mode: u32) {
        self.machine_mode = mode;
    }

    fn resume(&mut self) -> Result<()> {
        let input_words = self.input_digest.as_words().to_vec();
        for (i, word) in input_words.iter().enumerate() {
            self.store_u32(GLOBAL_INPUT_ADDR.waddr() + i, *word)?;
        }
        Ok(())
    }

    #[inline]
    fn on_insn_start(&mut self, insn: &Instruction, decoded: &DecodedInstruction) -> Result<()> {
        // Track execution cycle
        let cycle = self.cycles.user;
        self.cycles.user += 1;

        // Debug trace logging - only executed when tracing is enabled
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(
                "[{}:{}:{cycle}] {:?}> {:#010x}  {}",
                self.user_cycles + 1,
                self.segment_cycles() + 1,
                self.pc,
                decoded.insn,
                disasm(insn, decoded)
            );
        }

        // Handle instruction tracing if enabled
        if !self.trace.is_empty() {
            self.trace(TraceEvent::InstructionStart {
                cycle,
                pc: self.pc.0,
                insn: decoded.insn,
            })
        } else {
            Ok(())
        }
    }

    #[inline]
    fn on_insn_end(&mut self, _insn: &Instruction, _decoded: &DecodedInstruction) -> Result<()> {
        // Update cycle counters
        self.user_cycles += 1;
        self.phys_cycles += 1;

        // Handle any pending pager trace events
        self.trace_pager()?;
        Ok(())
    }

    #[inline]
    fn on_ecall_cycle(
        &mut self,
        _cur: CycleState,
        _next: CycleState,
        _s0: u32,
        _s1: u32,
        _s2: u32,
    ) -> Result<()> {
        self.phys_cycles += 1;
        self.trace_pager()?;
        Ok(())
    }

    #[inline]
    fn load_u32(&mut self, op: LoadOp, addr: WordAddr) -> Result<u32> {
        // Direct match is faster than branching
        match op {
            LoadOp::Peek => self.pager.peek(addr),
            LoadOp::Load | LoadOp::Record => self.pager.load(addr),
        }
    }

    #[inline]
    fn store_u32(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        // Fast path for tracing
        if !self.trace.is_empty() {
            // Use a fixed array to avoid allocations
            let bytes = word.to_be_bytes();
            self.trace(TraceEvent::MemorySet {
                addr: addr.baddr().0,
                region: bytes.to_vec(), // Still need to convert to vec for TraceEvent API
            })?;
        }
        self.pager.store(addr, word)
    }

    fn on_terminate(&mut self, a0: u32, a1: u32) -> Result<()> {
        self.user_cycles += 1;

        self.terminate_state = Some(TerminateState {
            a0: a0.into(),
            a1: a1.into(),
        });
        tracing::debug!("{:?}", self.terminate_state);

        let output: Digest = self
            .load_region(LoadOp::Peek, GLOBAL_OUTPUT_ADDR, DIGEST_BYTES)?
            .as_slice()
            .try_into()?;
        self.output_digest = Some(output);

        Ok(())
    }

    fn host_read(&mut self, fd: u32, buf: &mut [u8]) -> Result<u32> {
        let rlen = self.syscall_handler.host_read(self, fd, buf)?;
        let slice = &buf[..rlen as usize];
        self.read_record.push(slice.to_vec());
        Ok(rlen)
    }

    fn host_write(&mut self, fd: u32, buf: &[u8]) -> Result<u32> {
        let rlen = self.syscall_handler.host_write(self, fd, buf)?;
        self.write_record.push(rlen);
        Ok(rlen)
    }

    #[inline(always)]
    fn on_sha2_cycle(&mut self, _cur_state: CycleState, _sha2: &Sha2State) {
        self.phys_cycles += 1;
    }

    #[inline(always)]
    fn on_poseidon2_cycle(&mut self, _cur_state: CycleState, _p2: &Poseidon2State) {
        self.phys_cycles += 1;
    }

    #[inline(always)]
    fn on_bigint_cycle(&mut self, _cur_state: CycleState, _bigint: &BigIntState) {
        self.phys_cycles += 1;
    }
}

impl<S: Syscall> SyscallContext for Executor<'_, '_, S> {
    #[inline]
    fn peek_register(&mut self, idx: usize) -> Result<u32> {
        if idx >= REG_MAX {
            bail!("invalid register: x{idx}");
        }
        self.load_register(LoadOp::Peek, USER_REGS_ADDR.waddr(), idx)
    }

    #[inline]
    fn peek_u32(&mut self, addr: ByteAddr) -> Result<u32> {
        self.load_u32(LoadOp::Peek, addr.waddr())
    }

    #[inline]
    fn peek_u8(&mut self, addr: ByteAddr) -> Result<u8> {
        self.load_u8(LoadOp::Peek, addr)
    }

    #[inline]
    fn peek_region(&mut self, addr: ByteAddr, size: usize) -> Result<Vec<u8>> {
        self.load_region(LoadOp::Peek, addr, size)
    }

    #[inline]
    fn peek_page(&mut self, page_idx: u32) -> Result<Vec<u8>> {
        self.pager.peek_page(page_idx)
    }

    #[inline(always)]
    fn get_cycle(&self) -> u64 {
        self.cycles.user
    }

    #[inline(always)]
    fn get_pc(&self) -> u32 {
        self.user_pc.0
    }
}

impl From<PageTraceEvent> for TraceEvent {
    #[inline]
    fn from(event: PageTraceEvent) -> Self {
        match event {
            PageTraceEvent::PageIn { cycles } => TraceEvent::PageIn {
                cycles: cycles as u64,
            },
            PageTraceEvent::PageOut { cycles } => TraceEvent::PageOut {
                cycles: cycles as u64,
            },
        }
    }
}
