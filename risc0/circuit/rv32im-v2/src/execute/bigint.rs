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

use std::{collections::HashMap, io::Cursor};

use anyhow::{anyhow, bail, ensure, Result};
use derive_more::Debug;
use malachite::Natural;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;
use risc0_binfmt::WordAddr;

use super::{
    bibc::{self, BigIntIO},
    byte_poly::BytePolyProgram,
    platform::*,
    r0vm::{LoadOp, Risc0Context},
    CycleState,
};

pub(crate) const BIGINT_STATE_COUNT: usize = 21;
pub(crate) const BIGINT_ACCUM_STATE_COUNT: usize = 12;

/// BigInt width, in words, handled by the BigInt accelerator circuit.
pub(crate) const BIGINT_WIDTH_WORDS: usize = 4;

/// BigInt width, in bytes, handled by the BigInt accelerator circuit.
pub(crate) const BIGINT_WIDTH_BYTES: usize = 16; //BIGINT_WIDTH_WORDS * WORD_SIZE

pub(crate) type BigIntBytes = [u8; BIGINT_WIDTH_BYTES];
type BigIntWitness = HashMap<WordAddr, BigIntBytes>;

#[derive(Clone, Debug)]
pub(crate) struct BigIntState {
    pub is_ecall: bool,
    pub pc: WordAddr,
    pub poly_op: PolyOp,
    pub coeff: u32,
    pub bytes: BigIntBytes,
    pub next_state: CycleState,
}

struct BigInt {
    state: BigIntState,
    program: BytePolyProgram,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum PolyOp {
    Reset,
    Shift,
    SetTerm,
    AddTotal,
    Carry1,
    Carry2,
    EqZero,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum MemoryOp {
    Read,
    Write,
    Check,
}

#[derive(Debug)]
#[debug("{poly_op:?}({mem_op:?}, c:{coeff}, r:{reg}, o:{offset})")]
pub(crate) struct Instruction {
    pub poly_op: PolyOp,
    pub mem_op: MemoryOp,
    pub coeff: i32,
    pub reg: u32,
    pub offset: u32,
}

impl Instruction {
    // instruction encoding:
    // 3  2   2  2    1               0
    // 1  8   4  1    6               0
    // mmmmppppcccaaaaaoooooooooooooooo
    #[inline]
    pub fn decode(insn: u32) -> Result<Self> {
        // Extract all fields at once with bit masks
        let mem_op_bits = (insn >> 28) & 0x0f;
        let poly_op_bits = (insn >> 24) & 0x0f;
        let coeff_bits = (insn >> 21) & 0x07;
        let reg = (insn >> 16) & 0x1f;
        let offset = insn & 0xffff;

        // Convert to appropriate enum types
        let mem_op = MemoryOp::from_u32(mem_op_bits)
            .ok_or_else(|| anyhow!("Invalid mem_op in bigint program"))?;
        let poly_op = PolyOp::from_u32(poly_op_bits)
            .ok_or_else(|| anyhow!("Invalid poly_op in bigint program"))?;

        // The coefficient is offset by 4
        let coeff = coeff_bits as i32 - 4;

        Ok(Self {
            mem_op,
            poly_op,
            coeff,
            reg,
            offset,
        })
    }
}

impl BigInt {
    #[inline]
    fn run(&mut self, ctx: &mut dyn Risc0Context, witness: &BigIntWitness) -> Result<()> {
        ctx.on_bigint_cycle(CycleState::BigIntEcall, &self.state);
        // Hot loop - keep executing steps until we're done
        while self.state.next_state == CycleState::BigIntStep {
            self.step(ctx, witness)?;
        }
        Ok(())
    }

    #[inline]
    fn step(&mut self, ctx: &mut dyn Risc0Context, witness: &BigIntWitness) -> Result<()> {
        // Increment program counter
        self.state.pc.inc();

        // Fetch and decode instruction
        let insn_word = ctx.load_u32(LoadOp::Record, self.state.pc)?;
        let insn = Instruction::decode(insn_word)?;

        // Calculate effective address
        let base =
            ctx.load_aligned_addr_from_machine_register(LoadOp::Record, insn.reg as usize)?;
        let addr = base + insn.offset * BIGINT_WIDTH_WORDS as u32;

        tracing::trace!("step({:?}, {insn:?}, {addr:?})", self.state.pc);

        // Handle different memory operations based on instruction type
        if insn.mem_op == MemoryOp::Check && insn.poly_op != PolyOp::Reset {
            // Lazy computation of carry propagation
            if !self.program.in_carry {
                self.program.in_carry = true;
                self.program.total_carry = self.program.total.clone();

                // Pre-declare carry outside the loop to avoid multiple init/drops
                let mut carry = 0;

                // Optimize carry propagation (hot loop)
                for coeff in self.program.total_carry.coeffs.iter_mut() {
                    *coeff += carry;
                    // Error checking is rare, use ensure macro for cleaner code flow
                    ensure!(*coeff % 256 == 0, "bad carry");
                    *coeff /= 256;
                    carry = *coeff;
                }
                tracing::trace!("carry propagate complete");
            }

            // Pre-compute common base value
            const BASE_POINT: u32 = 128 * 256 * 64;

            // Pre-compute indices for better cache locality
            let offset = insn.offset as usize;
            let base_idx = offset * BIGINT_WIDTH_BYTES;

            // Single-path processing based on instruction type
            match insn.poly_op {
                PolyOp::Carry1 => {
                    for (i, ret) in self.state.bytes.iter_mut().enumerate() {
                        let coeff = self.program.total_carry.coeffs[base_idx + i] as u32;
                        let value = coeff.wrapping_add(BASE_POINT);
                        *ret = ((value >> 14) & 0xff) as u8;
                    }
                }
                PolyOp::Carry2 => {
                    for (i, ret) in self.state.bytes.iter_mut().enumerate() {
                        let coeff = self.program.total_carry.coeffs[base_idx + i] as u32;
                        let value = coeff.wrapping_add(BASE_POINT);
                        *ret = ((value >> 8) & 0x3f) as u8;
                    }
                }
                PolyOp::Shift | PolyOp::EqZero => {
                    for (i, ret) in self.state.bytes.iter_mut().enumerate() {
                        let coeff = self.program.total_carry.coeffs[base_idx + i] as u32;
                        let value = coeff.wrapping_add(BASE_POINT);
                        *ret = (value & 0xff) as u8;
                    }
                }
                _ => {
                    bail!("Invalid poly_op in bigint program")
                }
            }
        } else if insn.mem_op == MemoryOp::Read {
            // Load data from memory in batches
            for i in 0..BIGINT_WIDTH_WORDS {
                let word = ctx.load_u32(LoadOp::Record, addr + i)?;
                let bytes = word.to_le_bytes();
                // Direct copy for better performance
                let start_idx = i * WORD_SIZE;
                self.state.bytes[start_idx..start_idx + WORD_SIZE].copy_from_slice(&bytes);
            }
        } else if !addr.is_null() {
            // Get witness data if available
            self.state.bytes = *witness
                .get(&addr)
                .ok_or_else(|| anyhow!("Missing bigint witness: {addr:?}"))?;

            // Write data to memory if needed
            if insn.mem_op == MemoryOp::Write {
                // Use cast_slice for zero-copy conversion
                let words: &[u32] = bytemuck::cast_slice(&self.state.bytes);
                for (i, &word) in words.iter().enumerate() {
                    ctx.store_u32(addr + i, word)?;
                }
            }
        }

        // Execute program step
        self.program.step(&insn, &self.state.bytes)?;

        // Update state
        self.state.is_ecall = false;
        self.state.poly_op = insn.poly_op;
        self.state.coeff = (insn.coeff + 4) as u32;

        // Determine next state - use direct assignment rather than conditional expression
        if !self.state.is_ecall && insn.poly_op == PolyOp::Reset {
            self.state.next_state = CycleState::Decode;
        } else {
            self.state.next_state = CycleState::BigIntStep;
        }

        // Notify context of cycle completion
        ctx.on_bigint_cycle(CycleState::BigIntStep, &self.state);
        Ok(())
    }
}

struct BigIntIOImpl<'a> {
    ctx: &'a mut dyn Risc0Context,
    pub witness: BigIntWitness,
}

impl<'a> BigIntIOImpl<'a> {
    pub fn new(ctx: &'a mut dyn Risc0Context) -> Self {
        Self {
            ctx,
            witness: HashMap::new(),
        }
    }
}

#[inline]
fn bytes_le_to_bigint(bytes: &[u8]) -> Natural {
    // Pre-allocate with exact capacity needed
    let num_limbs = (bytes.len() + 3) / 4;
    let mut limbs = Vec::with_capacity(num_limbs);

    // Process in chunks of 4 bytes (u32 limbs)
    for chunk in bytes.chunks(4) {
        let mut arr = [0u8; 4];
        // Handle the last chunk (which may be less than 4 bytes)
        if chunk.len() == 4 {
            // Fast path for full chunks
            arr.copy_from_slice(chunk);
        } else {
            // Only partial chunk available
            arr[..chunk.len()].copy_from_slice(chunk);
        }
        limbs.push(u32::from_le_bytes(arr));
    }

    Natural::from_limbs_asc(&limbs)
}

#[inline]
fn bigint_to_bytes_le(value: &Natural) -> Vec<u8> {
    let limbs = value.to_limbs_asc();
    // Pre-allocate exactly the right size
    let mut out = Vec::with_capacity(limbs.len() * 4);

    // Convert each 32-bit limb to bytes
    for limb in limbs {
        out.extend_from_slice(&limb.to_le_bytes());
    }
    out
}

impl BigIntIO for BigIntIOImpl<'_> {
    #[inline]
    fn load(&mut self, arena: u32, offset: u32, count: u32) -> Result<Natural> {
        tracing::trace!("load(arena: {arena}, offset: {offset}, count: {count})");

        // Calculate effective address
        let base = self
            .ctx
            .load_aligned_addr_from_machine_register(LoadOp::Load, arena as usize)?;
        let addr = base + offset * BIGINT_WIDTH_WORDS as u32;

        // Load memory region as bytes
        let bytes = self
            .ctx
            .load_region(LoadOp::Load, addr.baddr(), count as usize)?;

        // Convert to Natural and return
        let val = bytes_le_to_bigint(&bytes);
        Ok(val)
    }

    #[inline]
    fn store(&mut self, arena: u32, offset: u32, count: u32, value: &Natural) -> Result<()> {
        // Calculate effective address
        let base = self
            .ctx
            .load_aligned_addr_from_machine_register(LoadOp::Load, arena as usize)?;
        let addr = base + offset * BIGINT_WIDTH_WORDS as u32;

        tracing::trace!("store(arena: {arena}, offset: {offset}, count: {count}, addr: {addr:?}, value: {value})");

        // Convert Natural to bytes
        let bytes = bigint_to_bytes_le(value);

        // Pre-allocate with zeros and copy value bytes
        let mut witness = vec![0u8; count as usize];
        witness[..bytes.len()].copy_from_slice(&bytes);

        // Chunk into fixed-size pieces and store in witness map
        let chunk_count = count as usize / BIGINT_WIDTH_BYTES;
        let chunks = witness.chunks_exact(BIGINT_WIDTH_BYTES);

        // Verify chunk count matches expected
        debug_assert_eq!(chunks.len(), chunk_count, "Incorrect chunk count");

        // Store each chunk in the witness map
        for (i, chunk) in chunks.enumerate() {
            let chunk_addr = addr + i * BIGINT_WIDTH_WORDS;
            // Use try_into() directly with unwrap since we know the size is correct
            let chunk_bytes: BigIntBytes = chunk.try_into().unwrap();
            self.witness.insert(chunk_addr, chunk_bytes);
        }

        Ok(())
    }
}

pub fn ecall(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::debug!("ecall");

    // Load all register values at once to minimize context switches
    let blob_ptr = ctx.load_aligned_addr_from_machine_register(LoadOp::Load, REG_A0)?;
    let nondet_program_ptr = ctx.load_aligned_addr_from_machine_register(LoadOp::Load, REG_T1)?;
    let verify_program_ptr =
        ctx.load_aligned_addr_from_machine_register(LoadOp::Record, REG_T2)? - 1;
    let consts_ptr = ctx.load_aligned_addr_from_machine_register(LoadOp::Load, REG_T3)?;

    // Load program sizes from blob
    let nondet_program_size = ctx.load_u32(LoadOp::Load, blob_ptr)?;
    let verify_program_size = ctx.load_u32(LoadOp::Load, blob_ptr + 1)?;
    let consts_size = ctx.load_u32(LoadOp::Load, blob_ptr + 2)?;

    // Log debug information
    tracing::debug!("blob_ptr: {blob_ptr:?}");
    tracing::debug!(
        "nondet_program_ptr: {nondet_program_ptr:?}, nondet_program_size: {nondet_program_size}"
    );

    // Load non-deterministic program
    let program_bytes_size = nondet_program_size as usize * WORD_SIZE;
    let program_bytes =
        ctx.load_region(LoadOp::Load, nondet_program_ptr.baddr(), program_bytes_size)?;
    tracing::debug!("program_bytes: {}", program_bytes.len());

    // Decode program
    let mut cursor = Cursor::new(program_bytes);
    let program = bibc::Program::decode(&mut cursor)?;

    // Evaluate program and collect witness data
    let witness = {
        let mut io = BigIntIOImpl::new(ctx);
        program.eval(&mut io)?;
        std::mem::take(&mut io.witness)
    };

    // Load verification program and constants
    ctx.load_region(
        LoadOp::Load,
        verify_program_ptr.baddr(),
        verify_program_size as usize * WORD_SIZE,
    )?;

    ctx.load_region(
        LoadOp::Load,
        consts_ptr.baddr(),
        consts_size as usize * WORD_SIZE,
    )?;

    // Initialize BigInt state and program
    let state = BigIntState {
        is_ecall: true,
        pc: verify_program_ptr,
        poly_op: PolyOp::Reset,
        coeff: 0,
        bytes: Default::default(),
        next_state: CycleState::BigIntStep,
    };

    // Create and run the BigInt program
    let mut bigint = BigInt {
        state,
        program: BytePolyProgram::new(),
    };

    // Execute program with witness data
    bigint.run(ctx, &witness)
}
