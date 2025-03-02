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

use anyhow::{bail, Result};
use risc0_binfmt::WordAddr;
use risc0_zkp::{
    core::{
        digest::DIGEST_WORDS,
        hash::poseidon2::{
            CELLS, M_INT_DIAG_HZN, ROUNDS_HALF_FULL, ROUNDS_PARTIAL, ROUND_CONSTANTS,
        },
    },
    field::baby_bear::{self},
};

use crate::{
    execute::{
        platform::*,
        r0vm::{LoadOp, Risc0Context},
    },
    zirgen::circuit::ExtVal,
};

const BABY_BEAR_P_U32: u32 = baby_bear::P;
const BABY_BEAR_P_U64: u64 = baby_bear::P as u64;

#[derive(Clone, Debug, Default)]
pub(crate) struct Poseidon2State {
    pub has_state: u32,
    pub state_addr: u32,
    pub buf_out_addr: u32,
    pub is_elem: u32,
    pub check_out: u32,
    pub load_tx_type: u32,
    pub next_state: CycleState,
    pub sub_state: u32,
    pub buf_in_addr: u32,
    pub count: u32,
    pub mode: u32,
    pub inner: [u32; CELLS],
    pub zcheck: ExtVal,
}

impl Poseidon2State {
    #[inline]
    fn new_ecall(state_addr: u32, buf_in_addr: u32, buf_out_addr: u32, bits_count: u32) -> Self {
        // Extract flags using bitwise operations
        let is_elem = bits_count & PFLAG_IS_ELEM;
        let check_out = bits_count & PFLAG_CHECK_OUT;
        
        Self {
            state_addr,
            buf_in_addr,
            buf_out_addr,
            // Use direct binary conversion instead of conditional
            has_state: (state_addr != 0) as u32,
            is_elem: (is_elem != 0) as u32,
            check_out: (check_out != 0) as u32,
            count: bits_count & 0xffff,
            mode: 1,
            load_tx_type: tx::READ,
            next_state: CycleState::PoseidonEntry,
            ..Default::default()
        }
    }

    #[inline(always)]
    fn step(
        &mut self,
        ctx: &mut dyn Risc0Context,
        cur_state: &mut CycleState,
        next_state: CycleState,
        sub_state: u32,
    ) {
        self.next_state = next_state;
        self.sub_state = sub_state;
        ctx.on_poseidon2_cycle(*cur_state, self);
        *cur_state = next_state;
    }

    #[inline]
    pub(crate) fn rest(
        &mut self,
        ctx: &mut dyn Risc0Context,
        final_state: CycleState,
    ) -> Result<()> {
        let mut cur_state = self.next_state;
        let state_addr = WordAddr(self.state_addr);

        // Fast path for loading state
        if self.has_state == 1 {
            self.step(ctx, &mut cur_state, CycleState::PoseidonLoadState, 0);
            // Load state in one contiguous block
            let state_offset = DIGEST_WORDS * 2;
            for i in 0..DIGEST_WORDS {
                self.inner[state_offset + i] = ctx.load_u32(LoadOp::Record, state_addr + i)?;
            }
        }

        // While we have data to process - hot path
        let mut buf_in_addr = WordAddr(self.buf_in_addr);
        
        while self.count > 0 {
            // Signal load state
            self.step(ctx, &mut cur_state, CycleState::PoseidonLoadIn, 0);

            // Use branch prediction hint for common path
            if self.is_elem != 0 {
                // Element mode - load two blocks of DIGEST_WORDS
                // First block
                for i in 0..DIGEST_WORDS {
                    self.inner[i] = ctx.load_u32(LoadOp::Record, buf_in_addr.postfix_inc())?;
                }
                self.buf_in_addr = buf_in_addr.0;
                
                // Process first block
                self.step(ctx, &mut cur_state, CycleState::PoseidonLoadIn, 1);
                
                // Second block
                let offset = DIGEST_WORDS;
                for i in 0..DIGEST_WORDS {
                    self.inner[offset + i] = ctx.load_u32(LoadOp::Record, buf_in_addr.postfix_inc())?;
                }
                self.buf_in_addr = buf_in_addr.0;
            } else {
                // Packed mode - each word contains two elements
                for i in 0..DIGEST_WORDS {
                    let word = ctx.load_u32(LoadOp::Record, buf_in_addr.postfix_inc())?;
                    // Extract low and high 16 bits efficiently
                    self.inner[2 * i] = word & 0xffff;
                    self.inner[2 * i + 1] = word >> 16;
                }
                self.buf_in_addr = buf_in_addr.0;
            }

            // Apply Poseidon2 mixing operations
            self.multiply_by_m_ext();
            
            // First half of full rounds
            for i in 0..ROUNDS_HALF_FULL {
                self.step(ctx, &mut cur_state, CycleState::PoseidonExtRound, i as u32);
                self.do_ext_round(i);
            }
            
            // Partial rounds
            self.step(ctx, &mut cur_state, CycleState::PoseidonIntRound, 0);
            self.do_int_rounds();
            
            // Second half of full rounds
            let second_half_start = ROUNDS_HALF_FULL;
            let second_half_end = ROUNDS_HALF_FULL * 2;
            for i in second_half_start..second_half_end {
                self.step(ctx, &mut cur_state, CycleState::PoseidonExtRound, i as u32);
                self.do_ext_round(i);
            }
            
            // Decrement counter
            self.count -= 1;
        }

        // Process output
        self.step(ctx, &mut cur_state, CycleState::PoseidonDoOut, 0);
        let buf_out_addr = WordAddr(self.buf_out_addr);
        
        if self.check_out != 0 {
            // Verification mode - check if output matches expected values
            for i in 0..DIGEST_WORDS {
                let addr = buf_out_addr + i;
                let word = ctx.load_u32(LoadOp::Record, addr)?;
                let cell = self.inner[i];
                
                // Fast fail path
                if word != cell {
                    tracing::warn!(
                        "buf_in_addr: {:?}, buf_out_addr: {buf_out_addr:?}, cell: {i}",
                        WordAddr(self.buf_in_addr)
                    );
                    bail!("poseidon2 check failed: {word:#010x} != {cell:#010x}");
                }
            }
        } else {
            // Output mode - store hash result
            for i in 0..DIGEST_WORDS {
                ctx.store_u32(buf_out_addr + i, self.inner[i])?;
            }
        }

        // Clear input buffer address
        self.buf_in_addr = 0;

        // Store state if needed
        if self.has_state == 1 {
            self.step(ctx, &mut cur_state, CycleState::PoseidonStoreState, 0);
            let state_offset = DIGEST_WORDS * 2;
            for i in 0..DIGEST_WORDS {
                ctx.store_u32(state_addr + i, self.inner[state_offset + i])?;
            }
        }

        // Transition to final state
        self.step(ctx, &mut cur_state, final_state, 0);

        Ok(())
    }

    // Reverted to original implementation for correctness
    fn multiply_by_m_ext(&mut self) {
        let mut out = [0; CELLS];
        let mut tmp_sums = [0; 4];

        for i in 0..CELLS / 4 {
            let chunk = multiply_by_4x4_circulant(&[
                self.inner[i * 4],
                self.inner[i * 4 + 1],
                self.inner[i * 4 + 2],
                self.inner[i * 4 + 3],
            ]);
            for j in 0..4 {
                let to_add = chunk[j] as u64;
                let to_add = (to_add % BABY_BEAR_P_U64) as u32;
                tmp_sums[j] += to_add;
                tmp_sums[j] %= BABY_BEAR_P_U32;
                out[i * 4 + j] += to_add;
                out[i * 4 + j] %= BABY_BEAR_P_U32;
            }
        }
        for i in 0..CELLS {
            self.inner[i] = (out[i] + tmp_sums[i % 4]) % BABY_BEAR_P_U32;
        }
    }

    // Exploit the fact that off-diagonal entries of M_INT are all 1.
    fn multiply_by_m_int(&mut self) {
        let mut sum = 0u64;
        for i in 0..CELLS {
            sum += self.inner[i] as u64;
        }
        sum %= BABY_BEAR_P_U64;
        
        for (i, diag) in M_INT_DIAG_HZN.iter().enumerate().take(CELLS) {
            let diag = diag.as_u32() as u64;
            let cell = self.inner[i] as u64;
            self.inner[i] = ((sum + diag * cell) % BABY_BEAR_P_U64) as u32;
        }
    }

    #[inline]
    fn do_ext_round(&mut self, mut idx: usize) {
        // Adjust index if in second half of full rounds
        if idx >= ROUNDS_HALF_FULL {
            idx += ROUNDS_PARTIAL;
        }

        // Add round constants then apply S-box
        self.add_round_constants_full(idx);
        
        // Apply the power map S-box to all cells
        for i in 0..CELLS {
            self.inner[i] = sbox2(self.inner[i]);
        }

        // Mix the state
        self.multiply_by_m_ext();
    }

    #[inline]
    fn do_int_rounds(&mut self) {
        // Rounds where we only apply the S-box to the first element
        let base_round = ROUNDS_HALF_FULL;
        
        for i in 0..ROUNDS_PARTIAL {
            // Add round constant to first element
            self.add_round_constants_partial(base_round + i);
            
            // Apply S-box to only the first element
            self.inner[0] = sbox2(self.inner[0]);
            
            // Apply internal mixing function
            self.multiply_by_m_int();
        }
    }

    #[inline]
    fn add_round_constants_full(&mut self, round: usize) {
        // Calculate base index for constants in this round
        let base_idx = round * CELLS;
        
        // Add constants to all cells
        for i in 0..CELLS {
            let constant = ROUND_CONSTANTS[base_idx + i].as_u32();
            self.inner[i] = (self.inner[i] + constant) % BABY_BEAR_P_U32;
        }
    }

    #[inline]
    fn add_round_constants_partial(&mut self, round: usize) {
        // Only add constant to first element
        let constant = ROUND_CONSTANTS[round * CELLS].as_u32();
        self.inner[0] = (self.inner[0] + constant) % BABY_BEAR_P_U32;
    }
}

fn multiply_by_4x4_circulant(x: &[u32; 4]) -> [u32; 4] {
    // See appendix B of Poseidon2 paper.
    const CIRC_FACTOR_2: u64 = 2;
    const CIRC_FACTOR_4: u64 = 4;
    let t0 = (x[0] as u64 + x[1] as u64) % BABY_BEAR_P_U64;
    let t1 = (x[2] as u64 + x[3] as u64) % BABY_BEAR_P_U64;
    let t2 = (CIRC_FACTOR_2 * x[1] as u64 + t1) % BABY_BEAR_P_U64;
    let t3 = (CIRC_FACTOR_2 * x[3] as u64 + t0) % BABY_BEAR_P_U64;
    let t4 = (CIRC_FACTOR_4 * t1 + t3) % BABY_BEAR_P_U64;
    let t5 = (CIRC_FACTOR_4 * t0 + t2) % BABY_BEAR_P_U64;
    let t6 = (t3 + t5) % BABY_BEAR_P_U64;
    let t7 = (t2 + t4) % BABY_BEAR_P_U64;
    [t6 as u32, t5 as u32, t7 as u32, t4 as u32]
}

fn sbox2(x: u32) -> u32 {
    let x = x as u64;
    let x2 = (x * x) % BABY_BEAR_P_U64;
    let x4 = (x2 * x2) % BABY_BEAR_P_U64;
    let x6 = (x4 * x2) % BABY_BEAR_P_U64;
    let x7 = (x6 * x) % BABY_BEAR_P_U64;
    x7 as u32
}

pub(crate) struct Poseidon2;

impl Poseidon2 {
    #[inline]
    pub fn ecall(ctx: &mut dyn Risc0Context) -> Result<()> {
        tracing::trace!("poseidon2 ecall");
        
        // Load all registers in a batch
        let state_addr = ctx.load_machine_register(LoadOp::Record, REG_A0)?;
        let buf_in_addr = ctx.load_machine_register(LoadOp::Record, REG_A1)?;
        let buf_out_addr = ctx.load_machine_register(LoadOp::Record, REG_A2)?;
        let bits_count = ctx.load_machine_register(LoadOp::Record, REG_A3)?;
        
        // Initialize the Poseidon2 state
        let mut p2 = Poseidon2State::new_ecall(state_addr, buf_in_addr, buf_out_addr, bits_count);
        
        // Execute the hash operation
        p2.rest(ctx, CycleState::Decode)
    }
}
