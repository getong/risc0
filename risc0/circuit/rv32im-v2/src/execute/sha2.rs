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
use risc0_zkp::core::hash::sha::BLOCK_WORDS;

use crate::execute::{
    platform::*,
    r0vm::{guest_addr, LoadOp, Risc0Context},
};

// Constants used in SHA2 implementation
#[allow(dead_code)]
const SHA2_LOAD_STATE_CYCLES: u32 = 4;
const SHA2_LOAD_DATA_CYCLES: u32 = BLOCK_WORDS as u32; // BLOCK_WORDS = 16
const SHA2_MIX_CYCLES: u32 = 48;
const SHA2_STORE_CYCLES: u32 = 4;
// Pre-computed value: 4 + 16 + 48 = 68
const SHA2_BACK: usize = 68;

#[derive(Clone, Debug)]
pub(crate) struct Sha2State {
    pub state_in_addr: WordAddr,
    pub state_out_addr: WordAddr,
    pub data_addr: WordAddr,
    pub count: u32,
    pub k_addr: WordAddr,
    pub round: u32,
    pub next_state: CycleState,
    pub a: u32,
    pub e: u32,
    pub w: u32,
}

impl Sha2State {
    #[inline(always)]
    fn step(
        &mut self,
        ctx: &mut dyn Risc0Context,
        cur_state: &mut CycleState,
        next_state: CycleState,
    ) {
        self.next_state = next_state;
        ctx.on_sha2_cycle(*cur_state, self);
        *cur_state = next_state;
    }
}

pub fn ecall(ctx: &mut dyn Risc0Context) -> Result<()> {
    // Load all arguments at once
    let reg_a0 = ctx.load_machine_register(LoadOp::Record, REG_A0)?;
    let reg_a1 = ctx.load_machine_register(LoadOp::Record, REG_A1)?;
    let reg_a2 = ctx.load_machine_register(LoadOp::Record, REG_A2)?;
    let count = ctx.load_machine_register(LoadOp::Record, REG_A3)? & 0xffff;
    let reg_a4 = ctx.load_machine_register(LoadOp::Record, REG_A4)?;
    
    // Process guest addresses after validating count to avoid wasting cycles on invalid calls
    if count > MAX_SHA_COUNT {
        bail!("Invalid count (too big) in sha2 ecall: {count}");
    }
    
    let state_in_addr = guest_addr(reg_a0)?.waddr();
    let state_out_addr = guest_addr(reg_a1)?.waddr();
    let data_addr = guest_addr(reg_a2)?.waddr();
    let k_addr = guest_addr(reg_a4)?.waddr();
    tracing::trace!("sha2: {count} blocks");

    let mut sha2 = Sha2State {
        state_in_addr,
        state_out_addr,
        data_addr,
        count,
        k_addr,
        round: 0,
        next_state: CycleState::ShaEcall,
        a: 0,
        e: 0,
        w: 0,
    };

    let mut cur_state = CycleState::ShaEcall;
    let mut old_a = RingBuffer::<SHA2_BACK>::new();
    let mut old_e = RingBuffer::<SHA2_BACK>::new();
    let mut old_w = RingBuffer::<BLOCK_WORDS>::new();

    // Unroll small fixed loop for better performance
    // i = 0
    sha2.round = 0;
    sha2.step(ctx, &mut cur_state, CycleState::ShaLoadState);
    let a = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 3u32)?;
    let e = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 7u32)?;
    sha2.a = a.to_be();
    sha2.e = e.to_be();
    old_a.push(sha2.a);
    old_e.push(sha2.e);
    ctx.store_u32(sha2.state_out_addr + 3u32, a)?;
    ctx.store_u32(sha2.state_out_addr + 7u32, e)?;
    
    // i = 1
    sha2.round = 1;
    sha2.step(ctx, &mut cur_state, CycleState::ShaLoadState);
    let a = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 2u32)?;
    let e = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 6u32)?;
    sha2.a = a.to_be();
    sha2.e = e.to_be();
    old_a.push(sha2.a);
    old_e.push(sha2.e);
    ctx.store_u32(sha2.state_out_addr + 2u32, a)?;
    ctx.store_u32(sha2.state_out_addr + 6u32, e)?;
    
    // i = 2
    sha2.round = 2;
    sha2.step(ctx, &mut cur_state, CycleState::ShaLoadState);
    let a = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 1u32)?;
    let e = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 5u32)?;
    sha2.a = a.to_be();
    sha2.e = e.to_be();
    old_a.push(sha2.a);
    old_e.push(sha2.e);
    ctx.store_u32(sha2.state_out_addr + 1u32, a)?;
    ctx.store_u32(sha2.state_out_addr + 5u32, e)?;
    
    // i = 3
    sha2.round = 3;
    sha2.step(ctx, &mut cur_state, CycleState::ShaLoadState);
    let a = ctx.load_u32(LoadOp::Record, sha2.state_in_addr)?;
    let e = ctx.load_u32(LoadOp::Record, sha2.state_in_addr + 4u32)?;
    sha2.a = a.to_be();
    sha2.e = e.to_be();
    old_a.push(sha2.a);
    old_e.push(sha2.e);
    ctx.store_u32(sha2.state_out_addr, a)?;
    ctx.store_u32(sha2.state_out_addr + 4u32, e)?;

    // HERE!
    while sha2.count != 0 {
        for i in 0..SHA2_LOAD_DATA_CYCLES {
            sha2.round = i;
            sha2.step(ctx, &mut cur_state, CycleState::ShaLoadData);
            let k = ctx.load_u32(LoadOp::Record, sha2.k_addr + i)?;
            sha2.w = ctx.load_u32(LoadOp::Record, sha2.data_addr)?.to_be();
            sha2.data_addr += 1u32;
            old_w.push(sha2.w);
            let (a, e) = compute_ae(&old_a, &old_e, k, sha2.w);
            sha2.a = a;
            sha2.e = e;
            old_a.push(a);
            old_e.push(e);
        }

        for i in 0..SHA2_MIX_CYCLES {
            sha2.round = i;
            sha2.step(ctx, &mut cur_state, CycleState::ShaMix);
            let k = ctx.load_u32(LoadOp::Record, sha2.k_addr + BLOCK_WORDS + i)?;
            sha2.w = compute_w(&old_w);
            old_w.push(sha2.w);
            let (a, e) = compute_ae(&old_a, &old_e, k, sha2.w);
            sha2.a = a;
            sha2.e = e;
            old_a.push(a);
            old_e.push(e);
        }

        for i in 0..SHA2_STORE_CYCLES {
            sha2.round = i;
            sha2.step(ctx, &mut cur_state, CycleState::ShaStoreState);
            sha2.a = old_a.back(4).wrapping_add(old_a.back(SHA2_BACK));
            sha2.e = old_e.back(4).wrapping_add(old_e.back(SHA2_BACK));
            sha2.w = 0;
            if i == 3 {
                sha2.count -= 1;
            }
            old_a.push(sha2.a);
            old_e.push(sha2.e);
            ctx.store_u32(sha2.state_out_addr + 3u32 - i, sha2.a.to_be())?;
            ctx.store_u32(sha2.state_out_addr + 7u32 - i, sha2.e.to_be())?;
        }
    }

    sha2.round = 0;
    sha2.step(ctx, &mut cur_state, CycleState::Decode);

    Ok(())
}

#[inline]
fn compute_ae(
    old_a: &RingBuffer<SHA2_BACK>,
    old_e: &RingBuffer<SHA2_BACK>,
    k: u32,
    w: u32,
) -> (u32, u32) {
    // Define helper functions as inlined functions rather than macros for better optimization
    #[inline(always)]
    fn ch(x: u32, y: u32, z: u32) -> u32 {
        (x & y) ^ (!x & z)
    }

    #[inline(always)]
    fn maj(x: u32, y: u32, z: u32) -> u32 {
        (x & y) ^ (x & z) ^ (y & z)
    }

    #[inline(always)]
    fn epsilon0(x: u32) -> u32 {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    }

    #[inline(always)]
    fn epsilon1(x: u32) -> u32 {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    }

    // Pre-load all values needed to reduce repeated array lookups
    let a = old_a.back(1);
    let b = old_a.back(2);
    let c = old_a.back(3);
    let d = old_a.back(4);
    let e = old_e.back(1);
    let f = old_e.back(2);
    let g = old_e.back(3);
    let h = old_e.back(4);

    // Calculate intermediate values
    let e1 = epsilon1(e);
    let ch_efg = ch(e, f, g);
    let e0 = epsilon0(a);
    let maj_abc = maj(a, b, c);
    
    // Compute final values using wrapping arithmetic
    let t1 = h.wrapping_add(e1).wrapping_add(ch_efg).wrapping_add(k).wrapping_add(w);
    let t2 = e0.wrapping_add(maj_abc);
    let new_e = d.wrapping_add(t1);
    let new_a = t1.wrapping_add(t2);
    
    (new_a, new_e)
}

#[inline]
fn compute_w(old_w: &RingBuffer<16>) -> u32 {
    // Define helper functions as inlined functions rather than macros for better optimization
    #[inline(always)]
    fn sigma0(x: u32) -> u32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }

    #[inline(always)]
    fn sigma1(x: u32) -> u32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }

    // Pre-load values to avoid repeated lookups
    let w2 = old_w.back(2);
    let w7 = old_w.back(7);
    let w15 = old_w.back(15);
    let w16 = old_w.back(16);
    
    // Compute using wrapping arithmetic
    sigma1(w2).wrapping_add(w7).wrapping_add(sigma0(w15)).wrapping_add(w16)
}

struct RingBuffer<const N: usize> {
    buf: [u32; N],
    cur: usize,
}

impl<const N: usize> RingBuffer<N> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            buf: [0; N],
            cur: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, value: u32) {
        self.buf[self.cur] = value;
        self.cur = (self.cur + 1) % N;
    }

    #[inline(always)]
    fn back(&self, i: usize) -> u32 {
        // This is a hot path in the SHA2 implementation
        self.buf[(N + self.cur - i) % N]
    }
}
