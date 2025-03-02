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

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use derive_more::Debug;
use risc0_binfmt::{MemoryImage2, Page, WordAddr};
use risc0_zkp::core::digest::Digest;

use super::{node_idx, platform::*};

pub const PAGE_WORDS: usize = PAGE_BYTES / WORD_SIZE;

// Pre-calculated constants for cycle counts - direct values for better performance
const PAGE_CYCLES: u32 = 322; // Original: 1 + 10 * 32 + 1 = 322
const NODE_CYCLES: u32 = 13;  // Original: 1 + 2 + 8 + 1 + 1 = 13

// POSEIDON constants (kept for documentation)
#[allow(dead_code)]
pub(crate) const POSEIDON_PAGE_ROUNDS: u32 = 32; // Pre-calculated: 256 / 8 = 32

// Pre-calculated: 1 + 1 + 1 + 2 + 2 + 1 + 1 + 1 = 10
pub(crate) const RESERVED_PAGING_CYCLES: u32 = 10;

const INVALID_IDX: u32 = u32::MAX;
// Pre-calculated based on memory size and page size
const NUM_PAGES: usize = MEMORY_PAGES;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) enum PageState {
    Unloaded,
    Loaded,
    Dirty,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PageTraceEvent {
    PageIn { cycles: u32 },
    PageOut { cycles: u32 },
}

#[derive(Debug)]
pub(crate) struct PagedMemory {
    pub image: MemoryImage2,
    #[debug(skip)]
    page_table: Vec<u32>,
    #[debug(skip)]
    page_cache: Vec<Page>,
    #[debug("{page_states:#x?}")]
    pub(crate) page_states: BTreeMap<u32, PageState>,
    pub cycles: u32,
    user_registers: [u32; REG_MAX],
    machine_registers: [u32; REG_MAX],
    tracing_enabled: bool,
    trace_events: Vec<PageTraceEvent>,
}

impl PagedMemory {
    pub(crate) fn new(mut image: MemoryImage2, tracing_enabled: bool) -> Self {
        let mut machine_registers = [0; REG_MAX];
        let mut user_registers = [0; REG_MAX];
        let page_idx = MACHINE_REGS_ADDR.waddr().page_idx();
        let page = image.get_page(page_idx).unwrap();
        for idx in 0..REG_MAX {
            machine_registers[idx] = page.load(MACHINE_REGS_ADDR.waddr() + idx);
            user_registers[idx] = page.load(USER_REGS_ADDR.waddr() + idx);
        }

        Self {
            image,
            page_table: vec![INVALID_IDX; NUM_PAGES],
            page_cache: Vec::new(),
            page_states: BTreeMap::new(),
            cycles: RESERVED_PAGING_CYCLES,
            user_registers,
            machine_registers,
            tracing_enabled,
            trace_events: vec![],
        }
    }

    pub(crate) fn reset(&mut self) {
        self.page_table.fill(INVALID_IDX);
        self.page_cache.clear();
        self.page_states.clear();
        self.cycles = RESERVED_PAGING_CYCLES;
    }

    #[inline]
    fn try_load_register(&self, addr: WordAddr) -> Option<u32> {
        // Register files have fixed addresses, so optimize for direct access
        if addr >= USER_REGS_ADDR.waddr() && addr < USER_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = (addr.0 - USER_REGS_ADDR.waddr().0) as usize;
            Some(self.user_registers[reg_idx])
        } else if addr >= MACHINE_REGS_ADDR.waddr() && addr < MACHINE_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = (addr.0 - MACHINE_REGS_ADDR.waddr().0) as usize;
            Some(self.machine_registers[reg_idx])
        } else {
            None
        }
    }

    #[inline]
    fn try_store_register(&mut self, addr: WordAddr, word: u32) -> bool {
        // Register files have fixed addresses, so optimize for direct access
        if addr >= USER_REGS_ADDR.waddr() && addr < USER_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = (addr.0 - USER_REGS_ADDR.waddr().0) as usize;
            self.user_registers[reg_idx] = word;
            true
        } else if addr >= MACHINE_REGS_ADDR.waddr() && addr < MACHINE_REGS_ADDR.waddr() + REG_MAX {
            let reg_idx = (addr.0 - MACHINE_REGS_ADDR.waddr().0) as usize;
            self.machine_registers[reg_idx] = word;
            true
        } else {
            false
        }
    }

    #[inline]
    fn peek_ram(&mut self, addr: WordAddr) -> Result<u32> {
        let page_idx = addr.page_idx();
        let cache_idx = self.page_table[page_idx as usize];
        if cache_idx == INVALID_IDX {
            // Unloaded, peek into image
            Ok(self.image.get_page(page_idx)?.load(addr))
        } else {
            // Loaded, get from cache - this is the common case
            Ok(self.page_cache[cache_idx as usize].load(addr))
        }
    }

    #[inline]
    pub(crate) fn peek(&mut self, addr: WordAddr) -> Result<u32> {
        if addr >= MEMORY_END_ADDR {
            bail!("Invalid peek address: {addr:?}");
        }

        // Check registers first as it's a fast path
        if let Some(word) = self.try_load_register(addr) {
            return Ok(word);
        }
        // Fall back to RAM access
        self.peek_ram(addr)
    }

    pub(crate) fn peek_page(&mut self, page_idx: u32) -> Result<Vec<u8>> {
        let cache_idx = self.page_table[page_idx as usize];
        if cache_idx == INVALID_IDX {
            // Unloaded, peek into image
            Ok(self.image.get_page(page_idx)?.0.clone())
        } else {
            // Loaded, get from cache
            Ok(self.page_cache[cache_idx as usize].0.clone())
        }
    }

    fn load_ram(&mut self, addr: WordAddr) -> Result<u32> {
        let page_idx = addr.page_idx();
        let node_idx = node_idx(page_idx);
        // tracing::trace!("load: {addr:?}, page: {page_idx:#08x}, node: {node_idx:#08x}");
        let mut cache_idx = self.page_table[page_idx as usize];
        if cache_idx == INVALID_IDX {
            self.load_page(page_idx)?;
            self.page_states.insert(node_idx, PageState::Loaded);
            cache_idx = self.page_table[page_idx as usize];
        }
        Ok(self.page_cache[cache_idx as usize].load(addr))
    }

    #[inline]
    pub(crate) fn load(&mut self, addr: WordAddr) -> Result<u32> {
        if addr >= MEMORY_END_ADDR {
            bail!("Invalid load address: {addr:?}");
        }

        // Check registers first as it's a fast path
        if let Some(word) = self.try_load_register(addr) {
            return Ok(word);
        }
        // Fall back to RAM access
        self.load_ram(addr)
    }

    fn store_ram(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        // tracing::trace!("store: {addr:?}, page: {page_idx:#08x}, word: {word:#010x}");
        let page_idx = addr.page_idx();
        let page = self.page_for_writing(page_idx)?;
        page.store(addr, word);
        Ok(())
    }

    #[inline]
    pub(crate) fn store(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        if addr >= MEMORY_END_ADDR {
            bail!("Invalid store address: {addr:?}");
        }

        // Check register access first as it's a fast path
        if self.try_store_register(addr, word) {
            return Ok(());
        }
        // Fall back to RAM access
        self.store_ram(addr, word)
    }

    #[inline]
    fn page_for_writing(&mut self, page_idx: u32) -> Result<&mut Page> {
        let node_idx = node_idx(page_idx);
        // Fast path: Use direct access for improved performance
        let state = match self.page_states.get(&node_idx) {
            Some(&state) => state,
            None => {
                self.load_page(page_idx)?;
                PageState::Loaded
            }
        };
        
        // Handle state transitions - only perform expensive operations if needed
        if state == PageState::Loaded {
            self.cycles += PAGE_CYCLES;
            self.trace_page_out(PAGE_CYCLES);
            self.fixup_costs(node_idx, PageState::Dirty);
            self.page_states.insert(node_idx, PageState::Dirty);
        }
        
        // Direct indexing is faster than get_mut+unwrap
        let cache_idx = self.page_table[page_idx as usize] as usize;
        // SAFETY: Cache index is guaranteed to be valid by the page loading logic
        Ok(&mut self.page_cache[cache_idx])
    }

    fn write_registers(&mut self) {
        // Copy register values first to avoid borrow conflicts
        let user_registers = self.user_registers;
        let machine_registers = self.machine_registers;
        // This works because we can assume that user and machine register files
        // live in the same page.
        let page_idx = MACHINE_REGS_ADDR.waddr().page_idx();
        let page = self.page_for_writing(page_idx).unwrap();
        for idx in 0..REG_MAX {
            page.store(MACHINE_REGS_ADDR.waddr() + idx, machine_registers[idx]);
            page.store(USER_REGS_ADDR.waddr() + idx, user_registers[idx]);
        }
    }

    pub(crate) fn commit(&mut self) -> Result<(Digest, MemoryImage2, Digest)> {
        // tracing::trace!("commit: {self:#?}");

        self.write_registers();

        let pre_state = self.image.image_id();
        let mut image = MemoryImage2::default();

        // Gather the original pages
        for (&node_idx, &page_state) in self.page_states.iter() {
            if node_idx < MEMORY_PAGES as u32 {
                continue;
            }
            let page_idx = page_idx(node_idx);
            tracing::trace!("commit: {page_idx:#08x}, state: {page_state:?}");

            // Copy original state of all pages accessed in this segment.
            image.set_page(page_idx, self.image.get_page(page_idx)?);

            // Update dirty pages into the image that accumulates over a session.
            if page_state == PageState::Dirty {
                let cache_idx = self.page_table[page_idx as usize] as usize;
                let page = &self.page_cache[cache_idx];
                self.image.set_page(page_idx, page.clone());
            }
        }

        // Add minimal needed 'uncles'
        for &node_idx in self.page_states.keys() {
            // If this is a leaf, break
            if node_idx >= MEMORY_PAGES as u32 {
                break;
            }

            let lhs_idx = node_idx * 2;
            let rhs_idx = node_idx * 2 + 1;

            // Otherwise, add whichever child digest (if any) is not loaded
            if !self.page_states.contains_key(&lhs_idx) {
                image.set_digest(lhs_idx, *self.image.get_digest(lhs_idx)?);
            }
            if !self.page_states.contains_key(&rhs_idx) {
                image.set_digest(rhs_idx, *self.image.get_digest(rhs_idx)?);
            }
        }

        let post_state = self.image.image_id();

        Ok((pre_state, image, post_state))
    }

    fn load_page(&mut self, page_idx: u32) -> Result<()> {
        tracing::trace!("load_page: {page_idx:#08x}");
        let page = self.image.get_page(page_idx)?;
        self.page_table[page_idx as usize] = self.page_cache.len() as u32;
        self.page_cache.push(page);
        self.cycles += PAGE_CYCLES;
        self.trace_page_in(PAGE_CYCLES);
        self.fixup_costs(node_idx(page_idx), PageState::Loaded);
        Ok(())
    }

    #[inline]
    fn fixup_costs(&mut self, mut node_idx: u32, goal: PageState) {
        tracing::trace!("fixup: {node_idx:#010x}: {goal:?}");
        // Optimize the loop by handling the common case efficiently
        while node_idx != 0 {
            // Direct access with fallback for better performance
            let state = match self.page_states.get(&node_idx) {
                Some(&s) => s,
                None => PageState::Unloaded,
            };
            
            if goal > state {
                if node_idx < MEMORY_PAGES as u32 {
                    // Track cycles differently based on state transitions
                    if state == PageState::Unloaded {
                        self.cycles += NODE_CYCLES;
                        self.trace_page_in(NODE_CYCLES);
                    }
                    if goal == PageState::Dirty {
                        self.cycles += NODE_CYCLES;
                        self.trace_page_out(NODE_CYCLES);
                    }
                }
                self.page_states.insert(node_idx, goal);
            }
            // Bitshift is slightly faster than division
            node_idx >>= 1;
        }
    }

    pub(crate) fn trace_events(&self) -> &[PageTraceEvent] {
        &self.trace_events
    }

    pub(crate) fn clear_trace_events(&mut self) {
        self.trace_events.clear();
    }

    #[inline]
    fn trace_page_in(&mut self, cycles: u32) {
        if self.tracing_enabled {
            self.trace_events.push(PageTraceEvent::PageIn { cycles });
        }
    }

    #[inline]
    fn trace_page_out(&mut self, cycles: u32) {
        if self.tracing_enabled {
            self.trace_events.push(PageTraceEvent::PageOut { cycles });
        }
    }
}

#[inline(always)]
pub(crate) fn page_idx(node_idx: u32) -> u32 {
    node_idx - MEMORY_PAGES as u32
}
