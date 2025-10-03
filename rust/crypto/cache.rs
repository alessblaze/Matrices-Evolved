/*

Matrices-Evolved - High Performance ops offloaded to Rust and C++
Copyright (c) 2025 Albert Blasczykowski (Aless Microsystems)

This program is licensed under the Aless Microsystems Source-Available License (Non-Commercial, No Military) v1.0 Available in the Root
Directory of the project as LICENSE in Text Format.
You may use, copy, modify, and distribute this program for Non-Commercial purposes only, subject to the terms of that license.
Use by or for military, intelligence, or defense entities or purposes is strictly prohibited.

If you distribute this program in object form or make it available to others over a network, you must provide the complete
corresponding source code for the provided functionality under this same license.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the License for details.

You should have received a copy of the License along with this program; if not, see the LICENSE file included with this source.

*/

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyString, PyInt, PyBool, PyAny};
use parking_lot::RwLock;
use tokio::sync::RwLock as AsyncRwLock;
use hashbrown::HashMap;
use std::sync::{Arc, LazyLock};
use num_bigint::BigInt;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use ahash::AHasher;

// Global timer for relative time tracking
static CACHE_START_TIME: LazyLock<u64> = LazyLock::new(|| {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
});

// Get relative time in milliseconds since cache system started
fn get_relative_time_ms() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    now.saturating_sub(*CACHE_START_TIME)
}

// Convert Python clock time (seconds) to our relative time (milliseconds)
fn python_time_to_relative_ms(python_seconds: f64) -> u64 {
    (python_seconds * 1000.0) as u64
}

// Cached debug flag for optimal performance
static DEBUG_ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("SYNAPSE_RUST_CACHE_DEBUG").is_ok());

// Cache configuration constants
const DEFAULT_ENTRY_SIZE: usize = 1;
const PYTHON_OBJECT_OVERHEAD: usize = 64;
const CALLBACK_OBJECT_OVERHEAD: usize = 64;
const ASYNC_RETRY_ATTEMPTS: usize = 5;
const INITIAL_RETRY_DELAY_MS: u64 = 1;

// Optimized debug macro that checks cached flag first
macro_rules! rust_debug_fast {
    ($($arg:tt)*) => {
        if *DEBUG_ENABLED {
            println!("[RUST] {}", format!($($arg)*));
        }
    };
}

type FastHashMap<K, V> = HashMap<K, V, ahash::RandomState>;

#[derive(Debug, Clone)]
pub enum CacheKey {
    String(String),
    IntSmall(i64),
    IntBig(BigInt),
    None,
    Tuple(Box<[CacheKey]>),
    Hashed { type_name: String, py_hash: u64, obj_id: u64 },
}

// Custom Eq to unify small/large ints
impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        use CacheKey::*;
        match (self, other) {
            (IntSmall(a), IntSmall(b)) => a == b,
            (IntBig(a), IntBig(b)) => a == b,
            (IntSmall(a), IntBig(b)) | (IntBig(b), IntSmall(a)) => BigInt::from(*a) == *b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other) && match (self, other) {
                (String(a), String(b)) => a == b,
                (Tuple(a), Tuple(b)) => a == b,
                (None, None) => true,
                (Hashed { type_name: ta, py_hash: ha, obj_id: ia }, Hashed { type_name: tb, py_hash: hb, obj_id: ib }) => ta == tb && ha == hb && ia == ib,
                _ => false,
            }
        }
    }
}
impl Eq for CacheKey {}

// Custom Hash to make IntSmall/IntBig produce identical digests for equal values
impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use CacheKey::*;
        match self {
            IntSmall(v) => {
                state.write_u8(1);
                BigInt::from(*v).hash(state);
            }
            IntBig(bi) => {
                state.write_u8(1);
                bi.hash(state);
            }
            String(s) => { state.write_u8(2); s.hash(state); }
            Tuple(t) => { state.write_u8(3); t.hash(state); }
            None => { state.write_u8(4); }
            Hashed { type_name, py_hash, obj_id } => { 
                state.write_u8(5); 
                type_name.hash(state); 
                py_hash.hash(state); 
                obj_id.hash(state); 
            }
        }
    }
}

impl CacheKey {
    fn memory_size(&self) -> usize {
        use CacheKey::*;
        match self {
            String(s) => std::mem::size_of::<std::string::String>() + s.len(),
            IntSmall(_) => std::mem::size_of::<i64>(),
            IntBig(bi) => std::mem::size_of::<BigInt>() + ((bi.bits() + 7) / 8) as usize,
            None => std::mem::size_of::<()>(),
            Tuple(parts) => {
                std::mem::size_of::<Box<[CacheKey]>>() + 
                parts.iter().map(|k| k.memory_size()).sum::<usize>()
            },
            Hashed { type_name, .. } => {
                std::mem::size_of::<std::string::String>() + type_name.len() + 2 * std::mem::size_of::<u64>()
            },
        }
    }
    
    /// Converts a Python object to a CacheKey for efficient hashing and comparison.
    /// Supports None, tuples, booleans, integers, strings, and arbitrary hashable objects.
    /// Time complexity: O(1) for primitives, O(k) for tuples where k is tuple length.
    fn from_bound(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            return Ok(CacheKey::None);
        }
        if obj.is_instance_of::<PyTuple>() {
            let tuple = obj.downcast::<PyTuple>()?;
            return Ok(CacheKey::Tuple(Self::extract_tuple_parts(tuple)?));
        }
        if obj.is_instance_of::<PyBool>() {
            // Normalize bools into the integer path
            let b: bool = obj.extract()?;
            return Ok(CacheKey::IntSmall(if b { 1 } else { 0 }));
        }
        if obj.is_instance_of::<PyInt>() {
            // Try small fast path
            if let Ok(v) = obj.extract::<i64>() {
                return Ok(CacheKey::IntSmall(v));
            }
            // Lossless big-int via decimal string -> BigInt
            let s = obj.str()?.to_string();
            let big = s.parse::<BigInt>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>("Failed to parse Python int")
            })?;
            return Ok(CacheKey::IntBig(big));
        }
        if obj.is_instance_of::<PyString>() {
            return Ok(CacheKey::String(obj.extract()?));
        }

        // Fallback: arbitrary Python object
        let type_name = obj.get_type().name()?.to_string();
        let py_hash = obj.hash().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Unhashable key of type {}", type_name)
            )
        })? as u64;
        let obj_id = obj.as_ptr() as usize as u64;

        Ok(CacheKey::Hashed { type_name, py_hash, obj_id })
    }
    
    /// Recursively extracts tuple elements into CacheKey boxed slice.
    /// Time complexity: O(k) where k is the total number of nested elements.
    fn extract_tuple_parts(tuple: &Bound<'_, PyTuple>) -> PyResult<Box<[CacheKey]>> {
        let mut parts = Vec::with_capacity(tuple.len());
        for i in 0..tuple.len() {
            let item = tuple.get_item(i)?;
            parts.push(Self::from_bound(&item)?);
        }
        Ok(parts.into_boxed_slice())
    }
    

}

#[derive(Debug)]
struct LruNode {
    prev: Option<Arc<CacheKey>>,
    next: Option<Arc<CacheKey>>,
}

#[derive(Debug)]
pub struct CacheEntry {
    value: Py<PyAny>,
    callbacks: Option<Vec<Py<PyAny>>>,  // None = no callbacks (memory optimization)
    original_key: Py<PyAny>,
    prefix_arcs: Option<Vec<Arc<CacheKey>>>,
    node: Option<Py<RustCacheNode>>,  // Auto-created node for global eviction
    node_id: Option<u64>,  // Global eviction node ID
}

/// Rust-based cache node for global eviction tracking
#[pyclass]
pub struct RustCacheNode {
    cache: std::sync::Weak<RwLock<UnifiedCache>>,
    key: Arc<CacheKey>,
    node_id: u64,
    callbacks: Option<Vec<Py<PyAny>>>,
    // Time-based eviction fields
    last_access_time: AtomicU64,
    creation_time: u64,
    access_count: AtomicU64,
}

static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

impl CacheEntry {
    fn new(value: Py<PyAny>, original_key: Py<PyAny>) -> Self {
        Self {
            value,
            callbacks: None,
            original_key,
            prefix_arcs: None,
            node: None,
            node_id: None,
        }
    }
    

    
    /// Estimate memory usage of this cache entry
    fn memory_size(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        
        // Py<PyAny> is just a pointer, but estimate Python object overhead
        size += PYTHON_OBJECT_OVERHEAD; // value object estimate
        size += PYTHON_OBJECT_OVERHEAD; // original_key object estimate
        
        // Callbacks vector
        if let Some(ref callbacks) = self.callbacks {
            size += callbacks.capacity() * std::mem::size_of::<Py<PyAny>>();
            size += callbacks.len() * CALLBACK_OBJECT_OVERHEAD; // callback object estimates
        }
        
        // Prefix arcs vector
        if let Some(ref prefix_arcs) = self.prefix_arcs {
            size += prefix_arcs.capacity() * std::mem::size_of::<Arc<CacheKey>>();
            size += prefix_arcs.iter().map(|arc| arc.memory_size()).sum::<usize>();
        }
        
        size
    }
    
    /// Add callbacks with deduplication - deduplication done outside locks
    fn add_callbacks_deduped(&mut self, deduped_callbacks: Vec<Py<PyAny>>) {
        if deduped_callbacks.is_empty() {
            return;
        }
        
        match &mut self.callbacks {
            Some(existing) => existing.extend(deduped_callbacks),
            None => self.callbacks = Some(deduped_callbacks),
        }
    }
    
    /// Run and clear all callbacks
    fn run_and_clear_callbacks(&mut self, py: Python) {
        if let Some(callbacks) = self.callbacks.take() {
            for callback in callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }
        }
    }
    
    /// Get callbacks for execution without clearing
    fn get_callbacks(&self) -> Vec<Py<PyAny>> {
        match &self.callbacks {
            Some(cb) => Python::with_gil(|py| cb.iter().map(|c| c.clone_ref(py)).collect()),
            None => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CallbackPolicy {
    Replace,
    Append,
}

#[derive(Debug)]
struct CacheShard {
    data: FastHashMap<Arc<CacheKey>, CacheEntry>,
    lru_nodes: FastHashMap<Arc<CacheKey>, Box<LruNode>>,
    head: Option<Arc<CacheKey>>,
    tail: Option<Arc<CacheKey>>,
    current_size: usize,
}

#[derive(Debug)]
struct GlobalEvictionList {
    head: Option<u64>,
    tail: Option<u64>,
    nodes: FastHashMap<u64, GlobalEvictionNode>,
    access_counter: AtomicU64,
}

#[derive(Debug)]
struct GlobalEvictionNode {
    key: Arc<CacheKey>,
    shard_index: usize,
    prev: Option<u64>,
    next: Option<u64>,
    access_order: u64,
}

impl GlobalEvictionList {
    fn new() -> Self {
        Self {
            head: None,
            tail: None,
            nodes: FastHashMap::default(),
            access_counter: AtomicU64::new(0),
        }
    }
    
    fn move_to_front(&mut self, node_id: u64) {
        if self.head == Some(node_id) { 
            // Update access order even if already at head
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.access_order = self.access_counter.fetch_add(1, Ordering::Relaxed);
            }
            return; 
        }
        
        let (prev, next) = if let Some(node) = self.nodes.get(&node_id) {
            (node.prev, node.next)
        } else {
            return;
        };
        
        // Remove from current position
        if let Some(prev) = prev {
            if let Some(prev_node) = self.nodes.get_mut(&prev) {
                prev_node.next = next;
            }
        }
        if let Some(next) = next {
            if let Some(next_node) = self.nodes.get_mut(&next) {
                next_node.prev = prev;
            }
        } else {
            self.tail = prev;
        }
        
        // Move to front
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.prev = None;
            node.next = self.head;
            node.access_order = self.access_counter.fetch_add(1, Ordering::Relaxed);
        }
        
        if let Some(old_head) = self.head {
            if let Some(head_node) = self.nodes.get_mut(&old_head) {
                head_node.prev = Some(node_id);
            }
        }
        
        self.head = Some(node_id);
        if self.tail.is_none() {
            self.tail = Some(node_id);
        }
    }
    
    fn add_node(&mut self, node_id: u64, key: Arc<CacheKey>, shard_index: usize) {
        let access_order = self.access_counter.fetch_add(1, Ordering::Relaxed);
        let node = GlobalEvictionNode {
            key: key.clone(),
            shard_index,
            prev: None,
            next: self.head,
            access_order,
        };
        
        if let Some(old_head) = self.head {
            if let Some(head_node) = self.nodes.get_mut(&old_head) {
                head_node.prev = Some(node_id);
            }
        }
        
        self.nodes.insert(node_id, node);
        self.head = Some(node_id);
        if self.tail.is_none() {
            self.tail = Some(node_id);
        }
        

    }
    
    fn remove_lru(&mut self) -> Option<(Arc<CacheKey>, usize)> {
        loop {
            let tail_id = self.tail?;
            if let Some(node) = self.nodes.remove(&tail_id) {
                self.tail = node.prev;
                if let Some(new_tail) = node.prev {
                    if let Some(tail_node) = self.nodes.get_mut(&new_tail) {
                        tail_node.next = None;
                    }
                } else {
                    self.head = None;
                }
                return Some((node.key, node.shard_index));
            }
            // Node was stale, continue to next tail
        }
    }
    
    fn remove_node(&mut self, node_id: u64) {
        if let Some(node) = self.nodes.remove(&node_id) {
            if let Some(prev) = node.prev {
                if let Some(prev_node) = self.nodes.get_mut(&prev) {
                    prev_node.next = node.next;
                }
            } else {
                self.head = node.next;
            }
            
            if let Some(next) = node.next {
                if let Some(next_node) = self.nodes.get_mut(&next) {
                    next_node.prev = node.prev;
                }
            } else {
                self.tail = node.prev;
            }
        }
    }
}

/// Unified cache implementation with sharding and global eviction tracking.
/// 
/// ## Lock Ordering
/// To prevent deadlocks, locks must be acquired in this order:
/// 1. hierarchy locks (hierarchy → prefix_counts → prefix_cache)
/// 2. shard locks (per-shard data and LRU)
/// 3. global_eviction lock
/// 
/// ## Memory Accounting
/// - `memory_usage`: Global atomic counter tracking total memory
/// - `shard.current_size`: Per-shard size tracking
/// - Both are updated consistently across all modification paths
/// 
/// ## Size vs Count Semantics
/// - `len()`: Always returns entry count
/// - `size()`: Returns total size when size_callback is set, otherwise same as len()
/// - Python `__len__()`: Returns size() when size_callback is set (non-standard behavior)
#[derive(Debug)]
pub struct UnifiedCache {
    name: String,
    capacity: usize,
    shards: Vec<RwLock<CacheShard>>,
    shard_count: usize,
    prefix_cache: RwLock<FastHashMap<CacheKey, Arc<CacheKey>>>,
    prefix_counts: RwLock<FastHashMap<CacheKey, usize>>,
    hierarchy: RwLock<FastHashMap<Arc<CacheKey>, FastHashMap<Arc<CacheKey>, ()>>>,
    global_eviction: RwLock<GlobalEvictionList>,
    metrics: Option<Py<PyAny>>,
    callback_policy: CallbackPolicy,
    size_callback: Arc<RwLock<Option<Py<PyAny>>>>,
    // Node tracking for global eviction
    enable_nodes: bool,
    // Internal stats tracking
    hits: AtomicU64,
    misses: AtomicU64,
    evictions_size: AtomicU64,
    evictions_invalidation: AtomicU64,
    memory_usage: AtomicU64,
    sets: AtomicU64,
    setdefault_hits: AtomicU64,
    prefix_invalidations: AtomicU64,
}

impl CacheShard {
    fn new(capacity_per_shard: usize) -> Self {
        Self {
            data: FastHashMap::with_capacity_and_hasher(capacity_per_shard, ahash::RandomState::new()),
            lru_nodes: FastHashMap::default(),
            head: None,
            tail: None,
            current_size: 0,
        }
    }
}

/// Deduplicate callbacks outside of any locks to avoid Python operations under locks
fn deduplicate_callbacks(py: Python, existing: &[Py<PyAny>], new_callbacks: Vec<Py<PyAny>>) -> Vec<Py<PyAny>> {
    if new_callbacks.is_empty() {
        return Vec::new();
    }
    
    let mut seen_hashes = HashSet::new();
    
    // Hash existing callbacks
    for existing_cb in existing {
        if let Ok(hash) = existing_cb.bind(py).hash() {
            seen_hashes.insert(hash);
        }
    }
    
    let mut deduped = Vec::new();
    for new_cb in new_callbacks {
        if let Ok(hash) = new_cb.bind(py).hash() {
            if seen_hashes.insert(hash) {
                deduped.push(new_cb);
            }
        } else {
            // Fallback to linear search for unhashable callbacks
            let mut is_duplicate = false;
            for existing_cb in existing {
                match existing_cb.bind(py).eq(&new_cb.bind(py)) {
                    Ok(true) => {
                        is_duplicate = true;
                        break;
                    }
                    Ok(false) => continue,
                    Err(_) => continue,
                }
            }
            if !is_duplicate {
                deduped.push(new_cb);
            }
        }
    }
    deduped
}

impl UnifiedCache {
    /// Creates a new UnifiedCache with specified capacity and configuration.
    /// Initializes all internal data structures with optimal sizing.
    /// Time complexity: O(1).
    pub fn new(name: String, capacity: usize, metrics: Option<Py<PyAny>>, callback_policy: CallbackPolicy) -> Self {
        Self::new_with_shards(name, capacity, metrics, callback_policy, None)
    }
    
    pub fn new_with_shards(name: String, capacity: usize, metrics: Option<Py<PyAny>>, callback_policy: CallbackPolicy, shard_count: Option<usize>) -> Self {
        let shard_count = shard_count.unwrap_or_else(|| (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / 2).max(4));
        let capacity_per_shard = (capacity + shard_count - 1) / shard_count;
        let shards = (0..shard_count).map(|_| RwLock::new(CacheShard::new(capacity_per_shard))).collect();
        
        Self {
            name,
            capacity,
            shards,
            shard_count,
            prefix_cache: RwLock::new(FastHashMap::default()),
            prefix_counts: RwLock::new(FastHashMap::default()),
            hierarchy: RwLock::new(FastHashMap::default()),
            global_eviction: RwLock::new(GlobalEvictionList::new()),
            metrics,
            callback_policy,
            size_callback: Arc::new(RwLock::new(None)),
            enable_nodes: true,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions_size: AtomicU64::new(0),
            evictions_invalidation: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            sets: AtomicU64::new(0),
            setdefault_hits: AtomicU64::new(0),
            prefix_invalidations: AtomicU64::new(0),
        }
    }
    
    fn get_shard_index(&self, key: &Arc<CacheKey>) -> usize {
        let mut hasher = ahash::AHasher::default();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shard_count
    }
    
    fn move_to_front_shard(&self, shard: &mut CacheShard, key: &Arc<CacheKey>) -> bool {
        if !shard.data.contains_key(key) || !shard.lru_nodes.contains_key(key) {
            return false;
        }
        
        if shard.head.as_ref() == Some(key) {
            return true;
        }
        
        let Some(node) = shard.lru_nodes.get(key) else { return false; };
        let prev = node.prev.clone();
        let next = node.next.clone();
        
        if let Some(prev_key) = &prev {
            if let Some(prev_node) = shard.lru_nodes.get_mut(prev_key) {
                prev_node.next = next.clone();
            }
        }
        
        if let Some(next_key) = &next {
            if let Some(next_node) = shard.lru_nodes.get_mut(next_key) {
                next_node.prev = prev.clone();
            }
        }
        
        if shard.tail.as_ref() == Some(key) {
            shard.tail = prev;
        }
        
        let Some(node) = shard.lru_nodes.get_mut(key) else { return false; };
        node.prev = None;
        node.next = shard.head.clone();
        
        if let Some(old_head) = &shard.head {
            if let Some(head_node) = shard.lru_nodes.get_mut(old_head) {
                head_node.prev = Some(key.clone());
            }
        }
        
        shard.head = Some(key.clone());
        if shard.tail.is_none() {
            shard.tail = Some(key.clone());
        }
        
        true
    }
    
    fn add_to_front_shard(&self, shard: &mut CacheShard, key: Arc<CacheKey>) {
        let node = Box::new(LruNode {
            prev: None,
            next: shard.head.clone(),
        });
        
        if let Some(old_head) = &shard.head {
            // Update the old head's prev pointer to point to the new head
            if let Some(head_node) = shard.lru_nodes.get_mut(old_head) {
                head_node.prev = Some(key.clone());
            }
            // Tail remains unchanged when adding to front of existing list
        } else {
            // First entry - set as both head and tail
            shard.tail = Some(key.clone());
        }
        
        shard.lru_nodes.insert(key.clone(), node);
        shard.head = Some(key.clone());
        

    }
    

    
    fn remove_from_lru_shard(&self, shard: &mut CacheShard, key: &Arc<CacheKey>) {
        let Some(node) = shard.lru_nodes.get(key) else { return; };
        let prev = node.prev.clone();
        let next = node.next.clone();
        
        if let Some(prev_key) = &prev {
            if let Some(prev_node) = shard.lru_nodes.get_mut(prev_key) {
                prev_node.next = next.clone();
            }
        } else {
            shard.head = next.clone();
        }
        
        if let Some(next_key) = &next {
            if let Some(next_node) = shard.lru_nodes.get_mut(next_key) {
                next_node.prev = prev.clone();
            }
        } else {
            shard.tail = prev;
        }
        
        shard.lru_nodes.remove(key);
    }
    

    
    fn add_hierarchy_static(key: &Arc<CacheKey>, prefix_cache: &RwLock<FastHashMap<CacheKey, Arc<CacheKey>>>, prefix_counts: &RwLock<FastHashMap<CacheKey, usize>>, hierarchy: &RwLock<FastHashMap<Arc<CacheKey>, FastHashMap<Arc<CacheKey>, ()>>>) -> Vec<Arc<CacheKey>> {
        let mut prefix_arcs = Vec::new();
        if let CacheKey::Tuple(parts) = key.as_ref() {
            for i in 1..parts.len() {
                let prefix = CacheKey::Tuple(parts[0..i].to_vec().into_boxed_slice());
                
                // Consistent lock order: hierarchy -> counts -> cache
                let mut hier = hierarchy.write();
                let mut counts = prefix_counts.write();
                let mut cache = prefix_cache.write();
                
                *counts.entry(prefix.clone()).or_insert(0) += 1;
                
                let prefix_arc = if let Some(arc) = cache.get(&prefix) {
                    arc.clone()
                } else {
                    let arc = Arc::new(prefix.clone());
                    cache.insert(prefix, arc.clone());
                    arc
                };
                
                hier.entry(prefix_arc.clone())
                    .or_insert_with(FastHashMap::default)
                    .insert(key.clone(), ());
                    
                prefix_arcs.push(prefix_arc);
            }
        }
        prefix_arcs
    }
    

    
    fn remove_hierarchy_static(key: &Arc<CacheKey>, prefix_arcs: &[Arc<CacheKey>], prefix_cache: &RwLock<FastHashMap<CacheKey, Arc<CacheKey>>>, prefix_counts: &RwLock<FastHashMap<CacheKey, usize>>, hierarchy: &RwLock<FastHashMap<Arc<CacheKey>, FastHashMap<Arc<CacheKey>, ()>>>) {
        for prefix_arc in prefix_arcs {
            // Consistent lock order: hierarchy -> counts -> cache
            let mut hier = hierarchy.write();
            let mut counts = prefix_counts.write();
            let mut cache = prefix_cache.write();
            
            if let Some(children) = hier.get_mut(prefix_arc) {
                children.remove(key);
                if children.is_empty() {
                    hier.remove(prefix_arc);
                }
            }
            
            if let Some(count) = counts.get_mut(prefix_arc.as_ref()) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(prefix_arc.as_ref());
                    cache.remove(prefix_arc.as_ref());
                }
            }
        }
    }
    

    

    

    
    /// Enable node tracking for global eviction
    pub fn enable_node_tracking(&mut self) {
        self.enable_nodes = true;
        
        // Create nodes for existing entries
        Python::with_gil(|_py| {
            for (shard_idx, shard_lock) in self.shards.iter().enumerate() {
                let keys_to_update: Vec<_> = {
                    let shard = shard_lock.read();
                    shard.data.keys().cloned().collect()
                };
                
                for key in keys_to_update {
                    let mut shard = shard_lock.write();
                    if let Some(entry) = shard.data.get_mut(&key) {
                        if entry.node_id.is_none() {
                            let node_id = NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
                            entry.node_id = Some(node_id);
                            drop(shard);
                            let mut global_eviction = self.global_eviction.write();
                            global_eviction.add_node(node_id, key.clone(), shard_idx);
                        }
                    }
                }
            }
        });
    }
    
    /// Get all active cache nodes for global eviction (only existing node objects)
    pub fn get_all_nodes(&self) -> Vec<Py<RustCacheNode>> {
        Python::with_gil(|py| {
            let mut nodes = Vec::new();
            for shard_lock in &self.shards {
                let shard = shard_lock.read();
                for entry in shard.data.values() {
                    if let Some(node) = &entry.node {
                        nodes.push(node.clone_ref(py));
                    }
                }
            }
            nodes
        })
    }
    
    /// Get all cache entries as nodes, creating node objects on demand
    pub fn get_all_nodes_complete(&self, cache_ref: &Arc<RwLock<UnifiedCache>>) -> Vec<Py<RustCacheNode>> {
        Python::with_gil(|py| {
            let mut nodes = Vec::new();
            for shard_lock in &self.shards {
                let mut shard = shard_lock.write();
                for (key, entry) in shard.data.iter_mut() {
                    if let Some(node) = &entry.node {
                        nodes.push(node.clone_ref(py));
                    } else if let Some(node_id) = entry.node_id {
                        // Create node on demand
                        if let Ok(node) = self.create_eviction_node(cache_ref, py, key, &[], Some(node_id)) {
                            entry.node = Some(node.clone_ref(py));
                            nodes.push(node);
                        }
                    }
                }
            }
            nodes
        })
    }
    
    /// Create a node for global eviction system integration
    fn create_eviction_node(&self, cache_ref: &Arc<RwLock<UnifiedCache>>, py: Python, key: &Arc<CacheKey>, callbacks: &[Py<PyAny>], existing_node_id: Option<u64>) -> PyResult<Py<RustCacheNode>> {
        let node_id = existing_node_id.unwrap_or_else(|| NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed));
        let now = get_relative_time_ms();
        
        let node = RustCacheNode {
            cache: Arc::downgrade(cache_ref),
            key: key.clone(),
            node_id,
            callbacks: if callbacks.is_empty() { 
                None 
            } else { 
                Some(callbacks.iter().map(|cb| cb.clone_ref(py)).collect()) 
            },
            last_access_time: AtomicU64::new(now),
            creation_time: now,
            access_count: AtomicU64::new(0),
        };
        
        Ok(Py::new(py, node)?)
    }
    



    

    

    

    

    

    
    /// Invalidates all cache entries that have the specified prefix.
    pub fn invalidate_prefix(&self, py: Python, prefix_key: &Arc<CacheKey>) -> Vec<(Arc<CacheKey>, CacheEntry)> {
        let mut removed_entries = Vec::new();
        
        // Get children from global hierarchy
        let child_keys: Vec<_> = {
            let hierarchy = self.hierarchy.read();
            if let Some(children) = hierarchy.get(prefix_key) {
                children.keys().cloned().collect()
            } else {
                Vec::new()
            }
        };
        
        // Remove entries from appropriate shards
        for child_key in child_keys {
            let shard_index = self.get_shard_index(&child_key);
            
            // Copy value and remove entry under lock
            let (entry, value_copy) = {
                let mut shard = self.shards[shard_index].write();
                if let Some(entry) = shard.data.remove(&child_key) {
                    let value_copy = entry.value.clone_ref(py);
                    self.remove_from_lru_shard(&mut shard, &child_key);
                    (Some(entry), Some(value_copy))
                } else {
                    (None, None)
                }
            };
            
            if let (Some(entry), Some(value_copy)) = (entry, value_copy) {
                // Calculate size outside lock
                let size = if self.size_callback.read().is_some() {
                    self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
                } else {
                    DEFAULT_ENTRY_SIZE
                };
                
                // Update size tracking
                {
                    let mut shard = self.shards[shard_index].write();
                    shard.current_size = shard.current_size.saturating_sub(size);
                }
                self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
                
                // Remove from global eviction list
                if let Some(node_id) = entry.node_id {
                    let mut global_eviction = self.global_eviction.write();
                    global_eviction.remove_node(node_id);
                }
                
                let prefix_arcs = entry.prefix_arcs.clone();
                if let Some(arcs) = prefix_arcs.as_ref() {
                    Self::remove_hierarchy_static(&child_key, arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
                }
                removed_entries.push((child_key, entry));
            }
        }
        
        self.prefix_invalidations.fetch_add(removed_entries.len() as u64, Ordering::Relaxed);
        removed_entries
    }
    
    /// Removes and returns a cache entry, or returns default if not found.
    pub fn pop(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let shard_index = self.get_shard_index(&cache_key);
        
        // Copy value and remove entry under lock
        let (entry, value_copy) = {
            let mut shard = self.shards[shard_index].write();
            if let Some(entry) = shard.data.remove(&cache_key) {
                let value_copy = entry.value.clone_ref(py);
                self.remove_from_lru_shard(&mut shard, &cache_key);
                (Some(entry), Some(value_copy))
            } else {
                (None, None)
            }
        };
        
        if let (Some(entry), Some(value_copy)) = (entry, value_copy) {
            // Calculate size outside lock
            let size = if self.size_callback.read().is_some() {
                self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
            } else {
                DEFAULT_ENTRY_SIZE
            };
            
            // Update size tracking
            {
                let mut shard = self.shards[shard_index].write();
                shard.current_size = shard.current_size.saturating_sub(size);
            }
            self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
            
            let callbacks = entry.get_callbacks();
            let value = entry.value;
            let prefix_arcs = entry.prefix_arcs.clone();
            
            // Remove from global eviction list
            if let Some(node_id) = entry.node_id {
                let mut global_eviction = self.global_eviction.write();
                global_eviction.remove_node(node_id);
            }
            
            if let Some(arcs) = prefix_arcs.as_ref() {
                Self::remove_hierarchy_static(&cache_key, arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
            }
            Ok((value, callbacks))
        } else {
            Ok((default.unwrap_or_else(|| py.None()), Vec::new()))
        }
    }
    
    /// Removes and returns the least recently used cache entry.
    pub fn popitem(&self, py: Python) -> Option<(Py<PyAny>, Py<PyAny>, Vec<Py<PyAny>>)> {
        // Loop until we find a live entry or the list is empty
        loop {
            let evict_info = {
                let mut global_eviction = self.global_eviction.write();
                global_eviction.remove_lru()
            };
            
            let Some((lru_key, shard_idx)) = evict_info else {
                return None; // No more entries
            };
            
            // Copy value and remove entry under lock
            let (entry, value_copy) = {
                let mut shard = self.shards[shard_idx].write();
                if let Some(entry) = shard.data.remove(&lru_key) {
                    let value_copy = entry.value.clone_ref(py);
                    self.remove_from_lru_shard(&mut shard, &lru_key);
                    (Some(entry), Some(value_copy))
                } else {
                    (None, None) // Stale node, continue loop
                }
            };
            
            if let (Some(entry), Some(value_copy)) = (entry, value_copy) {
                // Calculate size outside lock
                let size = if self.size_callback.read().is_some() {
                    self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
                } else {
                    DEFAULT_ENTRY_SIZE
                };
                
                // Update size tracking
                {
                    let mut shard = self.shards[shard_idx].write();
                    shard.current_size = shard.current_size.saturating_sub(size);
                }
                self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
                self.evictions_size.fetch_add(1, Ordering::Relaxed);
                
                let callbacks = entry.get_callbacks();
                let original_key = entry.original_key;
                let value = entry.value;
                let prefix_arcs = entry.prefix_arcs.clone();
                
                // Remove hierarchy after getting entry data
                if let Some(arcs) = prefix_arcs.as_ref() {
                    Self::remove_hierarchy_static(&lru_key, arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
                }
                
                return Some((original_key, value, callbacks));
            }
            // Entry was stale, continue to next LRU entry
        }
    }
    
    /// Inserts or updates a cache entry with the specified key and value.
    /// Handles capacity management by evicting LRU entries when needed.
    /// Updates LRU position and hierarchy relationships for tuple keys.
    /// Time complexity: O(k) where k is tuple length for hierarchy operations.
    pub fn set(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let shard_index = self.get_shard_index(&cache_key);

        let original_key = key.clone().unbind();
        let mut evicted_callbacks = Vec::new();
        
        // Track set operations
        self.sets.fetch_add(1, Ordering::Relaxed);
        
        // Check if entry exists and collect data for size/equality calculations
        let (entry_exists, old_value, existing_callbacks) = {
            let shard = self.shards[shard_index].read();
            if let Some(entry) = shard.data.get(&cache_key) {
                (true, Some(entry.value.clone_ref(py)), entry.get_callbacks())
            } else {
                (false, None, Vec::new())
            }
        };
        
        if entry_exists {
            // Calculate size difference and value equality outside locks
            let size_diff = if self.size_callback.read().is_some() {
                if let Some(old_val) = &old_value {
                    let old_size = self.calculate_size_unlocked(py, old_val).unwrap_or(DEFAULT_ENTRY_SIZE);
                    let new_size = self.calculate_size_unlocked(py, &value).unwrap_or(DEFAULT_ENTRY_SIZE);
                    Some(new_size as i64 - old_size as i64)
                } else {
                    None
                }
            } else {
                None
            };
            
            let value_changed = if let Some(old_val) = &old_value {
                !old_val.bind(py).eq(&value.bind(py)).unwrap_or(false)
            } else {
                true
            };
            
            // Deduplicate callbacks outside locks
            let deduped_callbacks = match self.callback_policy {
                CallbackPolicy::Replace => callbacks,
                CallbackPolicy::Append => {
                    if callbacks.is_empty() {
                        Vec::new()
                    } else {
                        deduplicate_callbacks(py, &existing_callbacks, callbacks)
                    }
                }
            };
            
            // Update entry under lock
            let change_callbacks = {
                let mut shard = self.shards[shard_index].write();
                let mut change_cbs = Vec::new();
                
                if let Some(entry) = shard.data.get_mut(&cache_key) {
                    if value_changed {
                        change_cbs.extend(entry.get_callbacks());
                        entry.callbacks = None;
                    }
                    
                    entry.value = value.clone_ref(py);
                    
                    match self.callback_policy {
                        CallbackPolicy::Replace => {
                            if !deduped_callbacks.is_empty() {
                                entry.callbacks = Some(deduped_callbacks);
                            }
                        },
                        CallbackPolicy::Append => {
                            entry.add_callbacks_deduped(deduped_callbacks);
                        },
                    }
                    
                    // Update size tracking
                    if let Some(diff) = size_diff {
                        if diff > 0 {
                            shard.current_size += diff as usize;
                            self.memory_usage.fetch_add(diff as u64, Ordering::Relaxed);
                        } else if diff < 0 {
                            shard.current_size = shard.current_size.saturating_sub((-diff) as usize);
                            self.memory_usage.fetch_sub((-diff) as u64, Ordering::Relaxed);
                        }
                    }
                    
                    self.move_to_front_shard(&mut shard, &cache_key);
                }
                
                change_cbs
            };
            
            // Execute change callbacks outside the lock
            for callback in change_callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }
        } else {

            // Add hierarchy outside of shard lock to maintain consistent lock order
            let prefix_arcs = Self::add_hierarchy_static(&cache_key, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
            
            // Calculate size outside lock
            let entry_size = if self.size_callback.read().is_some() {
                self.calculate_size_unlocked(py, &value).unwrap_or(DEFAULT_ENTRY_SIZE)
            } else {
                DEFAULT_ENTRY_SIZE
            };
            
            // Insert entry under lock
            let node_id = {
                let mut shard = self.shards[shard_index].write();
                
                let mut entry = CacheEntry::new(value.clone_ref(py), original_key);
                entry.callbacks = if callbacks.is_empty() { None } else { Some(callbacks) };
                entry.prefix_arcs = if prefix_arcs.is_empty() { None } else { Some(prefix_arcs) };
                
                // Update size tracking
                shard.current_size += entry_size;
                self.memory_usage.fetch_add(entry_size as u64, Ordering::Relaxed);
                
                // Add to global eviction list
                let node_id = NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
                entry.node_id = Some(node_id);
                
                shard.data.insert(cache_key.clone(), entry);
                self.add_to_front_shard(&mut shard, cache_key.clone());
                
                node_id
            };
            
            // Add to global eviction outside shard lock
            {
                let mut global_eviction = self.global_eviction.write();
                global_eviction.add_node(node_id, cache_key.clone(), shard_index);
            };
            
            // Evict LRU entries after insertion
            let mut eviction_attempts = 0;
            let max_eviction_attempts = self.capacity + 10;
            
            let use_size_based = self.size_callback.read().is_some();
            while (if use_size_based { self.size() } else { self.len() }) > self.capacity && eviction_attempts < max_eviction_attempts {
                eviction_attempts += 1;
                
                // Get LRU entry to evict
                let evict_info = {
                    let mut global_eviction = self.global_eviction.write();
                    global_eviction.remove_lru()
                };
                
                if let Some((lru_key, shard_idx)) = evict_info {
                    // Copy value and remove entry under lock
                    let (entry_data, value_copy) = {
                        let mut shard = self.shards[shard_idx].write();
                        if let Some(entry) = shard.data.remove(&lru_key) {
                            let value_copy = if use_size_based {
                                Some(entry.value.clone_ref(py))
                            } else {
                                None
                            };
                            self.remove_from_lru_shard(&mut shard, &lru_key);
                            self.evictions_size.fetch_add(1, Ordering::Relaxed);
                            
                            (Some((entry.get_callbacks(), entry.prefix_arcs.clone())), value_copy)
                        } else {
                            (None, None)
                        }
                    };
                    
                    if let Some((callbacks, prefix_arcs)) = entry_data {
                        // Calculate size outside lock
                        let size = if let Some(value_copy) = value_copy {
                            self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
                        } else {
                            DEFAULT_ENTRY_SIZE
                        };
                        
                        // Update size tracking
                        {
                            let mut shard = self.shards[shard_idx].write();
                            shard.current_size = shard.current_size.saturating_sub(size);
                        }
                        self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
                        
                        evicted_callbacks.extend(callbacks);
                        if let Some(arcs) = prefix_arcs.as_ref() {
                            Self::remove_hierarchy_static(&lru_key, arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
                        }
                    }
                } else {
                    break; // No more entries to evict
                }
            }
        }
        
        rust_debug_fast!("UnifiedCache::set() completed");
        Ok(evicted_callbacks)
    }
    
    /// Retrieves a cache entry and optionally updates its LRU position and metrics.
    /// Records cache hit/miss metrics if metrics are configured and update_metrics is true.
    /// Time complexity: O(1) for HashMap lookup and LRU update.
    pub fn get(&self, py: Python, key: &Bound<'_, PyAny>, callbacks: Option<Vec<Py<PyAny>>>, update_metrics: bool, update_last_access: bool) -> PyResult<Option<Py<PyAny>>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let shard_index = self.get_shard_index(&cache_key);

        // First, check if entry exists and get data for callback deduplication
        let (value, existing_callbacks, node_id) = {
            let shard = self.shards[shard_index].read();
            if let Some(entry) = shard.data.get(&cache_key) {
                let value = entry.value.clone_ref(py);
                let existing_cbs = entry.get_callbacks();
                let node_id = if update_last_access { entry.node_id } else { None };
                (Some(value), existing_cbs, node_id)
            } else {
                (None, Vec::new(), None)
            }
        };
        
        if let Some(value) = value {
            // Deduplicate callbacks outside locks if needed
            let deduped_callbacks = if let Some(new_callbacks) = callbacks {
                if new_callbacks.is_empty() {
                    Vec::new()
                } else {
                    match self.callback_policy {
                        CallbackPolicy::Replace => new_callbacks,
                        CallbackPolicy::Append => deduplicate_callbacks(py, &existing_callbacks, new_callbacks),
                    }
                }
            } else {
                Vec::new()
            };
            
            // Update callbacks and LRU under lock
            if !deduped_callbacks.is_empty() || update_last_access {
                let mut shard = self.shards[shard_index].write();
                if let Some(entry) = shard.data.get_mut(&cache_key) {
                    if !deduped_callbacks.is_empty() {
                        match self.callback_policy {
                            CallbackPolicy::Replace => {
                                entry.callbacks = Some(deduped_callbacks);
                            },
                            CallbackPolicy::Append => {
                                entry.add_callbacks_deduped(deduped_callbacks);
                            },
                        }
                    }
                    
                    if update_last_access {
                        self.move_to_front_shard(&mut shard, &cache_key);
                    }
                }
            }
            
            // Update global eviction list outside shard lock
            if let Some(node_id) = node_id {
                let mut global_eviction = self.global_eviction.write();
                global_eviction.move_to_front(node_id);
            }
            
            // Record metrics
            if update_metrics {
                self.hits.fetch_add(1, Ordering::Relaxed);
                if let Some(ref metrics) = self.metrics {
                    let _ = metrics.bind(py).call_method0("inc_hits");
                }
            }
            Ok(Some(value))
        } else {
            if update_metrics {
                self.misses.fetch_add(1, Ordering::Relaxed);
                if let Some(ref metrics) = self.metrics {
                    let _ = metrics.bind(py).call_method0("inc_misses");
                }
            }
            Ok(None)
        }
    }
    
    /// Removes a cache entry and performs full cleanup.
    /// Returns (found, callbacks) tuple indicating success and any callbacks to execute.
    /// Time complexity: O(k) where k is tuple length for hierarchy cleanup.
    pub fn invalidate(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<Py<PyAny>>)> {
        rust_debug_fast!("UnifiedCache::invalidate() called");
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        self.invalidate_by_key(py, &cache_key)
    }
    
    /// Invalidates a cache entry by its CacheKey directly
    pub fn invalidate_by_key(&self, py: Python, key: &Arc<CacheKey>) -> PyResult<(bool, Vec<Py<PyAny>>)> {
        let shard_index = self.get_shard_index(key);
        
        // Copy value and remove entry under lock
        let (entry, value_copy) = {
            let mut shard = self.shards[shard_index].write();
            if let Some(entry) = shard.data.remove(key) {
                rust_debug_fast!("Entry found, removing from cache and LRU");
                let value_copy = entry.value.clone_ref(py);
                self.remove_from_lru_shard(&mut shard, key);
                (Some(entry), Some(value_copy))
            } else {
                rust_debug_fast!("Entry not found");
                (None, None)
            }
        };
        
        if let (Some(entry), Some(value_copy)) = (entry, value_copy) {
            // Calculate size outside lock
            let size = if self.size_callback.read().is_some() {
                self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
            } else {
                DEFAULT_ENTRY_SIZE
            };
            
            // Update size tracking
            {
                let mut shard = self.shards[shard_index].write();
                shard.current_size = shard.current_size.saturating_sub(size);
            }
            self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
            self.evictions_invalidation.fetch_add(1, Ordering::Relaxed);
            
            // Remove from global eviction list
            if let Some(node_id) = entry.node_id {
                let mut global_eviction = self.global_eviction.write();
                global_eviction.remove_node(node_id);
            }
            
            // Get prefix arcs and remove hierarchy
            let prefix_arcs = entry.prefix_arcs.clone();
            if let Some(arcs) = prefix_arcs.as_ref() {
                Self::remove_hierarchy_static(key, arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
            }
            
            Ok((true, entry.get_callbacks()))
        } else {
            Ok((false, Vec::new()))
        }
    }
    
    /// Gets a cache entry by its CacheKey directly
    pub fn get_by_key(&self, key: &Arc<CacheKey>) -> bool {
        let shard_index = self.get_shard_index(key);
        self.shards[shard_index].read().data.contains_key(key)
    }
    
    /// Updates LRU position for a key without returning the value
    pub fn touch_key(&self, key: &Arc<CacheKey>) {
        let shard_index = self.get_shard_index(key);
        let mut shard = self.shards[shard_index].write();
        if shard.data.contains_key(key) {
            self.move_to_front_shard(&mut shard, key);
        }
    }
    
    /// Removes all entries from the cache and resets all internal state.
    /// Collects all callbacks from removed entries for execution.
    /// Time complexity: O(n) where n is the number of cache entries.
    pub fn clear(&self, py: Python) -> (usize, Vec<Py<PyAny>>) {
        let count = self.len();
        rust_debug_fast!("UnifiedCache::clear() called for cache '{}' with {} entries", self.name, count);
        
        let mut all_callbacks = Vec::new();
        
        // Clear all shards
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            for entry in shard.data.values() {
                all_callbacks.extend(entry.get_callbacks().iter().map(|cb| cb.clone_ref(py)));
            }
            shard.data.clear();
            // Hierarchy is now global, cleared above
            shard.lru_nodes.clear();
            shard.head = None;
            shard.tail = None;
            shard.current_size = 0;
        }
        
        self.prefix_cache.write().clear();
        self.prefix_counts.write().clear();
        self.hierarchy.write().clear();
        self.memory_usage.store(0, Ordering::Relaxed);
        
        rust_debug_fast!("UnifiedCache::clear() completed for cache '{}', cleared {} entries, {} callbacks", self.name, count, all_callbacks.len());
        (count, all_callbacks)
    }
    
    /// Returns the current number of entries in the cache.
    /// Time complexity: O(shards).
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.read().data.len()).sum()
    }
    
    /// Returns the total size when size_callback is set, otherwise same as len().
    pub fn size(&self) -> usize {
        if self.size_callback.read().is_some() {
            self.shards.iter().map(|s| s.read().current_size).sum()
        } else {
            self.len()
        }
    }
    
    /// Checks if a key exists in the cache without updating LRU position.
    /// Pure read operation that doesn't modify cache state.
    /// Time complexity: O(1).
    pub(crate) fn contains(&self, key: &Arc<CacheKey>) -> bool {
        let shard_index = self.get_shard_index(key);
        self.shards[shard_index].read().data.contains_key(key)
    }
    
    /// Returns the maximum capacity of the cache.
    /// Time complexity: O(1).
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Sets the size callback for calculating entry sizes.
    pub fn set_size_callback(&self, _py: Python, callback: Option<Py<PyAny>>) {
        *self.size_callback.write() = callback;
    }
    
    /// Calculates the size of a value using the size callback - call outside locks
    fn calculate_size_unlocked(&self, py: Python, value: &Py<PyAny>) -> PyResult<usize> {
        if let Some(ref callback) = *self.size_callback.read() {
            let result = callback.bind(py).call1((value,))?;
            Ok(result.extract()?)
        } else {
            Ok(DEFAULT_ENTRY_SIZE)
        }
    }
    

    
    /// Resizes the cache to a new capacity, evicting entries if necessary.
    pub fn resize(&mut self, py: Python, new_capacity: usize) -> Vec<Py<PyAny>> {
        let old_capacity = self.capacity;
        self.capacity = new_capacity;
        let mut evicted_callbacks = Vec::new();
        
        // If shrinking, evict LRU entries using global eviction list
        if new_capacity < old_capacity {
            let use_size_based = self.size_callback.read().is_some();
            while (if use_size_based { self.size() } else { self.len() }) > new_capacity {
                let evict_info = {
                    let mut global_eviction = self.global_eviction.write();
                    global_eviction.remove_lru()
                };
                
                if let Some((lru_key, shard_idx)) = evict_info {
                    // Copy value and remove entry under lock
                    let (entry_data, value_copy) = {
                        let mut shard = self.shards[shard_idx].write();
                        if let Some(entry) = shard.data.remove(&lru_key) {
                            let value_copy = if use_size_based {
                                Some(entry.value.clone_ref(py))
                            } else {
                                None
                            };
                            self.remove_from_lru_shard(&mut shard, &lru_key);
                            (Some((entry.get_callbacks(), entry.prefix_arcs.clone())), value_copy)
                        } else {
                            (None, None)
                        }
                    };
                    
                    if let Some((callbacks, prefix_arcs)) = entry_data {
                        // Calculate size outside lock if needed
                        let size = if let Some(value_copy) = value_copy {
                            self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
                        } else {
                            DEFAULT_ENTRY_SIZE
                        };
                        
                        // Update size tracking
                        {
                            let mut shard = self.shards[shard_idx].write();
                            shard.current_size = shard.current_size.saturating_sub(size);
                        }
                        self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
                        
                        evicted_callbacks.extend(callbacks);
                        if let Some(arcs) = prefix_arcs.as_ref() {
                            Self::remove_hierarchy_static(&lru_key, arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
                        }
                    }
                } else {
                    break;
                }
            }
        }
        
        evicted_callbacks
    }
    
    /// Atomic get-or-set operation - returns existing value or sets and returns new value.
    pub fn setdefault(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let shard_index = self.get_shard_index(&cache_key);
        
        // First check if entry exists
        {
            let mut shard = self.shards[shard_index].write();
            if let Some(entry) = shard.data.get(&cache_key) {
                self.setdefault_hits.fetch_add(1, Ordering::Relaxed);
                let existing_value = entry.value.clone_ref(py);
                self.move_to_front_shard(&mut shard, &cache_key);
                return Ok(existing_value);
            }
        }
        
        // Entry doesn't exist, prepare for insertion
        let original_key = key.clone().unbind();
        let prefix_arcs = Self::add_hierarchy_static(&cache_key, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
        
        // Calculate size outside lock
        let entry_size = if self.size_callback.read().is_some() {
            self.calculate_size_unlocked(py, &value).unwrap_or(DEFAULT_ENTRY_SIZE)
        } else {
            DEFAULT_ENTRY_SIZE
        };
        
        // Insert entry
        let node_id = {
            let mut shard = self.shards[shard_index].write();
            
            // Check if entry was inserted concurrently
            if let Some(entry) = shard.data.get(&cache_key) {
                self.setdefault_hits.fetch_add(1, Ordering::Relaxed);
                let existing_value = entry.value.clone_ref(py);
                self.move_to_front_shard(&mut shard, &cache_key);
                return Ok(existing_value);
            }
            
            let mut entry = CacheEntry::new(value.clone_ref(py), original_key);
            entry.prefix_arcs = if prefix_arcs.is_empty() { None } else { Some(prefix_arcs) };
            
            // Update size tracking
            shard.current_size += entry_size;
            self.memory_usage.fetch_add(entry_size as u64, Ordering::Relaxed);
            
            // Add to global eviction list
            let node_id = NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
            entry.node_id = Some(node_id);
            
            shard.data.insert(cache_key.clone(), entry);
            self.add_to_front_shard(&mut shard, cache_key.clone());
            
            node_id
        };
        
        // Add to global eviction outside shard lock
        {
            let mut global_eviction = self.global_eviction.write();
            global_eviction.add_node(node_id, cache_key.clone(), shard_index);
        }
        
        // Capacity enforcement after insertion
        let use_size_based = self.size_callback.read().is_some();
        while (if use_size_based { self.size() } else { self.len() }) > self.capacity {
            let evict_info = {
                let mut global_eviction = self.global_eviction.write();
                global_eviction.remove_lru()
            };
            
            if let Some((lru_key, shard_idx)) = evict_info {
                // Copy value and remove entry under lock
                let (entry_data, value_copy) = {
                    let mut shard = self.shards[shard_idx].write();
                    if let Some(entry) = shard.data.remove(&lru_key) {
                        let value_copy = if use_size_based {
                            Some(entry.value.clone_ref(py))
                        } else {
                            None
                        };
                        self.remove_from_lru_shard(&mut shard, &lru_key);
                        self.evictions_size.fetch_add(1, Ordering::Relaxed);
                        
                        (Some(entry.prefix_arcs.clone()), value_copy)
                    } else {
                        (None, None)
                    }
                };
                
                if let Some(prefix_arcs) = entry_data {
                    // Calculate size outside lock
                    let size = if let Some(value_copy) = value_copy {
                        self.calculate_size_unlocked(py, &value_copy).unwrap_or(DEFAULT_ENTRY_SIZE)
                    } else {
                        DEFAULT_ENTRY_SIZE
                    };
                    
                    // Update size tracking
                    {
                        let mut shard = self.shards[shard_idx].write();
                        shard.current_size = shard.current_size.saturating_sub(size);
                    }
                    self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
                    
                    if let Some(arcs) = prefix_arcs {
                        Self::remove_hierarchy_static(&lru_key, &arcs, &self.prefix_cache, &self.prefix_counts, &self.hierarchy);
                    }
                }
            } else {
                break;
            }
        }
        
        self.sets.fetch_add(1, Ordering::Relaxed);
        Ok(value)
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions_size.load(Ordering::Relaxed),
            self.evictions_invalidation.load(Ordering::Relaxed),
            self.memory_usage.load(Ordering::Relaxed),
            self.sets.load(Ordering::Relaxed),
            self.setdefault_hits.load(Ordering::Relaxed),
            self.prefix_invalidations.load(Ordering::Relaxed),
        )
    }
    
    /// Reset all statistics
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions_size.store(0, Ordering::Relaxed);
        self.evictions_invalidation.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
        self.sets.store(0, Ordering::Relaxed);
        self.setdefault_hits.store(0, Ordering::Relaxed);
        self.prefix_invalidations.store(0, Ordering::Relaxed);
    }
    
    /// Gets all entries that have the given tuple prefix.
    pub fn get_prefix_children(&self, py: Python, prefix_key: &Bound<'_, PyAny>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let cache_key = Arc::new(CacheKey::from_bound(prefix_key)?);
        let mut results = Vec::new();
        
        // Get children from global hierarchy
        let child_keys: Vec<_> = {
            let hierarchy = self.hierarchy.read();
            if let Some(children) = hierarchy.get(&cache_key) {
                children.keys().cloned().collect()
            } else {
                Vec::new()
            }
        };
        
        // Get entries from appropriate shards
        for child_key in child_keys {
            let shard_index = self.get_shard_index(&child_key);
            let shard = self.shards[shard_index].read();
            if let Some(entry) = shard.data.get(&child_key) {
                results.push((entry.original_key.clone_ref(py), entry.value.clone_ref(py)));
            }
        }
        
        Ok(results)
    }
}

#[pyclass]
pub struct RustLruCache {
    cache: Arc<RwLock<UnifiedCache>>,
    name: String,
}

#[pyclass]
pub struct AsyncRustLruCache {
    cache: Arc<AsyncRwLock<UnifiedCache>>,
    name: String,
}

#[pymethods]
impl RustCacheNode {
    #[getter]
    fn key(&self, py: Python) -> PyResult<Py<PyAny>> {
        fn reconstruct_key(key: &CacheKey, py: Python) -> PyResult<Py<PyAny>> {
            match key {
                CacheKey::String(s) => Ok(PyString::new(py, s).into()),
                CacheKey::IntSmall(i) => Ok(PyInt::new(py, *i).into()),
                CacheKey::IntBig(bi) => {
                    let s = bi.to_string();
                    if let Ok(i) = s.parse::<i64>() {
                        Ok(PyInt::new(py, i).into())
                    } else {
                        // Create PyLong directly from decimal string
                        let py_long = py.import("builtins")?.getattr("int")?.call1((&s,))?;
                        Ok(py_long.unbind())
                    }
                },
                CacheKey::None => Ok(py.None()),
                CacheKey::Tuple(parts) => {
                    let py_parts: PyResult<Vec<_>> = parts.iter()
                        .map(|part| reconstruct_key(part, py))
                        .collect();
                    let tuple = PyTuple::new(py, py_parts?)?;
                    Ok(tuple.into())
                },
                CacheKey::Hashed { type_name, py_hash, obj_id } => {
                    Ok(PyString::new(py, &format!("{}#{}@{}", type_name, py_hash, obj_id)).into())
                }
            }
        }
        reconstruct_key(&self.key, py)
    }
    
    #[getter]
    fn value(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            let shard_index = cache.get_shard_index(&self.key);
            let shard = cache.shards[shard_index].read();
            match shard.data.get(&self.key) {
                Some(entry) => Ok(Some(entry.value.clone_ref(py))),
                None => Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    fn get_cache_entry(&self) -> PyResult<()> {
        Ok(())
    }
    
    fn add_callbacks(&mut self, py: Python, callbacks: Vec<Py<PyAny>>) -> PyResult<()> {
        if callbacks.is_empty() {
            return Ok(());
        }
        
        match &mut self.callbacks {
            Some(existing) => {
                // Use HashSet for O(n) deduplication instead of O(n²)
                let mut seen_hashes = HashSet::new();
                
                // Hash existing callbacks
                for existing_cb in existing.iter() {
                    if let Ok(hash) = existing_cb.bind(py).hash() {
                        seen_hashes.insert(hash);
                    }
                }
                
                // Add new callbacks if not already seen
                for new_cb in callbacks {
                    if let Ok(hash) = new_cb.bind(py).hash() {
                        if seen_hashes.insert(hash) {
                            existing.push(new_cb);
                        }
                    } else {
                        // Fallback to linear search for unhashable callbacks
                        let mut is_duplicate = false;
                        for existing_cb in existing.iter() {
                            match existing_cb.bind(py).eq(&new_cb.bind(py)) {
                                Ok(true) => {
                                    is_duplicate = true;
                                    break;
                                }
                                Ok(false) => continue,
                                Err(_) => {
                                    // If comparison fails, assume not duplicate to avoid data loss
                                    continue;
                                }
                            }
                        }
                        if !is_duplicate {
                            existing.push(new_cb);
                        }
                    }
                }
            }
            None => {
                self.callbacks = Some(callbacks);
            }
        }
        Ok(())
    }
    
    fn run_and_clear_callbacks(&mut self, py: Python) {
        if let Some(callbacks) = self.callbacks.take() {
            for callback in callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }
        }
    }
    
    fn drop_from_cache(&mut self, py: Python) -> PyResult<bool> {
        if let Some(cache_arc) = self.cache.upgrade() {
            let (found, callbacks) = {
                let cache = cache_arc.read();
                cache.invalidate_by_key(py, &self.key)?
            };
            
            for callback in callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }
            
            self.run_and_clear_callbacks(py);
            Ok(found)
        } else {
            Ok(false) // Cache has been dropped
        }
    }
    
    fn update_last_access(&self, py: Python, clock: Option<Py<PyAny>>) {
        // Use Python clock time if provided, otherwise use relative time
        let now = if let Some(clock_obj) = clock {
            if let Ok(time_method) = clock_obj.bind(py).getattr("time") {
                if let Ok(time_result) = time_method.call0() {
                    if let Ok(time_seconds) = time_result.extract::<f64>() {
                        python_time_to_relative_ms(time_seconds)
                    } else {
                        get_relative_time_ms()
                    }
                } else {
                    get_relative_time_ms()
                }
            } else {
                get_relative_time_ms()
            }
        } else {
            get_relative_time_ms()
        };
        
        self.last_access_time.store(now, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
        
        // Update LRU position in cache and global eviction list
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            let shard_index = cache.get_shard_index(&self.key);
            let mut shard = cache.shards[shard_index].write();
            cache.move_to_front_shard(&mut shard, &self.key);
            drop(shard);
            
            // Also move in global eviction list
            let mut global_eviction = cache.global_eviction.write();
            global_eviction.move_to_front(self.node_id);
        }
    }
    
    /// Get last access time in relative milliseconds
    fn get_last_access_time(&self) -> u64 {
        self.last_access_time.load(Ordering::Relaxed)
    }
    
    /// Get creation time in relative milliseconds
    fn get_creation_time(&self) -> u64 {
        self.creation_time
    }
    
    /// Get last access time in Python clock seconds (for compatibility)
    fn get_last_access_time_seconds(&self) -> f64 {
        self.last_access_time.load(Ordering::Relaxed) as f64 / 1000.0
    }
    
    /// Get creation time in Python clock seconds (for compatibility)
    fn get_creation_time_seconds(&self) -> f64 {
        self.creation_time as f64 / 1000.0
    }
    
    /// Get access count
    fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
    
    /// Check if node is older than given time (for time-based eviction)
    fn is_older_than(&self, time_ms: u64) -> bool {
        self.last_access_time.load(Ordering::Relaxed) < time_ms
    }
    
    /// Check if node is older than given Python clock time in seconds
    fn is_older_than_seconds(&self, time_seconds: f64) -> bool {
        let time_ms = python_time_to_relative_ms(time_seconds);
        self.last_access_time.load(Ordering::Relaxed) < time_ms
    }
    
    /// Get memory usage of this node
    fn get_memory_usage(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        size += self.key.memory_size();
        if let Some(ref callbacks) = self.callbacks {
            size += callbacks.len() * std::mem::size_of::<Py<PyAny>>();
        }
        size
    }
    
    /// Calculate value size using cache's size callback
    fn calculate_value_size(&self, py: Python) -> PyResult<usize> {
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            let shard_index = cache.get_shard_index(&self.key);
            let shard = cache.shards[shard_index].read();
            if let Some(entry) = shard.data.get(&self.key) {
                cache.calculate_size_unlocked(py, &entry.value)
            } else {
                Ok(DEFAULT_ENTRY_SIZE)
            }
        } else {
            Ok(DEFAULT_ENTRY_SIZE)
        }
    }
    
    /// Get total memory footprint (node + value)
    fn get_total_memory_usage(&self, py: Python) -> PyResult<usize> {
        let node_size = self.get_memory_usage();
        let value_size = self.calculate_value_size(py)?;
        Ok(node_size + value_size)
    }
    
    /// Check if this node's key still exists in cache
    fn is_valid(&self) -> bool {
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            cache.contains(&self.key)
        } else {
            false
        }
    }
    
    /// Get node ID for tracking
    fn get_node_id(&self) -> u64 {
        self.node_id
    }
    
    /// Get callbacks without clearing (for inspection)
    fn get_callbacks(&self) -> Vec<Py<PyAny>> {
        match &self.callbacks {
            Some(cb) => Python::with_gil(|py| cb.iter().map(|c| c.clone_ref(py)).collect()),
            None => Vec::new()
        }
    }
    

    
    /// Compare current value with new value (for change detection)
    fn value_changed(&self, py: Python, new_value: Py<PyAny>) -> PyResult<bool> {
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            let shard_index = cache.get_shard_index(&self.key);
            let shard = cache.shards[shard_index].read();
            if let Some(entry) = shard.data.get(&self.key) {
                match entry.value.bind(py).eq(&new_value.bind(py)) {
                    Ok(is_equal) => Ok(!is_equal),
                    Err(e) => {
                        eprintln!("Warning: Failed to compare values in cache: {}", e);
                        Ok(true)
                    }
                }
            } else {
                Ok(true)
            }
        } else {
            Ok(true)
        }
    }
    
    /// Set value and run callbacks if changed
    fn set_value_if_changed(&mut self, py: Python, new_value: Py<PyAny>) -> PyResult<bool> {
        let changed = self.value_changed(py, new_value.clone_ref(py))?;
        if changed {
            if let Some(cache_arc) = self.cache.upgrade() {
                let cache = cache_arc.read();
                let shard_index = cache.get_shard_index(&self.key);
                
                // Get old value and callbacks outside lock
                let (old_value, entry_callbacks) = {
                    let shard = cache.shards[shard_index].read();
                    if let Some(entry) = shard.data.get(&self.key) {
                        (Some(entry.value.clone_ref(py)), entry.get_callbacks())
                    } else {
                        (None, Vec::new())
                    }
                };
                
                // Calculate size difference outside lock
                let size_diff = if let (Some(old_val), Some(ref size_callback)) = (&old_value, cache.size_callback.read().as_ref()) {
                    let old_size = size_callback.bind(py).call1((old_val.bind(py),))
                        .and_then(|r| r.extract::<usize>())
                        .unwrap_or(DEFAULT_ENTRY_SIZE);
                    let new_size = size_callback.bind(py).call1((new_value.bind(py),))
                        .and_then(|r| r.extract::<usize>())
                        .unwrap_or(DEFAULT_ENTRY_SIZE);
                    Some(new_size as i64 - old_size as i64)
                } else {
                    None
                };
                
                // Update entry under lock
                {
                    let mut shard = cache.shards[shard_index].write();
                    if let Some(entry) = shard.data.get_mut(&self.key) {
                        entry.value = new_value;
                        entry.callbacks = None; // Clear callbacks
                        
                        // Update size tracking
                        if let Some(diff) = size_diff {
                            if diff > 0 {
                                shard.current_size += diff as usize;
                                cache.memory_usage.fetch_add(diff as u64, Ordering::Relaxed);
                            } else if diff < 0 {
                                shard.current_size = shard.current_size.saturating_sub((-diff) as usize);
                                cache.memory_usage.fetch_sub((-diff) as u64, Ordering::Relaxed);
                            }
                        }
                    }
                }
                
                // Execute callbacks outside lock
                for callback in entry_callbacks {
                    if let Err(e) = callback.bind(py).call0() {
                        eprintln!("Cache callback error: {}", e);
                    }
                }
            }
            self.run_and_clear_callbacks(py);
        }
        Ok(changed)
    }
    
    /// Update access time for global eviction
    fn update_access_time(&self) {
        let now = get_relative_time_ms();
        self.last_access_time.store(now, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    

    

}

#[pymethods]
impl RustLruCache {
    /// Creates a new thread-safe Rust LRU cache with the specified configuration.
    /// Wraps UnifiedCache in Arc<Mutex<>> for safe concurrent access from Python.
    #[new]
    #[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None, size_callback=None, shard_count=None))]
    fn new(
        max_size: usize, 
        cache_name: Option<String>, 
        metrics: Option<Py<PyAny>>,
        callback_policy: Option<String>,
        size_callback: Option<Py<PyAny>>,
        shard_count: Option<usize>
    ) -> Self {
        let name = cache_name.unwrap_or_else(|| "rust_cache".to_string());
        let policy = match callback_policy.as_deref() {
            Some("append") => CallbackPolicy::Append,
            _ => CallbackPolicy::Replace,
        };
        let cache = Arc::new(RwLock::new(UnifiedCache::new_with_shards(name.clone(), max_size, metrics, policy, shard_count)));
        
        // Set size callback if provided
        if let Some(callback) = size_callback {
            Python::with_gil(|py| {
                cache.write().set_size_callback(py, Some(callback));
            });
        }
        
        Self { cache, name }
    }
    
    /// Returns the cache name for identification and debugging.
    #[getter]
    fn get_name(&self) -> &str {
        &self.name
    }
    
    /// Retrieves a value from the cache, returning default if not found.
    /// Thread-safe wrapper around UnifiedCache::get with mutex locking.
    #[pyo3(signature = (key, default=None, callbacks=None))]
    fn get(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<Py<PyAny>> {
        let cache = self.cache.read();
        match cache.get(py, key, callbacks, true, true)? {
            Some(value) => Ok(value),
            None => Ok(default.unwrap_or_else(|| py.None()))
        }
    }
    
    /// Inserts or updates a cache entry with automatic callback execution.
    /// Executes evicted callbacks outside the mutex lock to prevent deadlocks.
    fn set(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<()> {
        let callbacks = callbacks.unwrap_or_default();
        
        let (evicted_callbacks, needs_node) = {
            let cache = self.cache.read();
            let result = cache.set(py, key, value, callbacks)?;
            
            // Check if we need to create a node
            let needs_node = if cache.enable_nodes {
                let cache_key = Arc::new(CacheKey::from_bound(key)?);
                let shard_index = cache.get_shard_index(&cache_key);
                let shard = cache.shards[shard_index].read();
                shard.data.get(&cache_key)
                    .map(|entry| entry.node.is_none())
                    .unwrap_or(false)
            } else {
                false
            };
            
            (result, needs_node)
        };
        
        // Create node outside the cache lock to avoid borrow conflicts
        if needs_node {
            let cache_key = Arc::new(CacheKey::from_bound(key)?);
            
            // Create the node first
            let node = {
                let cache = self.cache.read();
                let shard_index = cache.get_shard_index(&cache_key);
                let shard = cache.shards[shard_index].read();
                let existing_node_id = shard.data.get(&cache_key).and_then(|e| e.node_id);
                cache.create_eviction_node(&self.cache, py, &cache_key, &[], existing_node_id)
            }?;
            
            // Then update the entry - re-check existence after concurrent removal
            let cache = self.cache.write();
            let shard_index = cache.get_shard_index(&cache_key);
            let mut shard = cache.shards[shard_index].write();
            if let Some(entry) = shard.data.get_mut(&cache_key) {
                if entry.node.is_none() {
                    entry.node = Some(node);
                }
                // If entry.node is already Some, the node was created concurrently - discard our node
            }
            // If entry doesn't exist, it was removed concurrently - discard our node
        }
        
        for callback in evicted_callbacks {
            if let Err(e) = callback.bind(py).call0() {
                eprintln!("Cache callback error: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Removes a cache entry and executes associated callbacks.
    /// Thread-safe with callback execution outside the mutex lock.
    fn invalidate(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let (_, callbacks) = {
            let cache = self.cache.read();
            cache.invalidate(py, key)?
        };
        
        for callback in callbacks {
            if let Err(e) = callback.bind(py).call0() {
                eprintln!("Cache callback error: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Invalidates all entries with the specified prefix (TreeCache functionality).
    /// Much faster than scanning all keys for prefix matches.
    fn invalidate_prefix(&self, py: Python, prefix_key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let cache_key = Arc::new(CacheKey::from_bound(prefix_key)?);
        let mut all_callbacks = Vec::new();
        let count = {
            let cache = self.cache.read();
            let removed_entries = cache.invalidate_prefix(py, &cache_key);
            let entry_count = removed_entries.len();
            
            for (_, entry) in removed_entries {
                all_callbacks.extend(entry.get_callbacks());
            }
            
            entry_count
        };
        
        for callback in all_callbacks {
            if let Err(e) = callback.bind(py).call0() {
                eprintln!("Cache callback error: {}", e);
            }
        }
        
        Ok(count)
    }
    
    /// Removes and returns a cache entry, or returns default if not found.
    /// Executes callbacks associated with the removed entry.
    fn pop(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let (value, callbacks) = {
            let cache = self.cache.read();
            cache.pop(py, key, default)?
        };
        
        for callback in callbacks {
            if let Err(e) = callback.bind(py).call0() {
                eprintln!("Cache callback error: {}", e);
            }
        }
        
        Ok(value)
    }
    
    /// Removes all cache entries and executes all associated callbacks.
    /// Returns the number of entries that were removed.
    fn clear(&self, py: Python) -> PyResult<usize> {
        let (len, callbacks) = {
            let cache = self.cache.read();
            cache.clear(py)
        };
        
        for callback in callbacks {
            if let Err(e) = callback.bind(py).call0() {
                eprintln!("Cache callback error: {}", e);
            }
        }
        
        Ok(len)
    }
    
    /// TreeCache compatibility method - delegates to standard get().
    /// Provided for Python wrapper compatibility with tuple keys.
    fn get_with_tuple(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        self.get(py, key, default, None)
    }
    
    /// TreeCache compatibility method - delegates to standard set().
    /// Provided for Python wrapper compatibility with tuple keys.
    fn set_with_tuple(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<()> {
        self.set(py, key, value, callbacks)
    }
    
    /// Removes and returns the least recently used entry as a (key, value) tuple.
    /// Executes callbacks and raises KeyError if cache is empty.
    fn popitem(&self, py: Python) -> PyResult<Py<PyAny>> {
        let result = {
            let cache = self.cache.read();
            cache.popitem(py)
        };

        if let Some((original_key, value, callbacks)) = result {
            for callback in callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }

            // Safely bind original_key - handle potential GIL issues
            match original_key.bind(py).downcast::<PyAny>() {
                Ok(key_bound) => {
                    let tuple_ref = PyTuple::new(py, [key_bound, value.bind(py)])?;
                    Ok(tuple_ref.unbind().into())
                }
                Err(_) => {
                    // Fallback if binding fails - return just the value
                    Ok(value)
                }
            }
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err("Cache is empty"))
        }
    }
    
    /// Python len() support - returns total size when size_callback is set, otherwise entry count.
    /// NOTE: This diverges from standard dict semantics where len() always returns count.
    /// When size_callback is set:
    ///   - len(cache) returns total size (sum of all entry sizes)
    ///   - Use cache.len() method to get actual entry count
    /// When no size_callback:
    ///   - len(cache) returns entry count (standard dict behavior)
    fn __len__(&self) -> PyResult<usize> {
        let cache = self.cache.read();
        Ok(if cache.size_callback.read().is_some() { cache.size() } else { cache.len() })
    }
    
    /// Python 'in' operator support - checks if key exists in cache.
    fn __contains__(&self, _py: Python, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        Ok(self.cache.read().contains(&cache_key))
    }
    
    /// Python dict[key] support - returns value or raises KeyError.
    fn __getitem__(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let cache = self.cache.read();
        match cache.get(py, key, None, true, true)? {
            Some(value) => Ok(value),
            None => Err(pyo3::exceptions::PyKeyError::new_err("Key not found"))
        }
    }
    
    /// Python dict[key] = value support - inserts entry without callbacks.
    fn __setitem__(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<()> {
        self.set(py, key, value, None)
    }
    
    /// Python del dict[key] support - removes entry or raises KeyError.
    fn __delitem__(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let (found, callbacks) = {
            let cache = self.cache.read();
            cache.invalidate(py, key)?
        };
        
        if found {
            for callback in callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err("Key not found"))
        }
    }
    
    /// Resizes the cache to a new capacity.
    fn resize(&self, py: Python, new_capacity: usize) -> PyResult<()> {
        let evicted_callbacks = {
            let mut cache = self.cache.write();
            cache.resize(py, new_capacity)
        };
        
        for callback in evicted_callbacks {
            if let Err(e) = callback.bind(py).call0() {
                eprintln!("Cache callback error: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Gets all entries that have the given tuple prefix.
    fn get_prefix_children(&self, py: Python, prefix_key: &Bound<'_, PyAny>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let cache = self.cache.read();
        cache.get_prefix_children(py, prefix_key)
    }
    
    /// Checks if a key exists in the cache without updating LRU position.
    fn contains(&self, _py: Python, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        Ok(self.cache.read().contains(&cache_key))
    }
    
    /// Sets the size callback for calculating entry sizes.
    fn set_size_callback(&self, py: Python, callback: Option<Py<PyAny>>) -> PyResult<()> {
        let cache = self.cache.read();
        cache.set_size_callback(py, callback);
        Ok(())
    }
    
    /// Atomic get-or-set operation
    fn setdefault(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let cache = self.cache.read();
        cache.setdefault(py, key, value)
    }
    
    /// TreeCache multi-get - returns iterator over prefix matches
    fn get_multi(&self, py: Python, prefix_key: &Bound<'_, PyAny>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        self.get_prefix_children(py, prefix_key)
    }
    
    /// TreeCache multi-delete - removes all entries with prefix
    fn del_multi(&self, py: Python, prefix_key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.invalidate_prefix(py, prefix_key)
    }
    
    /// Get cache statistics for debugging
    fn get_stats(&self) -> PyResult<(usize, usize)> {
        let cache = self.cache.read();
        let total_entries: usize = cache.shards.iter().map(|s| s.read().data.len()).sum();
        Ok((total_entries, cache.capacity))
    }
    
    /// Get actual entry count (always returns count, never size)
    fn len(&self) -> PyResult<usize> {
        let cache = self.cache.read();
        Ok(cache.len())
    }
    
    /// Get total size (returns size when size_callback is set, otherwise same as len)
    fn size(&self) -> PyResult<usize> {
        let cache = self.cache.read();
        Ok(cache.size())
    }
    
    /// Check if cache is empty
    fn is_empty(&self) -> PyResult<bool> {
        let cache = self.cache.read();
        Ok(cache.shards.iter().all(|s| s.read().data.is_empty()))
    }
    
    /// Get estimated memory usage in bytes (upper-bound approximation).
    /// Includes container capacity overhead and may overcount actual residency.
    /// Use as a rough estimate, not exact memory footprint.
    fn get_memory_usage(&self) -> PyResult<usize> {
        let cache = self.cache.read();
        let mut total = 0;
        
        for shard_lock in &cache.shards {
            let shard = shard_lock.read();
            total += shard.data.capacity() * std::mem::size_of::<(Arc<CacheKey>, CacheEntry)>();
            total += shard.lru_nodes.capacity() * std::mem::size_of::<(Arc<CacheKey>, Box<LruNode>)>();
            
            for (key, entry) in &shard.data {
                total += key.memory_size();
                total += entry.memory_size();
            }
        }
        
        Ok(total)
    }
    
    /// Enhanced get with update_metrics and update_last_access parameters
    fn get_advanced(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>, callbacks: Option<Vec<Py<PyAny>>>, _update_metrics: Option<bool>, _update_last_access: Option<bool>) -> PyResult<Py<PyAny>> {
        let cache = self.cache.read();
        let cb_list = callbacks.unwrap_or_default();
        
        let update_metrics = _update_metrics.unwrap_or(true);
        let update_last_access = _update_last_access.unwrap_or(true);
        match cache.get(py, key, Some(cb_list), update_metrics, update_last_access)? {
            Some(value) => Ok(value),
            None => Ok(default.unwrap_or_else(|| py.None()))
        }
    }
    
    /// Enable automatic node creation for global eviction tracking
    fn enable_node_tracking(&self) -> PyResult<()> {
        let mut cache = self.cache.write();
        cache.enable_node_tracking();
        Ok(())
    }
    
    /// Get all active cache nodes for global eviction (existing objects only)
    fn get_all_nodes(&self) -> PyResult<Vec<Py<RustCacheNode>>> {
        let cache = self.cache.read();
        Ok(cache.get_all_nodes())
    }
    
    /// Get all cache entries as nodes, creating node objects on demand
    fn get_all_nodes_complete(&self) -> PyResult<Vec<Py<RustCacheNode>>> {
        let cache = self.cache.read();
        Ok(cache.get_all_nodes_complete(&self.cache))
    }
    
    /// Get node for key from cache (for global eviction integration)
    fn get_node_for_key(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<Option<Py<RustCacheNode>>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let cache = self.cache.read();
        let shard_index = cache.get_shard_index(&cache_key);
        let shard = cache.shards[shard_index].read();
        
        Ok(shard.data.get(&cache_key)
            .and_then(|entry| entry.node.as_ref())
            .map(|node| node.clone_ref(py)))
    }
    
    /// Get cache statistics (hits, misses, evictions_size, evictions_invalidation, memory_usage, sets, setdefault_hits, prefix_invalidations)
    fn get_cache_stats(&self) -> PyResult<(u64, u64, u64, u64, u64, u64, u64, u64)> {
        let cache = self.cache.read();
        Ok(cache.get_stats())
    }
    
    /// Reset all cache statistics
    fn reset_cache_stats(&self) -> PyResult<()> {
        let cache = self.cache.read();
        cache.reset_stats();
        Ok(())
    }
    
    /// Get hit count
    fn get_hits(&self) -> PyResult<u64> {
        let cache = self.cache.read();
        Ok(cache.hits.load(std::sync::atomic::Ordering::Relaxed))
    }
    
    /// Get miss count
    fn get_misses(&self) -> PyResult<u64> {
        let cache = self.cache.read();
        Ok(cache.misses.load(std::sync::atomic::Ordering::Relaxed))
    }
    
    /// Get eviction count by size
    fn get_evictions_size(&self) -> PyResult<u64> {
        let cache = self.cache.read();
        Ok(cache.evictions_size.load(std::sync::atomic::Ordering::Relaxed))
    }
    
    /// Get eviction count by invalidation
    fn get_evictions_invalidation(&self) -> PyResult<u64> {
        let cache = self.cache.read();
        Ok(cache.evictions_invalidation.load(std::sync::atomic::Ordering::Relaxed))
    }
}

#[pymethods]
impl AsyncRustLruCache {
    #[new]
    #[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None))]
    fn new(
        max_size: usize, 
        cache_name: Option<String>, 
        metrics: Option<Py<PyAny>>,
        callback_policy: Option<String>
    ) -> Self {
        rust_debug_fast!("AsyncRustLruCache::new() called with max_size={}", max_size);
        let name = cache_name.unwrap_or_else(|| "async_rust_cache".to_string());
        let policy = match callback_policy.as_deref() {
            Some("append") => CallbackPolicy::Append,
            _ => CallbackPolicy::Replace,
        };
        rust_debug_fast!("Creating AsyncMutex<UnifiedCache>");
        let cache = Arc::new(AsyncRwLock::new(UnifiedCache::new_with_shards(name.clone(), max_size, metrics, policy, None)));
        rust_debug_fast!("AsyncRustLruCache created successfully");
        
        Self { cache, name }
    }
    
    #[getter]
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get<'p>(&self, py: Python<'p>, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Bound<'p, PyAny>> {
        rust_debug_fast!("🔍 AsyncRustLruCache::get() called for cache '{}'", self.name);
        let cache = self.cache.clone();
        let key = key.clone().unbind();
        let result = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let cache = cache.read().await;
            Python::with_gil(|py| {
                let key_bound = key.bind(py);
                match cache.get(py, &key_bound, None, true, true)? {
                    Some(value) => Ok(value),
                    None => Ok(default.unwrap_or_else(|| py.None()))
                }
            })
        });
        result
    }
    
    fn set<'p>(&self, py: Python<'p>, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<Bound<'p, PyAny>> {
        rust_debug_fast!("📝 AsyncRustLruCache::set() called for cache '{}'", self.name);
        let cache = self.cache.clone();
        let key = key.clone().unbind();
        let result = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let callbacks = callbacks.unwrap_or_default();
            let evicted_callbacks = {
                let cache = cache.read().await;
                Python::with_gil(|py| {
                    let key_bound = key.bind(py);
                    cache.set(py, &key_bound, value, callbacks)
                })?
            };
            
            Python::with_gil(|py| {
                for callback in evicted_callbacks {
                    if let Err(e) = callback.bind(py).call0() {
                        eprintln!("Cache callback error: {}", e);
                    }
                }
                Ok(py.None())
            })
        });
        result
    }
    
    fn invalidate<'p>(&self, py: Python<'p>, key: &Bound<'_, PyAny>) -> PyResult<Bound<'p, PyAny>> {
        rust_debug_fast!("❌ AsyncRustLruCache::invalidate() called for cache '{}'", self.name);
        let cache = self.cache.clone();
        let key = key.clone().unbind();
        let result = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (_, callbacks) = {
                let cache = cache.read().await;
                Python::with_gil(|py| {
                    let key_bound = key.bind(py);
                    cache.invalidate(py, &key_bound)
                })?
            };
            
            Python::with_gil(|py| {
                for callback in callbacks {
                    if let Err(e) = callback.bind(py).call0() {
                        eprintln!("Cache callback error: {}", e);
                    }
                }
                Ok(py.None())
            })
        });
        result
    }
    
    fn clear<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        rust_debug_fast!("🧹 AsyncRustLruCache::clear() called for cache '{}'", self.name);
        let cache = self.cache.clone();
        let result = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (len, callbacks) = {
                let cache = cache.read().await;
                Python::with_gil(|py| cache.clear(py))
            };
            
            Python::with_gil(|py| {
                for callback in callbacks {
                    if let Err(e) = callback.bind(py).call0() {
                        eprintln!("Cache callback error: {}", e);
                    }
                }
                Ok(len)
            })
        });
        result
    }
    
    fn clear_sync(&self, py: Python) -> PyResult<usize> {
        // Retry with exponential backoff for cleanup scenarios
        let mut delay_ms = INITIAL_RETRY_DELAY_MS;
        for attempt in 0..ASYNC_RETRY_ATTEMPTS {
            match self.cache.try_write() {
                Ok(cache) => {
                    rust_debug_fast!("AsyncRustLruCache::clear_sync() - acquired lock on attempt {}", attempt + 1);
                    let (len, callbacks) = cache.clear(py);
                    for callback in callbacks {
                        if let Err(e) = callback.bind(py).call0() {
                            eprintln!("Cache callback error: {}", e);
                        }
                    }
                    return Ok(len);
                }
                Err(_) => {
                    if attempt < ASYNC_RETRY_ATTEMPTS - 1 {
                        rust_debug_fast!("AsyncRustLruCache::clear_sync() - attempt {} failed, retrying in {}ms", attempt + 1, delay_ms);
                        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                        delay_ms *= 2; // Exponential backoff: 1ms, 2ms, 4ms, 8ms
                    }
                }
            }
        }
        rust_debug_fast!("AsyncRustLruCache::clear_sync() - all attempts failed, cache busy");
        Ok(0)
    }
}

/// Factory function to create a new RustLruCache instance from Python.
/// Provides a convenient interface for Python code to instantiate the cache.
#[pyfunction]
#[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None, size_callback=None))]
pub fn create_rust_lru_cache(
    max_size: usize, 
    cache_name: Option<String>, 
    metrics: Option<Py<PyAny>>,
    callback_policy: Option<String>,
    size_callback: Option<Py<PyAny>>
) -> PyResult<RustLruCache> {
    Ok(RustLruCache::new(max_size, cache_name, metrics, callback_policy, size_callback, None))
}

#[pyfunction]
#[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None))]
pub fn create_async_rust_lru_cache(
    max_size: usize, 
    cache_name: Option<String>, 
    metrics: Option<Py<PyAny>>,
    callback_policy: Option<String>
) -> PyResult<AsyncRustLruCache> {
    Ok(AsyncRustLruCache::new(max_size, cache_name, metrics, callback_policy))
}

// Note: We don't need a custom tokio runtime initialization.
// pyo3_async_runtimes::tokio::future_into_py handles runtime management automatically.