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
use std::collections::{HashSet, hash_map::DefaultHasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
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
            IntBig(bi) => std::mem::size_of::<BigInt>() + (bi.bits() / 8) as usize,
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
    // Global eviction list management
    prev_node: Option<u64>,
    next_node: Option<u64>,
    in_global_list: bool,
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
        }
    }
    
    fn new_with_node(value: Py<PyAny>, original_key: Py<PyAny>, node: Py<RustCacheNode>) -> Self {
        Self {
            value,
            callbacks: None,
            original_key,
            prefix_arcs: None,
            node: Some(node),
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
    
    /// Add callbacks with deduplication like original LruCache
    fn add_callbacks(&mut self, py: Python, new_callbacks: Vec<Py<PyAny>>) -> PyResult<()> {
        if new_callbacks.is_empty() {
            return Ok(());
        }
        
        if self.callbacks.is_none() {
            self.callbacks = Some(Vec::new());
        }
        
        let callbacks = self.callbacks.as_mut().unwrap();
        
        // Use HashSet for O(n) deduplication instead of O(n²)
        let mut seen_hashes = HashSet::new();
        
        // Hash existing callbacks
        for existing_cb in callbacks.iter() {
            if let Ok(hash) = existing_cb.bind(py).hash() {
                seen_hashes.insert(hash);
            }
        }
        
        // Add new callbacks if not already seen
        for new_cb in new_callbacks {
            if let Ok(hash) = new_cb.bind(py).hash() {
                if seen_hashes.insert(hash) {
                    callbacks.push(new_cb);
                }
            } else {
                // Fallback to linear search for unhashable callbacks
                let mut is_duplicate = false;
                for existing_cb in callbacks.iter() {
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
                    callbacks.push(new_cb);
                }
            }
        }
        Ok(())
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
pub struct UnifiedCache {
    name: String,
    capacity: usize,
    data: FastHashMap<Arc<CacheKey>, CacheEntry>,
    hierarchy: FastHashMap<Arc<CacheKey>, FastHashMap<Arc<CacheKey>, ()>>,
    prefix_cache: FastHashMap<CacheKey, Arc<CacheKey>>,
    prefix_counts: FastHashMap<CacheKey, usize>,
    metrics: Option<Py<PyAny>>,
    callback_policy: CallbackPolicy,
    lru_nodes: FastHashMap<Arc<CacheKey>, Box<LruNode>>,
    head: Option<Arc<CacheKey>>,
    tail: Option<Arc<CacheKey>>,
    size_callback: Option<Py<PyAny>>,
    current_size: usize,
    // Node tracking for global eviction
    enable_nodes: bool,
}

impl UnifiedCache {
    /// Creates a new UnifiedCache with specified capacity and configuration.
    /// Initializes all internal data structures with optimal sizing.
    /// Time complexity: O(1).
    pub fn new(name: String, capacity: usize, metrics: Option<Py<PyAny>>, callback_policy: CallbackPolicy) -> Self {
        Self {
            name,
            capacity,
            data: FastHashMap::with_capacity_and_hasher(capacity, ahash::RandomState::new()),
            hierarchy: FastHashMap::default(),
            prefix_cache: FastHashMap::default(),
            prefix_counts: FastHashMap::default(),
            metrics,
            callback_policy,
            lru_nodes: FastHashMap::default(),
            head: None,
            tail: None,
            size_callback: None,
            current_size: 0,
            enable_nodes: false,
        }
    }
    
    /// Enable node tracking for global eviction
    pub fn enable_node_tracking(&mut self) {
        self.enable_nodes = true;
    }
    
    /// Get all active cache nodes for global eviction
    pub fn get_all_nodes(&self) -> Vec<Py<RustCacheNode>> {
        Python::with_gil(|py| {
            self.data.values()
                .filter_map(|entry| entry.node.as_ref())
                .map(|node| node.clone_ref(py))
                .collect()
        })
    }
    
    /// Create a node for global eviction system integration
    fn create_eviction_node(&self, cache_ref: &Arc<RwLock<UnifiedCache>>, py: Python, key: &Arc<CacheKey>, callbacks: &[Py<PyAny>]) -> PyResult<Py<RustCacheNode>> {
        let node_id = NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let now = get_relative_time_ms();
        
        let node = RustCacheNode {
            cache: Arc::downgrade(cache_ref), // Proper weak reference to cache
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
            prev_node: None,
            next_node: None,
            in_global_list: false,
        };
        
        Ok(Py::new(py, node)?)
    }
    


    /// Moves an existing cache entry to the front of the LRU list (most recently used).
    /// Updates doubly-linked list pointers in O(1) time using HashMap lookups.
    /// Returns true if the key was found and moved, false otherwise.
    /// Time complexity: O(1).
    fn move_to_front(&mut self, key: &Arc<CacheKey>) -> bool {
        rust_debug_fast!("move_to_front() called for key: {:?}", key);
        // Consistency check: key must exist in both data and lru_nodes
        if !self.data.contains_key(key) || !self.lru_nodes.contains_key(key) {
            rust_debug_fast!("move_to_front() failed: key not found in data or lru_nodes");
            return false;
        }
        
        if self.head.as_ref() == Some(key) {
            rust_debug_fast!("move_to_front() key already at head");
            return true;
        }
        
        let Some(node) = self.lru_nodes.get(key) else {
            return false;
        };
        
        let prev = node.prev.clone();
        let next = node.next.clone();
        
        // Update previous node's next pointer - safely handle missing nodes
        if let Some(prev_key) = &prev {
            if let Some(prev_node) = self.lru_nodes.get_mut(prev_key) {
                prev_node.next = next.clone();
            }
        }
        
        // Update next node's prev pointer - safely handle missing nodes
        if let Some(next_key) = &next {
            if let Some(next_node) = self.lru_nodes.get_mut(next_key) {
                next_node.prev = prev.clone();
            }
        }
        
        // Update tail if this was the tail
        if self.tail.as_ref() == Some(key) {
            self.tail = prev;
        }
        
        // Move to front - double-check node still exists
        let Some(node) = self.lru_nodes.get_mut(key) else {
            return false;
        };
        
        node.prev = None;
        node.next = self.head.clone();
        
        // Update old head's prev pointer - safely handle missing nodes
        if let Some(old_head) = &self.head {
            if let Some(head_node) = self.lru_nodes.get_mut(old_head) {
                head_node.prev = Some(key.clone());
            }
        }
        
        self.head = Some(key.clone());
        
        if self.tail.is_none() {
            self.tail = Some(key.clone());
        }
        
        true
    }
    
    /// Removes and returns the least recently used (tail) entry from the LRU list.
    /// Updates tail pointer and removes the node from the HashMap.
    /// Used during cache eviction when capacity is exceeded.
    /// Time complexity: O(1).
    fn remove_lru(&mut self) -> Option<Arc<CacheKey>> {
        let tail_key = self.tail.clone()?;
        
        // Consistency check: tail must exist in both data and lru_nodes
        if !self.data.contains_key(&tail_key) || !self.lru_nodes.contains_key(&tail_key) {
            // Inconsistent state - reset tail and try to recover
            self.tail = None;
            self.head = None;
            return None;
        }
        
        let tail_node = self.lru_nodes.get(&tail_key)?;
        let prev = tail_node.prev.clone();
        
        self.tail = prev.clone();
        
        if let Some(new_tail) = &prev {
            if let Some(new_tail_node) = self.lru_nodes.get_mut(new_tail) {
                new_tail_node.next = None;
            }
        } else {
            self.head = None;
        }
        
        self.lru_nodes.remove(&tail_key);
        Some(tail_key)
    }
    
    /// Adds a new entry to the front of the LRU list (most recently used position).
    /// Creates new LruNode and updates head/tail pointers as needed.
    /// Time complexity: O(1).
    fn add_to_front(&mut self, key: Arc<CacheKey>) {
        let node = Box::new(LruNode {
            prev: None,
            next: self.head.clone(),
        });
        
        if let Some(old_head) = &self.head {
            if let Some(head_node) = self.lru_nodes.get_mut(old_head) {
                head_node.prev = Some(key.clone());
            }
        }
        
        self.lru_nodes.insert(key.clone(), node);
        self.head = Some(key.clone());
        
        if self.tail.is_none() {
            self.tail = Some(key);
        }
    }
    
    /// Removes a specific entry from the LRU doubly-linked list.
    /// Updates prev/next pointers of adjacent nodes and head/tail as needed.
    /// Time complexity: O(1).
    fn remove_from_lru(&mut self, key: &Arc<CacheKey>) {
        let Some(node) = self.lru_nodes.get(key) else {
            return;
        };
        
        let prev = node.prev.clone();
        let next = node.next.clone();
        
        if let Some(prev_key) = &prev {
            if let Some(prev_node) = self.lru_nodes.get_mut(prev_key) {
                prev_node.next = next.clone();
            }
        } else {
            self.head = next.clone();
        }
        
        if let Some(next_key) = &next {
            if let Some(next_node) = self.lru_nodes.get_mut(next_key) {
                next_node.prev = prev.clone();
            }
        } else {
            self.tail = prev;
        }
        
        self.lru_nodes.remove(key);
    }
    
    /// Gets or creates an Arc<CacheKey> for a prefix with reference counting.
    /// Implements prefix interning to avoid duplicate Arc allocations.
    /// Increments reference count for existing prefixes.
    /// Time complexity: O(1).
    fn get_or_create_prefix(&mut self, prefix: CacheKey) -> Arc<CacheKey> {
        *self.prefix_counts.entry(prefix.clone()).or_insert(0) += 1;
        
        if let Some(arc) = self.prefix_cache.get(&prefix) {
            arc.clone()
        } else {
            let arc = Arc::new(prefix.clone());
            self.prefix_cache.insert(prefix, arc.clone());
            arc
        }
    }
    
    /// Decrements reference count for a prefix and cleans up if count reaches zero.
    /// Removes prefix from both prefix_cache and prefix_counts when no longer used.
    /// Time complexity: O(1).
    fn release_prefix(&mut self, prefix: &CacheKey) {
        if let Some(count) = self.prefix_counts.get_mut(prefix) {
            *count -= 1;
            if *count == 0 {
                self.prefix_counts.remove(prefix);
                self.prefix_cache.remove(prefix);
            }
        }
    }
    
    /// Adds tuple key to hierarchy by creating all prefix relationships.
    /// Returns Vec of prefix Arcs for storage in CacheEntry.
    /// Time complexity: O(k) where k is tuple length.
    fn add_hierarchy(&mut self, key: &Arc<CacheKey>) -> Vec<Arc<CacheKey>> {
        let mut prefix_arcs = Vec::new();
        if let CacheKey::Tuple(parts) = key.as_ref() {
            for i in 1..parts.len() {
                let prefix = CacheKey::Tuple(parts[0..i].to_vec().into_boxed_slice());
                let prefix_arc = self.get_or_create_prefix(prefix);
                self.hierarchy
                    .entry(prefix_arc.clone())
                    .or_insert_with(FastHashMap::default)
                    .insert(key.clone(), ());
                prefix_arcs.push(prefix_arc);
            }
        }
        prefix_arcs
    }
    
    /// Removes tuple key from hierarchy using stored prefix references.
    /// No allocations or HashMap lookups needed.
    /// Time complexity: O(k) where k is tuple length.
    fn remove_hierarchy_fast(&mut self, key: &Arc<CacheKey>, prefix_arcs: Option<&Vec<Arc<CacheKey>>>) {
        let Some(arcs) = prefix_arcs else {
            return;
        };
        
        for prefix_arc in arcs {
            if let Some(children) = self.hierarchy.get_mut(prefix_arc) {
                children.remove(key);
                if children.is_empty() {
                    self.hierarchy.remove(prefix_arc);
                }
            }
            self.release_prefix(prefix_arc);
        }
    }
    
    /// Invalidates all cache entries that have the specified prefix.
    /// Removes all children from the hierarchy and returns the removed entries.
    /// Time complexity: O(children) where children is the number of matching entries.
    fn invalidate_prefix(&mut self, py: Python, prefix_key: &Arc<CacheKey>) -> Vec<(Arc<CacheKey>, CacheEntry)> {
        let mut removed_entries = Vec::new();
        
        if let Some(children) = self.hierarchy.get(prefix_key) {
            let child_keys: Vec<_> = children.keys().cloned().collect();
            for child_key in child_keys {
                if let Some(entry) = self.data.remove(&child_key) {
                    // Update size tracking for removed entry
                    if self.size_callback.is_some() {
                        let entry_size = self.calculate_size(py, &entry.value).unwrap_or(DEFAULT_ENTRY_SIZE);
                        self.current_size = self.current_size.saturating_sub(entry_size);
                    }
                    
                    self.remove_from_lru(&child_key);
                    // Clean up hierarchy state for each removed child
                    self.remove_hierarchy_fast(&child_key, entry.prefix_arcs.as_ref());
                    removed_entries.push((child_key, entry));
                } else {
                    // Fallback: reconstruct arcs from the child key to release prefix refs
                    if let CacheKey::Tuple(parts) = child_key.as_ref() {
                        for i in 1..parts.len() {
                            let prefix = CacheKey::Tuple(parts[0..i].to_vec().into_boxed_slice());
                            let prefix_arc = if let Some(a) = self.prefix_cache.get(&prefix) {
                                a.clone()
                            } else {
                                Arc::new(prefix.clone())
                            };
                            if let Some(children_map) = self.hierarchy.get_mut(&prefix_arc) {
                                children_map.remove(&child_key);
                                if children_map.is_empty() {
                                    self.hierarchy.remove(&prefix_arc);
                                }
                            }
                            self.release_prefix(&prefix);
                        }
                    }
                    self.remove_from_lru(&child_key);
                }
            }
            self.hierarchy.remove(prefix_key);
        }
        
        removed_entries
    }
    
    /// Removes and returns a cache entry, or returns default if not found.
    /// Uses stored prefix references for O(k) cleanup without allocations.
    /// Returns (value, callbacks) tuple for callback execution.
    /// Time complexity: O(k) where k is tuple length.
    pub fn pop(&mut self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        
        if let Some(entry) = self.data.remove(&cache_key) {
            // Update size tracking for removed entry
            if self.size_callback.is_some() {
                let entry_size = self.calculate_size(py, &entry.value).unwrap_or(1);
                self.current_size = self.current_size.saturating_sub(entry_size);
            }
            
            self.remove_from_lru(&cache_key);
            self.remove_hierarchy_fast(&cache_key, entry.prefix_arcs.as_ref());
            let callbacks = entry.get_callbacks();
            Ok((entry.value, callbacks))
        } else {
            Ok((default.unwrap_or_else(|| py.None()), Vec::new()))
        }
    }
    
    /// Removes and returns the least recently used cache entry.
    /// Pure LRU design: if no tail exists, cache is empty.
    /// Returns (original_key, value, callbacks) tuple.
    /// Time complexity: O(k) where k is tuple length for hierarchy cleanup.
    fn popitem(&mut self, _py: Python) -> Option<(Py<PyAny>, Py<PyAny>, Vec<Py<PyAny>>)> {
        rust_debug_fast!("popitem() called, tail: {:?}", self.tail);
        let lru_key = self.remove_lru()?;
        rust_debug_fast!("popitem() removing LRU key: {:?}", lru_key);
        let entry = self.data.remove(&lru_key)?;
        
        // Update size tracking for removed entry
        if self.size_callback.is_some() {
            let entry_size = self.calculate_size(_py, &entry.value).unwrap_or(DEFAULT_ENTRY_SIZE);
            self.current_size = self.current_size.saturating_sub(entry_size);
        }
        
        self.remove_hierarchy_fast(&lru_key, entry.prefix_arcs.as_ref());
        rust_debug_fast!("popitem() removed entry, callbacks count: {}", entry.get_callbacks().len());
        let callbacks = entry.get_callbacks();
        let original_key = entry.original_key;
        let value = entry.value;
        Some((original_key, value, callbacks))
    }
    
    /// Inserts or updates a cache entry with the specified key and value.
    /// Handles capacity management by evicting LRU entries when needed.
    /// Updates LRU position and hierarchy relationships for tuple keys.
    /// Time complexity: O(k) where k is tuple length for hierarchy operations.
    pub fn set(&mut self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        rust_debug_fast!("UnifiedCache::set() called");
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let original_key = key.clone().unbind();
        let mut evicted_callbacks = Vec::new();
        
        if self.data.contains_key(&cache_key) {
            rust_debug_fast!("Key exists, updating entry");
            
            // Calculate sizes before getting mutable reference
            let (old_size, new_size) = if self.size_callback.is_some() {
                let old_val = self.data.get(&cache_key).map(|e| &e.value);
                let old_size = if let Some(old_val) = old_val {
                    self.calculate_size(py, old_val).unwrap_or(DEFAULT_ENTRY_SIZE)
                } else {
                    DEFAULT_ENTRY_SIZE
                };
                let new_size = self.calculate_size(py, &value).unwrap_or(DEFAULT_ENTRY_SIZE);
                (old_size, new_size)
            } else {
                (0, 0)
            };
            
            if let Some(entry) = self.data.get_mut(&cache_key) {
                // Check if value changed and run callbacks if so (Python LruCache behavior)
                let value_changed = !entry.value.bind(py).eq(&value.bind(py)).unwrap_or(false);
                if value_changed {
                    entry.run_and_clear_callbacks(py);
                }
                
                // Update size tracking for existing entry
                if self.size_callback.is_some() {
                    self.current_size = self.current_size.saturating_sub(old_size).saturating_add(new_size);
                }
                
                entry.value = value;
                // Handle callback updates based on policy
                match self.callback_policy {
                    CallbackPolicy::Replace => {
                        if !callbacks.is_empty() {
                            entry.callbacks = Some(callbacks);
                        }
                        // If empty, preserve existing callbacks (Python LruCache behavior)
                    },
                    CallbackPolicy::Append => {
                        entry.add_callbacks(py, callbacks)?;
                    },
                }
            }
            
            if !self.move_to_front(&cache_key) {
                self.add_to_front(cache_key);
            }
        } else {
            rust_debug_fast!("New key, inserting entry");
            // Calculate new entry size for eviction logic
            let new_entry_size = if self.size_callback.is_some() {
                self.calculate_size(py, &value).unwrap_or(DEFAULT_ENTRY_SIZE)
            } else {
                DEFAULT_ENTRY_SIZE
            };
            
            // Evict LRU entries until we have space for the new entry
            while (if self.size_callback.is_some() { self.current_size + new_entry_size } else { self.data.len() + 1 }) > self.capacity {
                rust_debug_fast!("Cache full, evicting LRU entry. Current size: {}, new entry size: {}, capacity: {}", self.len(), new_entry_size, self.capacity);
                if let Some((_, _, callbacks)) = self.popitem(py) {
                    rust_debug_fast!("Evicted entry with {} callbacks", callbacks.len());
                    evicted_callbacks.extend(callbacks);
                } else {
                    rust_debug_fast!("popitem() returned None, breaking eviction loop");
                    break; // No more entries to evict
                }
            }
            
            let prefix_arcs = self.add_hierarchy(&cache_key);
            
            // Update size tracking for new entry (size already calculated above)
            if self.size_callback.is_some() {
                self.current_size = self.current_size.saturating_add(new_entry_size);
            }
            
            // Auto-create node for global eviction if enabled
            let mut entry = CacheEntry::new(value, original_key);
            entry.callbacks = if callbacks.is_empty() { None } else { Some(callbacks) };
            entry.prefix_arcs = if prefix_arcs.is_empty() { None } else { Some(prefix_arcs) };
            
            self.data.insert(cache_key.clone(), entry);
            self.add_to_front(cache_key);
        }
        
        rust_debug_fast!("UnifiedCache::set() completed");
        Ok(evicted_callbacks)
    }
    
    /// Retrieves a cache entry and updates its LRU position.
    /// Records cache hit/miss metrics if metrics are configured.
    /// Time complexity: O(1) for HashMap lookup and LRU update.
    pub fn get(&mut self, py: Python, key: &Bound<'_, PyAny>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<Option<Py<PyAny>>> {
        rust_debug_fast!("UnifiedCache::get() called");
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        
        // Check for inconsistent state between data and lru_nodes
        let data_exists = self.data.contains_key(&cache_key);
        let lru_exists = self.lru_nodes.contains_key(&cache_key);
        
        if data_exists != lru_exists {
            rust_debug_fast!("Consistency error detected in get() - data: {}, lru: {}", data_exists, lru_exists);
            self.check_consistency();
            // Re-check after consistency repair
            if !self.data.contains_key(&cache_key) {
                if let Some(ref metrics) = self.metrics {
                    let _ = metrics.bind(py).call_method1("record_cache_miss", (&self.name,));
                }
                return Ok(None);
            }
        }
        
        if let Some(entry) = self.data.get_mut(&cache_key) {
            rust_debug_fast!("Cache hit, returning value");
            let value = entry.value.clone_ref(py);
            
            // Add callbacks if provided (Python LruCache behavior)
            if let Some(new_callbacks) = callbacks {
                match self.callback_policy {
                    CallbackPolicy::Replace => {
                        if !new_callbacks.is_empty() {
                            entry.callbacks = Some(new_callbacks);
                        }
                        // If empty, preserve existing callbacks (Python add_callbacks behavior)
                    },
                    CallbackPolicy::Append => {
                        entry.add_callbacks(py, new_callbacks)?;
                    },
                }
            }
            

            
            if !self.move_to_front(&cache_key) {
                // If move_to_front fails, try adding to front as fallback
                self.add_to_front(cache_key);
            }
            
            if let Some(ref metrics) = self.metrics {
                let _ = metrics.bind(py).call_method1("record_cache_hit", (&self.name,));
            }
            
            Ok(Some(value))
        } else {
            rust_debug_fast!("Cache miss");
            if let Some(ref metrics) = self.metrics {
                let _ = metrics.bind(py).call_method1("record_cache_miss", (&self.name,));
            }
            Ok(None)
        }
    }
    
    /// Removes a cache entry and performs full cleanup.
    /// Returns (found, callbacks) tuple indicating success and any callbacks to execute.
    /// Time complexity: O(k) where k is tuple length for hierarchy cleanup.
    pub fn invalidate(&mut self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<Py<PyAny>>)> {
        rust_debug_fast!("UnifiedCache::invalidate() called");
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        self.invalidate_by_key(py, &cache_key)
    }
    
    /// Invalidates a cache entry by its CacheKey directly
    pub fn invalidate_by_key(&mut self, py: Python, key: &Arc<CacheKey>) -> PyResult<(bool, Vec<Py<PyAny>>)> {
        if let Some(entry) = self.data.remove(key) {
            rust_debug_fast!("Entry found, removing from cache and LRU");
            
            // Update size tracking for removed entry
            if self.size_callback.is_some() {
                let entry_size = self.calculate_size(py, &entry.value).unwrap_or(1);
                self.current_size = self.current_size.saturating_sub(entry_size);
            }
            

            
            self.remove_from_lru(key);
            self.remove_hierarchy_fast(key, entry.prefix_arcs.as_ref());
            Ok((true, entry.get_callbacks()))
        } else {
            rust_debug_fast!("Entry not found");
            Ok((false, Vec::new()))
        }
    }
    
    /// Gets a cache entry by its CacheKey directly
    pub fn get_by_key(&self, key: &Arc<CacheKey>) -> Option<&CacheEntry> {
        self.data.get(key)
    }
    
    /// Updates LRU position for a key without returning the value
    pub fn touch_key(&mut self, key: &Arc<CacheKey>) {
        if self.data.contains_key(key) {
            self.move_to_front(key);
        }
    }
    
    /// Removes all entries from the cache and resets all internal state.
    /// Collects all callbacks from removed entries for execution.
    /// Time complexity: O(n) where n is the number of cache entries.
    pub fn clear(&mut self, py: Python) -> (usize, Vec<Py<PyAny>>) {
        let count = self.len();
        rust_debug_fast!("UnifiedCache::clear() called for cache '{}' with {} entries", self.name, count);
        
        let mut all_callbacks = Vec::new();
        for entry in self.data.values() {
            all_callbacks.extend(entry.get_callbacks().iter().map(|cb| cb.clone_ref(py)));
        }
        
        self.data.clear();
        self.hierarchy.clear();
        self.prefix_cache.clear();
        self.prefix_counts.clear();
        self.lru_nodes.clear();
        self.head = None;
        self.tail = None;
        self.current_size = 0;
        
        rust_debug_fast!("UnifiedCache::clear() completed for cache '{}', cleared {} entries, {} callbacks", self.name, count, all_callbacks.len());
        (count, all_callbacks)
    }
    
    /// Returns the current number of entries in the cache.
    /// Time complexity: O(1).
    pub fn len(&self) -> usize {
        if self.size_callback.is_some() {
            self.current_size
        } else {
            self.data.len()
        }
    }
    
    /// Checks if a key exists in the cache without updating LRU position.
    /// Pure read operation that doesn't modify cache state.
    /// Time complexity: O(1).
    pub(crate) fn contains(&self, key: &Arc<CacheKey>) -> bool {
        self.data.contains_key(key)
    }
    
    /// Returns the maximum capacity of the cache.
    /// Time complexity: O(1).
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Sets the size callback for calculating entry sizes.
    pub fn set_size_callback(&mut self, py: Python, callback: Option<Py<PyAny>>) {
        let has_callback = callback.is_some();
        self.size_callback = callback;
        
        // Recompute current size from existing entries
        if has_callback {
            self.current_size = 0;
            for entry in self.data.values() {
                let entry_size = self.calculate_size(py, &entry.value).unwrap_or(DEFAULT_ENTRY_SIZE);
                self.current_size = self.current_size.saturating_add(entry_size);
            }
        } else {
            self.current_size = 0;
        }
    }
    
    /// Calculates the size of a value using the size callback.
    fn calculate_size(&self, py: Python, value: &Py<PyAny>) -> PyResult<usize> {
        if let Some(ref callback) = self.size_callback {
            let result = callback.bind(py).call1((value,))?;
            Ok(result.extract()?)
        } else {
            Ok(DEFAULT_ENTRY_SIZE)
        }
    }
    
    /// Checks and repairs cache consistency between data and lru_nodes.
    /// Returns true if inconsistencies were found and repaired.
    /// Time complexity: O(n) where n is the number of cache entries.
    fn check_consistency(&mut self) -> bool {
        let mut inconsistent = false;
        
        // Check that every data entry has a corresponding lru_node
        let data_keys: Vec<_> = self.data.keys().cloned().collect();
        for key in &data_keys {
            if !self.lru_nodes.contains_key(key) {
                rust_debug_fast!("Consistency error: data key {:?} missing from lru_nodes, reconstructing", key);
                // Reconstruct missing LRU node instead of dropping data entry
                let node = Box::new(LruNode { prev: None, next: None });
                self.lru_nodes.insert(key.clone(), node);
                inconsistent = true;
            }
        }
        
        // Check that every lru_node has a corresponding data entry
        let lru_keys: Vec<_> = self.lru_nodes.keys().cloned().collect();
        for key in &lru_keys {
            if !self.data.contains_key(key) {
                rust_debug_fast!("Consistency error: lru_node key {:?} missing from data", key);
                // Remove orphaned lru_nodes
                self.lru_nodes.remove(key);
                inconsistent = true;
            }
        }
        
        // If we found inconsistencies, rebuild the LRU chain
        if inconsistent {
            rust_debug_fast!("Rebuilding LRU chain due to inconsistencies");
            self.head = None;
            self.tail = None;
            
            // Rebuild LRU chain for all data entries (now all have nodes)
            let valid_keys: Vec<_> = self.data.keys().cloned().collect();
            for key in valid_keys {
                self.add_to_front(key);
            }
        }
        
        inconsistent
    }
    
    /// Resizes the cache to a new capacity, evicting entries if necessary.
    /// Time complexity: O(evicted_entries * k) where k is average tuple length.
    pub fn resize(&mut self, py: Python, new_capacity: usize) -> Vec<Py<PyAny>> {
        let mut evicted_callbacks = Vec::new();
        self.capacity = new_capacity;
        
        // Evict entries if we're over capacity
        while (if self.size_callback.is_some() { self.current_size } else { self.data.len() }) > new_capacity {
            if let Some((_, _, callbacks)) = self.popitem(py) {
                evicted_callbacks.extend(callbacks);
            } else {
                break;
            }
        }
        
        evicted_callbacks
    }
    
    /// Atomic get-or-set operation - returns existing value or sets and returns new value.
    /// Time complexity: O(1) for HashMap lookup, O(k) for hierarchy if new tuple key.
    pub fn setdefault(&mut self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        
        // Check if key exists first
        if let Some(entry) = self.data.get(&cache_key) {
            let existing_value = entry.value.clone_ref(py);
            // Move to front after getting value
            if !self.move_to_front(&cache_key) {
                self.add_to_front(cache_key);
            }
            Ok(existing_value)
        } else {
            // Key doesn't exist, set it and return the value
            let value_clone = value.clone_ref(py);
            let evicted_callbacks = self.set(py, key, value_clone, Vec::new())?;
            // Execute callbacks outside the method to avoid borrow issues
            for callback in evicted_callbacks {
                if let Err(e) = callback.bind(py).call0() {
                    eprintln!("Cache callback error: {}", e);
                }
            }
            Ok(value)
        }
    }
    
    /// Gets all entries that have the given tuple prefix.
    /// Returns Vec of (original_key, value) pairs.
    /// Time complexity: O(children) where children is number of matching entries.
    pub fn get_prefix_children(&mut self, py: Python, prefix_key: &Bound<'_, PyAny>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let cache_key = Arc::new(CacheKey::from_bound(prefix_key)?);
        let mut results = Vec::new();
        
        // Collect child keys first to avoid borrow checker issues
        let child_keys: Vec<Arc<CacheKey>> = if let Some(children) = self.hierarchy.get(&cache_key) {
            children.keys().cloned().collect()
        } else {
            Vec::new()
        };
        
        // Now update LRU and collect results
        for child_key in child_keys {
            if let Some(entry) = self.data.get(&child_key) {
                // Clone values before mutable operations
                let original_key = entry.original_key.clone_ref(py);
                let value = entry.value.clone_ref(py);
                
                // Update LRU position for accessed entries
                if !self.move_to_front(&child_key) {
                    self.add_to_front(child_key.clone());
                }
                
                results.push((original_key, value));
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
        match self.key.as_ref() {
            CacheKey::String(s) => Ok(PyString::new(py, s).into()),
            CacheKey::IntSmall(i) => Ok(PyInt::new(py, *i).into()),
            CacheKey::IntBig(bi) => {
                let s = bi.to_string();
                Ok(PyString::new(py, &s).into())
            },
            CacheKey::None => Ok(py.None()),
            CacheKey::Tuple(parts) => {
                let py_parts: PyResult<Vec<_>> = parts.iter()
                    .map(|part| {
                        match part {
                            CacheKey::String(s) => Ok(PyString::new(py, s).into()),
                            CacheKey::IntSmall(i) => Ok(PyInt::new(py, *i).into()),
                            CacheKey::None => Ok(py.None()),
                            _ => Ok(PyString::new(py, &format!("{:?}", part)).into())
                        }
                    })
                    .collect();
                let tuple = PyTuple::new(py, py_parts?)?;
                Ok(tuple.into())
            },
            CacheKey::Hashed { type_name, py_hash, obj_id } => {
                Ok(PyString::new(py, &format!("{}#{}@{}", type_name, py_hash, obj_id)).into())
            }
        }
    }
    
    #[getter]
    fn value(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            match cache.get_by_key(&self.key) {
                Some(entry) => Ok(Some(entry.value.clone_ref(py))),
                None => Ok(None)
            }
        } else {
            Ok(None) // Cache has been dropped
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
                let mut cache = cache_arc.write();
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
    
    fn update_last_access(&self, clock: Option<Py<PyAny>>) {
        // Use Python clock time if provided, otherwise use relative time
        let now = if let Some(clock_obj) = clock {
            Python::with_gil(|py| {
                if let Ok(time_method) = clock_obj.bind(py).getattr("time") {
                    if let Ok(time_result) = time_method.call0() {
                        if let Ok(time_seconds) = time_result.extract::<f64>() {
                            return python_time_to_relative_ms(time_seconds);
                        }
                    }
                }
                get_relative_time_ms()
            })
        } else {
            get_relative_time_ms()
        };
        
        self.last_access_time.store(now, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
        
        // Update LRU position in cache
        if let Some(cache_arc) = self.cache.upgrade() {
            let mut cache = cache_arc.write();
            cache.touch_key(&self.key);
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
            if let Some(entry) = cache.get_by_key(&self.key) {
                cache.calculate_size(py, &entry.value)
            } else {
                Ok(DEFAULT_ENTRY_SIZE) // Default size if entry not found
            }
        } else {
            Ok(DEFAULT_ENTRY_SIZE) // Cache has been dropped
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
            false // Cache has been dropped
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
    
    /// Add to global eviction list (for time-based eviction)
    fn add_to_global_list(&mut self, prev_id: Option<u64>, next_id: Option<u64>) {
        self.prev_node = prev_id;
        self.next_node = next_id;
        self.in_global_list = true;
    }
    
    /// Remove from global eviction list
    fn remove_from_global_list(&mut self) -> (Option<u64>, Option<u64>) {
        let prev = self.prev_node;
        let next = self.next_node;
        self.prev_node = None;
        self.next_node = None;
        self.in_global_list = false;
        (prev, next)
    }
    
    /// Move to front of global eviction list
    fn move_to_front_global(&mut self, new_next: Option<u64>) {
        self.prev_node = None;
        self.next_node = new_next;
    }
    
    /// Check if node is in global eviction list
    fn is_in_global_list(&self) -> bool {
        self.in_global_list
    }
    
    /// Compare current value with new value (for change detection)
    fn value_changed(&self, py: Python, new_value: Py<PyAny>) -> PyResult<bool> {
        if let Some(cache_arc) = self.cache.upgrade() {
            let cache = cache_arc.read();
            if let Some(entry) = cache.get_by_key(&self.key) {
                match entry.value.bind(py).eq(&new_value.bind(py)) {
                    Ok(is_equal) => Ok(!is_equal),
                    Err(e) => {
                        // If comparison fails, assume it's a change to be safe
                        eprintln!("Warning: Failed to compare values in cache: {}", e);
                        Ok(true)
                    }
                }
            } else {
                Ok(true) // No current value, so it's a change
            }
        } else {
            Ok(true) // Cache has been dropped, consider it a change
        }
    }
    
    /// Set value and run callbacks if changed
    fn set_value_if_changed(&mut self, py: Python, new_value: Py<PyAny>) -> PyResult<bool> {
        let changed = self.value_changed(py, new_value.clone_ref(py))?;
        if changed {
            // Update cache value
            if let Some(cache_arc) = self.cache.upgrade() {
                let mut cache = cache_arc.write();
                if let Some(entry) = cache.data.get_mut(&self.key) {
                    entry.value = new_value;
                    entry.run_and_clear_callbacks(py);
                }
            }
            // Run node callbacks
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
    
    /// Remove from global eviction system
    fn remove_from_global_eviction(&mut self) -> bool {
        if self.in_global_list {
            self.in_global_list = false;
            self.prev_node = None;
            self.next_node = None;
            true
        } else {
            false
        }
    }
    

}

#[pymethods]
impl RustLruCache {
    /// Creates a new thread-safe Rust LRU cache with the specified configuration.
    /// Wraps UnifiedCache in Arc<Mutex<>> for safe concurrent access from Python.
    #[new]
    #[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None, size_callback=None))]
    fn new(
        max_size: usize, 
        cache_name: Option<String>, 
        metrics: Option<Py<PyAny>>,
        callback_policy: Option<String>,
        size_callback: Option<Py<PyAny>>
    ) -> Self {
        let name = cache_name.unwrap_or_else(|| "rust_cache".to_string());
        let policy = match callback_policy.as_deref() {
            Some("append") => CallbackPolicy::Append,
            _ => CallbackPolicy::Replace,
        };
        let cache = Arc::new(RwLock::new(UnifiedCache::new(name.clone(), max_size, metrics, policy)));
        
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
        let mut cache = self.cache.write();
        match cache.get(py, key, callbacks)? {
            Some(value) => Ok(value),
            None => Ok(default.unwrap_or_else(|| py.None()))
        }
    }
    
    /// Inserts or updates a cache entry with automatic callback execution.
    /// Executes evicted callbacks outside the mutex lock to prevent deadlocks.
    fn set(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<()> {
        let callbacks = callbacks.unwrap_or_default();
        
        let (evicted_callbacks, needs_node) = {
            let mut cache = self.cache.write();
            let result = cache.set(py, key, value, callbacks)?;
            
            // Check if we need to create a node
            let needs_node = if cache.enable_nodes {
                let cache_key = Arc::new(CacheKey::from_bound(key)?);
                cache.data.get(&cache_key)
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
                cache.create_eviction_node(&self.cache, py, &cache_key, &[])
            }?;
            
            // Then update the entry
            let mut cache = self.cache.write();
            if let Some(entry) = cache.data.get_mut(&cache_key) {
                if entry.node.is_none() {
                    entry.node = Some(node);
                }
            }
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
            let mut cache = self.cache.write();
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
            let mut cache = self.cache.write();
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
            let mut cache = self.cache.write();
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
            let mut cache = self.cache.write();
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
            let mut cache = self.cache.write();
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
    
    /// Python len() support - returns current number of cache entries.
    fn __len__(&self) -> PyResult<usize> {
        let cache = self.cache.read();
        Ok(if cache.size_callback.is_some() {
            cache.current_size
        } else {
            cache.data.len()
        })
    }
    
    /// Python 'in' operator support - checks if key exists in cache.
    fn __contains__(&self, _py: Python, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        Ok(self.cache.read().contains(&cache_key))
    }
    
    /// Python dict[key] support - returns value or raises KeyError.
    fn __getitem__(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mut cache = self.cache.write();
        match cache.get(py, key, None)? {
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
            let mut cache = self.cache.write();
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
        let mut cache = self.cache.write();
        cache.get_prefix_children(py, prefix_key)
    }
    
    /// Checks if a key exists in the cache without updating LRU position.
    fn contains(&self, _py: Python, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        Ok(self.cache.read().contains(&cache_key))
    }
    
    /// Sets the size callback for calculating entry sizes.
    fn set_size_callback(&self, py: Python, callback: Option<Py<PyAny>>) -> PyResult<()> {
        let mut cache = self.cache.write();
        cache.set_size_callback(py, callback);
        Ok(())
    }
    
    /// Atomic get-or-set operation
    fn setdefault(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let mut cache = self.cache.write();
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
        Ok((cache.data.len(), cache.capacity))
    }
    
    /// Check if cache is empty
    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.cache.read().data.is_empty())
    }
    
    /// Get estimated memory usage in bytes (matches original LruCache behavior)
    fn get_memory_usage(&self) -> PyResult<usize> {
        let cache = self.cache.read();
        let mut total = 0;
        
        // Cache structure overhead
        total += cache.data.capacity() * std::mem::size_of::<(Arc<CacheKey>, CacheEntry)>();
        total += cache.lru_nodes.capacity() * std::mem::size_of::<(Arc<CacheKey>, Box<LruNode>)>();
        
        // Per-entry memory usage
        for (key, entry) in &cache.data {
            total += key.memory_size();
            total += entry.memory_size();
        }
        
        // LRU node overhead
        total += cache.lru_nodes.len() * std::mem::size_of::<LruNode>();
        
        Ok(total)
    }
    
    /// Enhanced get with update_metrics and update_last_access parameters
    fn get_advanced(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>, callbacks: Option<Vec<Py<PyAny>>>, _update_metrics: Option<bool>, _update_last_access: Option<bool>) -> PyResult<Py<PyAny>> {
        let mut cache = self.cache.write();
        let cb_list = callbacks.unwrap_or_default();
        
        // For now, ignore update_metrics and update_last_access parameters
        // They would require more sophisticated integration with metrics system
        match cache.get(py, key, Some(cb_list))? {
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
    
    /// Get all active cache nodes for global eviction
    fn get_all_nodes(&self) -> PyResult<Vec<Py<RustCacheNode>>> {
        let cache = self.cache.read();
        Ok(cache.get_all_nodes())
    }
    
    /// Get node for key from cache (for global eviction integration)
    fn get_node_for_key(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<Option<Py<RustCacheNode>>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        let cache = self.cache.read();
        
        Ok(cache.data.get(&cache_key)
            .and_then(|entry| entry.node.as_ref())
            .map(|node| node.clone_ref(py)))
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
        let cache = Arc::new(AsyncRwLock::new(UnifiedCache::new(name.clone(), max_size, metrics, policy)));
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
            let mut cache = cache.write().await;
            Python::with_gil(|py| {
                let key_bound = key.bind(py);
                match cache.get(py, &key_bound, None)? {
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
                let mut cache = cache.write().await;
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
                let mut cache = cache.write().await;
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
                let mut cache = cache.write().await;
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
                Ok(mut cache) => {
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
    Ok(RustLruCache::new(max_size, cache_name, metrics, callback_policy, size_callback))
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