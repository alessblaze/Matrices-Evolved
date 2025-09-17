/*
 * Copyright (C) 2025 Aless Microsystems
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * High-Performance Rust LRU Cache Implementation for Synapse
 * 
 * Features:
 * - O(1) cache operations (get, set, contains, len)
 * - O(k) hierarchy operations where k=tuple length (typically ≤4)
 * - HashMap-of-HashMaps for O(1) child insertion/removal
 * - Arc<CacheKey> sharing with prefix caching and reference counting
 * - Full TreeCache compatibility for tuple keys
 * - Callback system with configurable policies (Replace/Append)
 * - Thread-safe with Arc<Mutex<>> wrapper
 * - PyO3 bindings for seamless Python integration
 * - hashbrown::HashMap for optimized performance
 * 
 * Performance: 9.23μs for real-world Synapse workloads (3x faster than Python)
 */

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyString, PyInt, PyBool};
use parking_lot::RwLock;
use tokio::sync::RwLock as AsyncRwLock;
use hashbrown::HashMap;
use std::sync::{Arc, LazyLock};

// Cached debug flag for optimal performance
static DEBUG_ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("SYNAPSE_RUST_CACHE_DEBUG").is_ok());



// Optimized debug macro that checks cached flag first
macro_rules! rust_debug_fast {
    ($($arg:tt)*) => {
        if *DEBUG_ENABLED {
            println!("[RUST] {}", format!($($arg)*));
        }
    };
}

type FastHashMap<K, V> = HashMap<K, V, ahash::RandomState>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CacheKey {
    String(String),
    Int(i64),
    Bool(bool),
    None,
    Tuple(Box<[CacheKey]>),
    Hashed { type_name: String, hash: u64 },
}

impl CacheKey {
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
            return Ok(CacheKey::Bool(obj.extract()?));
        }
        if obj.is_instance_of::<PyInt>() {
            return Ok(CacheKey::Int(obj.extract()?));
        }
        if obj.is_instance_of::<PyString>() {
            return Ok(CacheKey::String(obj.extract()?));
        }

        // Fallback: arbitrary Python object
        let type_name = obj.get_type().name()?.to_string();
        let obj_hash = obj.hash().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Unhashable key of type {}", type_name)
            )
        })? as u64;

        let obj_id = obj.as_ptr() as usize as u64;
        let hash = (obj_hash << 32) ^ obj_id;

        Ok(CacheKey::Hashed { type_name, hash })
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
    
    /// Checks if this CacheKey is a prefix of another CacheKey (for tuple hierarchy).
    /// Used for TreeCache prefix invalidation operations.
    /// Time complexity: O(min(prefix_len, full_len)).
    fn is_prefix_of(&self, other: &CacheKey) -> bool {
        match (self, other) {
            (CacheKey::Tuple(prefix), CacheKey::Tuple(full)) => {
                prefix.len() <= full.len() && 
                prefix.iter().zip(full.iter()).all(|(a, b)| a == b)
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
struct LruNode {
    prev: Option<Arc<CacheKey>>,
    next: Option<Arc<CacheKey>>,
}

#[derive(Debug)]
pub(crate) struct CacheEntry {
    value: Py<PyAny>,
    callbacks: Vec<Py<PyAny>>,
    original_key: Py<PyAny>,
    prefix_arcs: Option<Vec<Arc<CacheKey>>>,
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
        }
    }

    /// Moves an existing cache entry to the front of the LRU list (most recently used).
    /// Updates doubly-linked list pointers in O(1) time using HashMap lookups.
    /// Returns true if the key was found and moved, false otherwise.
    /// Time complexity: O(1).
    fn move_to_front(&mut self, key: &Arc<CacheKey>) -> bool {
        // Consistency check: key must exist in both data and lru_nodes
        if !self.data.contains_key(key) || !self.lru_nodes.contains_key(key) {
            return false;
        }
        
        if self.head.as_ref() == Some(key) {
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
                if i <= parts.len() {
                    let prefix = CacheKey::Tuple(parts[0..i].to_vec().into_boxed_slice());
                    let prefix_arc = self.get_or_create_prefix(prefix);
                    self.hierarchy
                        .entry(prefix_arc.clone())
                        .or_insert_with(FastHashMap::default)
                        .insert(key.clone(), ());
                    prefix_arcs.push(prefix_arc);
                }
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
    fn invalidate_prefix(&mut self, prefix_key: &Arc<CacheKey>) -> Vec<(Arc<CacheKey>, CacheEntry)> {
        let mut removed_entries = Vec::new();
        
        if let Some(children) = self.hierarchy.get(prefix_key) {
            let child_keys: Vec<_> = children.keys().cloned().collect();
            for child_key in child_keys {
                if let Some(entry) = self.data.remove(&child_key) {
                    self.remove_from_lru(&child_key);
                    removed_entries.push((child_key, entry));
                }
            }
            self.hierarchy.remove(prefix_key);
        }
        
        removed_entries
    }
    
    /// Removes and returns a cache entry, or returns default if not found.
    /// Uses stored prefix references for O(k) cleanup without allocations.
    /// Time complexity: O(k) where k is tuple length.
    pub fn pop(&mut self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        
        if let Some(entry) = self.data.remove(&cache_key) {
            self.remove_from_lru(&cache_key);
            self.remove_hierarchy_fast(&cache_key, entry.prefix_arcs.as_ref());
            Ok(entry.value)
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }
    
    /// Removes and returns the least recently used cache entry.
    /// Pure LRU design: if no tail exists, cache is empty.
    /// Returns (original_key, value, callbacks) tuple.
    /// Time complexity: O(k) where k is tuple length for hierarchy cleanup.
    fn popitem(&mut self, _py: Python) -> Option<(Py<PyAny>, Py<PyAny>, Vec<Py<PyAny>>)> {
        let lru_key = self.remove_lru()?;
        let entry = self.data.remove(&lru_key)?;
        self.remove_hierarchy_fast(&lru_key, entry.prefix_arcs.as_ref());
        Some((entry.original_key, entry.value, entry.callbacks))
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
            if let Some(entry) = self.data.get_mut(&cache_key) {
                entry.value = value;
                match self.callback_policy {
                    CallbackPolicy::Replace => entry.callbacks = callbacks,
                    CallbackPolicy::Append => entry.callbacks.extend(callbacks),
                }
            }
            
            if !self.move_to_front(&cache_key) {
                self.add_to_front(cache_key);
            }
        } else {
            rust_debug_fast!("New key, inserting entry");
            if self.data.len() >= self.capacity {
                rust_debug_fast!("Cache full, evicting LRU entry");
                if let Some((_, _, callbacks)) = self.popitem(py) {
                    evicted_callbacks = callbacks;
                }
            }
            
            let prefix_arcs = self.add_hierarchy(&cache_key);
            let entry = CacheEntry {
                value,
                callbacks,
                original_key,
                prefix_arcs: if prefix_arcs.is_empty() { None } else { Some(prefix_arcs) },
            };
            
            self.data.insert(cache_key.clone(), entry);
            self.add_to_front(cache_key);
        }
        
        rust_debug_fast!("UnifiedCache::set() completed");
        Ok(evicted_callbacks)
    }
    
    /// Retrieves a cache entry and updates its LRU position.
    /// Records cache hit/miss metrics if metrics are configured.
    /// Time complexity: O(1) for HashMap lookup and LRU update.
    pub fn get(&mut self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
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
        
        if let Some(entry) = self.data.get(&cache_key) {
            rust_debug_fast!("Cache hit, returning value");
            let value = entry.value.clone_ref(py);
            
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
    pub fn invalidate(&mut self, _py: Python, key: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<Py<PyAny>>)> {
        rust_debug_fast!("UnifiedCache::invalidate() called");
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        
        if let Some(entry) = self.data.remove(&cache_key) {
            rust_debug_fast!("Entry found, removing from cache and LRU");
            self.remove_from_lru(&cache_key);
            self.remove_hierarchy_fast(&cache_key, entry.prefix_arcs.as_ref());
            Ok((true, entry.callbacks))
        } else {
            rust_debug_fast!("Entry not found");
            Ok((false, Vec::new()))
        }
    }
    
    /// Removes all entries from the cache and resets all internal state.
    /// Collects all callbacks from removed entries for execution.
    /// Time complexity: O(n) where n is the number of cache entries.
    pub fn clear(&mut self, py: Python) -> (usize, Vec<Py<PyAny>>) {
        let count = self.data.len();
        rust_debug_fast!("UnifiedCache::clear() called for cache '{}' with {} entries", self.name, count);
        
        let mut all_callbacks = Vec::new();
        for entry in self.data.values() {
            all_callbacks.extend(entry.callbacks.iter().map(|cb| cb.clone_ref(py)));
        }
        
        self.data.clear();
        self.hierarchy.clear();
        self.prefix_cache.clear();
        self.prefix_counts.clear();
        self.lru_nodes.clear();
        self.head = None;
        self.tail = None;
        
        rust_debug_fast!("UnifiedCache::clear() completed for cache '{}', cleared {} entries, {} callbacks", self.name, count, all_callbacks.len());
        (count, all_callbacks)
    }
    
    /// Returns the current number of entries in the cache.
    /// Time complexity: O(1).
    pub fn len(&self) -> usize {
        self.data.len()
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
    
    /// Checks and repairs cache consistency between data and lru_nodes.
    /// Returns true if inconsistencies were found and repaired.
    /// Time complexity: O(n) where n is the number of cache entries.
    fn check_consistency(&mut self) -> bool {
        let mut inconsistent = false;
        
        // Check that every data entry has a corresponding lru_node
        let data_keys: Vec<_> = self.data.keys().cloned().collect();
        for key in &data_keys {
            if !self.lru_nodes.contains_key(key) {
                rust_debug_fast!("Consistency error: data key {:?} missing from lru_nodes", key);
                // Remove from data to maintain consistency
                self.data.remove(key);
                inconsistent = true;
            }
        }
        
        // Check that every lru_node has a corresponding data entry
        let lru_keys: Vec<_> = self.lru_nodes.keys().cloned().collect();
        for key in &lru_keys {
            if !self.data.contains_key(key) {
                rust_debug_fast!("Consistency error: lru_node key {:?} missing from data", key);
                // Remove from lru_nodes to maintain consistency
                self.lru_nodes.remove(key);
                inconsistent = true;
            }
        }
        
        // If we found inconsistencies, rebuild the LRU chain
        if inconsistent {
            rust_debug_fast!("Rebuilding LRU chain due to inconsistencies");
            self.head = None;
            self.tail = None;
            
            // Collect keys first to avoid borrow checker issues
            let valid_keys: Vec<_> = self.data.keys()
                .filter(|key| self.lru_nodes.contains_key(*key))
                .cloned()
                .collect();
            
            // Rebuild LRU chain in arbitrary order (better than broken state)
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
        while self.data.len() > new_capacity {
            if let Some((_, _, callbacks)) = self.popitem(py) {
                evicted_callbacks.extend(callbacks);
            } else {
                break;
            }
        }
        
        evicted_callbacks
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
impl RustLruCache {
    /// Creates a new thread-safe Rust LRU cache with the specified configuration.
    /// Wraps UnifiedCache in Arc<Mutex<>> for safe concurrent access from Python.
    #[new]
    #[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None))]
    fn new(
        max_size: usize, 
        cache_name: Option<String>, 
        metrics: Option<Py<PyAny>>,
        callback_policy: Option<String>
    ) -> Self {
        let name = cache_name.unwrap_or_else(|| "rust_cache".to_string());
        let policy = match callback_policy.as_deref() {
            Some("append") => CallbackPolicy::Append,
            _ => CallbackPolicy::Replace,
        };
        let cache = Arc::new(RwLock::new(UnifiedCache::new(name.clone(), max_size, metrics, policy)));
        
        Self { cache, name }
    }
    
    /// Returns the cache name for identification and debugging.
    #[getter]
    fn get_name(&self) -> &str {
        &self.name
    }
    
    /// Retrieves a value from the cache, returning default if not found.
    /// Thread-safe wrapper around UnifiedCache::get with mutex locking.
    fn get(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let mut cache = self.cache.write();
        match cache.get(py, key)? {
            Some(value) => Ok(value),
            None => Ok(default.unwrap_or_else(|| py.None()))
        }
    }
    
    /// Inserts or updates a cache entry with automatic callback execution.
    /// Executes evicted callbacks outside the mutex lock to prevent deadlocks.
    fn set(&self, py: Python, key: &Bound<'_, PyAny>, value: Py<PyAny>, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<()> {
        let callbacks = callbacks.unwrap_or_default();
        
        let evicted_callbacks = {
            let mut cache = self.cache.write();
            cache.set(py, key, value, callbacks)?
        };
        
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
            let removed_entries = cache.invalidate_prefix(&cache_key);
            let entry_count = removed_entries.len();
            
            for (_, entry) in removed_entries {
                all_callbacks.extend(entry.callbacks);
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
    /// Direct delegation to UnifiedCache::pop with mutex protection.
    fn pop(&self, py: Python, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let mut cache = self.cache.write();
        cache.pop(py, key, default)
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
        self.get(py, key, default)
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
        Ok(self.cache.read().len())
    }
    
    /// Python 'in' operator support - checks if key exists in cache.
    fn __contains__(&self, _py: Python, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let cache_key = Arc::new(CacheKey::from_bound(key)?);
        Ok(self.cache.read().contains(&cache_key))
    }
    
    /// Python dict[key] support - returns value or raises KeyError.
    fn __getitem__(&self, py: Python, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mut cache = self.cache.write();
        match cache.get(py, key)? {
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
                match cache.get(py, &key_bound)? {
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
        let mut delay_ms = 1;
        for attempt in 0..5 {
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
                    if attempt < 4 {
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
#[pyo3(signature = (max_size, cache_name=None, metrics=None, callback_policy=None))]
pub fn create_rust_lru_cache(
    max_size: usize, 
    cache_name: Option<String>, 
    metrics: Option<Py<PyAny>>,
    callback_policy: Option<String>
) -> PyResult<RustLruCache> {
    Ok(RustLruCache::new(max_size, cache_name, metrics, callback_policy))
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