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
use pyo3::types::{PySet};
use std::collections::BTreeMap;
use hashbrown::{HashMap, HashSet};
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_ENTITIES_RESULT: usize = 100_000; // Cap for get_all_entities_changed
const MAX_ENTITIES_TOTAL: usize = 1_000_000; // Emergency brake for total entity count

#[pyclass]
pub struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    stale: AtomicU64,
}

#[pymethods]
impl CacheMetrics {
    fn inc_hits(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }
    
    fn inc_misses(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
    
    fn inc_stale(&self) {
        self.stale.fetch_add(1, Ordering::Relaxed);
    }
    
    #[getter]
    fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    
    #[getter] 
    fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
    
    #[getter]
    fn stale(&self) -> u64 {
        self.stale.load(Ordering::Relaxed)
    }
    
    fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.stale.store(0, Ordering::Relaxed);
    }
    
    fn __len__(&self) -> usize {
        (self.hits() + self.misses() + self.stale()) as usize
    }
    
    fn __repr__(&self) -> String {
        format!("CacheMetrics(hits={}, misses={}, stale={})", self.hits(), self.misses(), self.stale())
    }
}

#[pyclass]
pub struct AllEntitiesChangedResult {
    entities: Option<Arc<Vec<Arc<str>>>>,
    truncated: bool,
}

#[pyclass]
struct AllEntitiesChangedResultIter {
    entities: Arc<Vec<Arc<str>>>,
    index: usize,
}

#[pymethods]
impl AllEntitiesChangedResultIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self) -> Option<String> {
        if self.index < self.entities.len() {
            let result = self.entities[self.index].to_string();
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[pymethods]
impl AllEntitiesChangedResult {
    #[getter]
    fn hit(&self) -> bool {
        self.entities.is_some()
    }
    
    #[getter]
    fn entities(&self) -> Vec<String> {
        self.entities.as_ref().map_or_else(Vec::new, |v| v.iter().map(|s| s.to_string()).collect())
    }
    
    #[getter]
    fn truncated(&self) -> bool {
        self.truncated
    }
    
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<AllEntitiesChangedResultIter> {
        Ok(AllEntitiesChangedResultIter {
            entities: slf.entities.clone().unwrap_or_else(|| Arc::new(Vec::new())),
            index: 0,
        })
    }
    
    fn __len__(&self) -> usize {
        self.entities.as_ref().map_or(0, |v| v.len())
    }
    
    fn __repr__(&self) -> String {
        format!("AllEntitiesChangedResult(hit={}, count={}, truncated={})", 
                self.hit(), 
                self.entities.as_ref().map_or(0, |e| e.len()),
                self.truncated)
    }
}

// Single state structure to avoid multiple locks and deadlock risks
struct CacheState {
    cache: BTreeMap<i64, HashSet<Arc<str>>>,
    entity_to_key: HashMap<Arc<str>, i64>,
    earliest_known_stream_pos: i64,
    max_positions: usize,
}

#[pyclass]
pub struct RustStreamChangeCache {
    state: Arc<RwLock<CacheState>>,
    original_max_positions: usize,
    name: String,
    server_name: String,
    metrics: Arc<CacheMetrics>,
}

#[pymethods]
impl RustStreamChangeCache {
    #[classattr]
    const MAX_ENTITIES_RESULT: usize = MAX_ENTITIES_RESULT;
    #[new]
    #[pyo3(signature = (*, name, server_name, current_stream_pos, max_size=10000, prefilled_cache=None))]
    fn new(
        name: String,
        server_name: String,
        current_stream_pos: i64,
        max_size: usize,
        prefilled_cache: Option<std::collections::HashMap<String, i64>>,
    ) -> PyResult<Self> {
        let state = Arc::new(RwLock::new(CacheState {
            cache: BTreeMap::new(),
            entity_to_key: HashMap::new(),
            earliest_known_stream_pos: current_stream_pos,
            max_positions: max_size,
        }));
        
        let metrics = Arc::new(CacheMetrics {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stale: AtomicU64::new(0),
        });
        
        let instance = Self {
            state,
            original_max_positions: max_size,
            name,
            server_name,
            metrics,
        };
        
        // Prefill cache if provided
        if let Some(prefill) = prefilled_cache {
            if !prefill.is_empty() {
                let mut state = instance.state.write();
                let mut min_prefill_pos = current_stream_pos;
                for (entity, stream_pos) in prefill {
                    let entity_key: Arc<str> = Arc::from(entity);
                    state.entity_to_key.insert(entity_key.clone(), stream_pos);
                    state.cache.entry(stream_pos).or_insert_with(HashSet::new).insert(entity_key);
                    min_prefill_pos = min_prefill_pos.min(stream_pos);
                }
                // Clamp earliest position to prevent rewinding too far
                let min_allowed = current_stream_pos.saturating_sub(max_size as i64);
                state.earliest_known_stream_pos = min_prefill_pos.max(min_allowed);
            }
        }
        
        Ok(instance)
    }
    
    fn set_cache_factor(&self, factor: f64) -> PyResult<usize> {
        // Validate factor
        if !factor.is_finite() || factor <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cache factor must be finite and positive, got: {}", factor)
            ));
        }
        
        let new_size = ((self.original_max_positions as f64 * factor).floor() as usize).max(1);
        let mut state = self.state.write();
        if new_size != state.max_positions {
            state.max_positions = new_size;
            self.evict_locked(&mut state);
        }
        Ok(new_size)
    }
    
    fn has_entity_changed(&self, entity: String, stream_pos: i64) -> bool {
        let state = self.state.read();
        
        // Cache not valid before earliest known position - this is stale data
        if stream_pos < state.earliest_known_stream_pos {
            self.metrics.inc_stale();
            return false; // Can't determine if changed - assume no change
        }
        
        match state.entity_to_key.get(entity.as_str()) {
            None => {
                self.metrics.inc_misses();
                false // Unknown entity hasn't changed
            }
            Some(&pos) => {
                self.metrics.inc_hits(); // Cache provided answer
                stream_pos < pos
            }
        }
    }
    
    fn get_entities_changed(
        &self,
        entities: &Bound<'_, PyAny>,
        stream_pos: i64,
        perf_factor: Option<i64>,
    ) -> PyResult<PyObject> {
        let perf_factor = perf_factor.unwrap_or(1);
        
        // Convert Python collection to Vec<Arc<str>>
        let entity_list: Vec<Arc<str>> = {
            use pyo3::types::PyIterator;
            let iter = PyIterator::from_object(entities)?;
            let mut result = Vec::new();
            for item in iter {
                result.push(Arc::from(item?.extract::<String>()?));
            }
            result
        };
        
        let state = self.state.read();
        if state.cache.is_empty() || stream_pos < state.earliest_known_stream_pos {
            if stream_pos < state.earliest_known_stream_pos {
                self.metrics.inc_stale();
            } else {
                self.metrics.inc_misses();
            }
            let entity_strings: Vec<String> = entity_list.iter().map(|s| s.to_string()).collect();
            return Ok(PySet::new(entities.py(), &entity_strings)?.into());
        }
        
        // Performance optimization: check individual entities vs scan all changes
        // Heuristic: if scanning all changes would examine more positions than entities,
        // check each entity individually instead
        if let Some((&max_stream_pos, _)) = state.cache.last_key_value() {
            if max_stream_pos - stream_pos > perf_factor * entity_list.len() as i64 {
                // Check each entity individually (cache-based answer)
                let changed: Vec<String> = entity_list
                    .into_iter()
                    .filter(|entity| {
                        state.entity_to_key.get(entity).map_or(-1, |&pos| pos) > stream_pos
                    })
                    .map(|entity| entity.to_string())
                    .collect();
                self.metrics.inc_hits(); // Used cache to determine result
                return Ok(PySet::new(entities.py(), &changed)?.into());
            }
        }
        
        // Scan changes and check intersection on-the-fly to reduce allocations
        let mut intersection = Vec::new();
        let mut count = 0;
        
        // Use HashSet for efficient lookup
        let entity_set: HashSet<Arc<str>> = entity_list.into_iter().collect();
        for (_, entities_set) in state.cache.range((stream_pos + 1)..) {
            for entity in entities_set {
                if entity_set.contains(entity) {
                    intersection.push(entity.to_string());
                }
                count += 1;
                if count >= MAX_ENTITIES_RESULT {
                    break;
                }
            }
            if count >= MAX_ENTITIES_RESULT {
                break;
            }
        }
        
        self.metrics.inc_hits(); // Used cache to determine result
        Ok(PySet::new(entities.py(), &intersection)?.into())
    }
    
    fn has_any_entity_changed(&self, stream_pos: i64) -> bool {
        let state = self.state.read();
        
        if stream_pos < state.earliest_known_stream_pos {
            self.metrics.inc_stale();
            return false; // Can't determine if changed - assume no change
        }
        
        if state.cache.is_empty() {
            self.metrics.inc_misses();
            return false;
        }
        
        self.metrics.inc_hits();
        state.cache.last_key_value().map_or(false, |(&max_pos, _)| stream_pos < max_pos)
    }
    
    fn get_all_entities_changed(&self, stream_pos: i64) -> AllEntitiesChangedResult {
        let state = self.state.read();
        
        if stream_pos < state.earliest_known_stream_pos {
            self.metrics.inc_stale();
            return AllEntitiesChangedResult { entities: None, truncated: false };
        }
        
        let mut changed_entities = Vec::new();
        let mut count = 0;
        let mut truncated = false;
        
        // Cap results to prevent unbounded memory allocation
        for (_, entities) in state.cache.range((stream_pos + 1)..) {
            for entity in entities {
                if count >= MAX_ENTITIES_RESULT {
                    truncated = true;
                    break;
                }
                changed_entities.push(entity.clone());
                count += 1;
            }
            if truncated {
                break;
            }
        }
        
        self.metrics.inc_hits(); // Used cache to determine result
        AllEntitiesChangedResult {
            entities: Some(Arc::new(changed_entities)),
            truncated,
        }
    }
    
    fn entity_has_changed(&self, entity: String, stream_pos: i64) {
        let mut state = self.state.write();
        
        // Ignore changes before earliest known position
        if stream_pos <= state.earliest_known_stream_pos {
            return;
        }
        
        let entity_key: Arc<str> = Arc::from(entity);
        
        // Check if entity already exists at this or later position
        if let Some(&old_pos) = state.entity_to_key.get(&entity_key) {
            if old_pos >= stream_pos {
                return; // Nothing to do
            }
            
            // Remove from old position
            if let Some(old_entities) = state.cache.get_mut(&old_pos) {
                old_entities.remove(&entity_key);
                if old_entities.is_empty() {
                    state.cache.remove(&old_pos);
                }
            }
        }
        
        // Add to new position
        state.entity_to_key.insert(entity_key.clone(), stream_pos);
        state.cache.entry(stream_pos).or_insert_with(HashSet::new).insert(entity_key);
        
        self.evict_locked(&mut state);
    }
    
    /// Mark all entities as changed. This clears the entire cache and sets a new earliest position.
    /// WARNING: This operation is destructive and will lose all cached data.
    fn all_entities_changed(&self, stream_pos: i64) {
        let mut state = self.state.write();
        state.cache.clear();
        state.entity_to_key.clear();
        state.earliest_known_stream_pos = stream_pos;
    }
    
    fn get_max_pos_of_last_change(&self, entity: String) -> Option<i64> {
        self.state.read().entity_to_key.get(entity.as_str()).copied()
    }
    
    fn get_earliest_known_position(&self) -> i64 {
        self.state.read().earliest_known_stream_pos
    }
    
    /// Returns the number of entities currently cached (not stream positions)
    fn __len__(&self) -> usize {
        self.state.read().entity_to_key.len()
    }
    
    /// Return (hits, misses, stale) as a tuple.
    fn metrics_counts(&self) -> (u64, u64, u64) {
        (self.metrics.hits(), self.metrics.misses(), self.metrics.stale())
    }
    
    fn get_stale(&self) -> u64 {
        self.metrics.stale()
    }

    fn get_hits(&self) -> u64 {
        self.metrics.hits()
    }

    fn get_misses(&self) -> u64 {
        self.metrics.misses()
    }
    

    
    fn __repr__(&self) -> String {
        let state = self.state.read();
        format!(
            "RustStreamChangeCache(name='{}', server='{}', size={}, max_positions={}, earliest_pos={}, hits={}, misses={}, stale={})",
            self.name, self.server_name, state.cache.len(), state.max_positions, state.earliest_known_stream_pos,
            self.metrics.hits(), self.metrics.misses(), self.metrics.stale()
        )
    }
}

impl RustStreamChangeCache {
    fn evict_locked(&self, state: &mut CacheState) {
        // Evict oldest entries if cache is too large (by positions or total entities)
        let mut last_evicted: Option<i64> = None;
        while state.cache.len() > state.max_positions || state.entity_to_key.len() > MAX_ENTITIES_TOTAL {
            if let Some((old_stream_pos, old_entities)) = state.cache.pop_first() {
                last_evicted = Some(old_stream_pos);
                for entity in old_entities {
                    state.entity_to_key.remove(&entity);
                }
            } else {
                break;
            }
        }
        if let Some(pos) = last_evicted {
            state.earliest_known_stream_pos = pos + 1;
        }
    }
}