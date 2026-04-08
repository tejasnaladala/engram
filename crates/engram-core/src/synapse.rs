use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::types::Weight;

/// Compressed Sparse Row (CSR) synapse matrix.
///
/// Stores directed connections from pre-synaptic to post-synaptic neurons.
/// Memory layout is optimized for the primary access pattern: "given a spiking
/// pre-synaptic neuron, find all its post-synaptic targets and weights."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseMatrix {
    /// Number of pre-synaptic neurons (rows)
    pub pre_count: u32,
    /// Number of post-synaptic neurons (columns)
    pub post_count: u32,
    /// Row pointers: row_ptr[i] is the start index in col_idx/values for row i.
    /// row_ptr has length pre_count + 1. row_ptr[pre_count] == col_idx.len().
    pub row_ptr: Vec<u32>,
    /// Column indices (post-synaptic neuron IDs) for non-zero entries
    pub col_idx: Vec<u32>,
    /// Synaptic weights for non-zero entries
    pub values: Vec<Weight>,
    /// Axonal delay in timesteps for each synapse
    pub delays: Vec<u8>,
}

/// A single outgoing connection
#[derive(Debug, Clone, Copy)]
pub struct Synapse {
    pub post_id: u32,
    pub weight: Weight,
    pub delay: u8,
}

impl SynapseMatrix {
    /// Create an empty synapse matrix
    pub fn empty(pre_count: u32, post_count: u32) -> Self {
        Self {
            pre_count,
            post_count,
            row_ptr: vec![0; (pre_count + 1) as usize],
            col_idx: Vec::new(),
            values: Vec::new(),
            delays: Vec::new(),
        }
    }

    /// Create a random sparse connectivity pattern.
    ///
    /// Each possible (pre, post) connection exists with probability `density`.
    /// Weights are initialized uniformly in [0, w_max].
    /// Self-connections (pre == post when pre_count == post_count) are excluded.
    pub fn random_sparse<R: Rng>(
        pre_count: u32,
        post_count: u32,
        density: f64,
        w_max: Weight,
        rng: &mut R,
    ) -> Self {
        let mut row_ptr = Vec::with_capacity((pre_count + 1) as usize);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        let mut delays = Vec::new();

        row_ptr.push(0);

        for pre in 0..pre_count {
            for post in 0..post_count {
                // Skip self-connections
                if pre_count == post_count && pre == post {
                    continue;
                }
                if rng.random::<f64>() < density {
                    col_idx.push(post);
                    values.push(rng.random::<f32>() * w_max);
                    delays.push(1); // minimal delay
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }

        Self {
            pre_count,
            post_count,
            row_ptr,
            col_idx,
            values,
            delays,
        }
    }

    /// Get all outgoing synapses from a pre-synaptic neuron
    pub fn outgoing(&self, pre_id: u32) -> impl Iterator<Item = Synapse> + '_ {
        let start = self.row_ptr[pre_id as usize] as usize;
        let end = self.row_ptr[(pre_id + 1) as usize] as usize;
        (start..end).map(move |i| Synapse {
            post_id: self.col_idx[i],
            weight: self.values[i],
            delay: self.delays[i],
        })
    }

    /// Get the number of outgoing connections from a pre-synaptic neuron
    pub fn out_degree(&self, pre_id: u32) -> u32 {
        let start = self.row_ptr[pre_id as usize];
        let end = self.row_ptr[(pre_id + 1) as usize];
        end - start
    }

    /// Propagate spikes: for each spiking pre-neuron, accumulate weighted
    /// input to post-synaptic neurons. Returns (post_id, total_current) pairs.
    pub fn propagate(&self, spiking_pre: &[u32]) -> Vec<(u32, f64)> {
        // Use a dense accumulator for the post-synaptic side
        let mut post_currents = vec![0.0f64; self.post_count as usize];
        let mut any_input = false;

        for &pre_id in spiking_pre {
            for syn in self.outgoing(pre_id) {
                post_currents[syn.post_id as usize] += syn.weight as f64;
                any_input = true;
            }
        }

        if !any_input {
            return Vec::new();
        }

        // Collect non-zero entries
        post_currents
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 1e-9)
            .map(|(i, &c)| (i as u32, c))
            .collect()
    }

    /// Update the weight of a specific synapse. Returns true if found.
    pub fn update_weight(&mut self, pre_id: u32, post_id: u32, delta: Weight) -> bool {
        let start = self.row_ptr[pre_id as usize] as usize;
        let end = self.row_ptr[(pre_id + 1) as usize] as usize;

        // Binary search within the row for the post_id
        if let Ok(offset) = self.col_idx[start..end].binary_search(&post_id) {
            self.values[start + offset] += delta;
            return true;
        }
        // Linear scan fallback (col_idx may not be sorted after construction)
        for i in start..end {
            if self.col_idx[i] == post_id {
                self.values[i] += delta;
                return true;
            }
        }
        false
    }

    /// Clamp all weights to [min, max]
    pub fn clamp_weights(&mut self, min: Weight, max: Weight) {
        for w in &mut self.values {
            *w = w.clamp(min, max);
        }
    }

    /// Total number of synapses (non-zero entries)
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sort each row's columns for binary search
    pub fn sort_rows(&mut self) {
        for pre in 0..self.pre_count {
            let start = self.row_ptr[pre as usize] as usize;
            let end = self.row_ptr[(pre + 1) as usize] as usize;
            if start == end {
                continue;
            }
            // Create index permutation
            let mut indices: Vec<usize> = (start..end).collect();
            indices.sort_by_key(|&i| self.col_idx[i]);
            // Apply permutation
            let col_sorted: Vec<u32> = indices.iter().map(|&i| self.col_idx[i]).collect();
            let val_sorted: Vec<Weight> = indices.iter().map(|&i| self.values[i]).collect();
            let del_sorted: Vec<u8> = indices.iter().map(|&i| self.delays[i]).collect();
            self.col_idx[start..end].copy_from_slice(&col_sorted);
            self.values[start..end].copy_from_slice(&val_sorted);
            self.delays[start..end].copy_from_slice(&del_sorted);
        }
    }

    /// Reset all weights to zero
    pub fn reset_weights(&mut self) {
        for w in &mut self.values {
            *w = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random_sparse_creates_connections() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mat = SynapseMatrix::random_sparse(10, 10, 0.5, 1.0, &mut rng);
        assert!(mat.nnz() > 0);
        assert!(mat.nnz() < 100); // at most 90 with no self-connections
    }

    #[test]
    fn propagate_accumulates_correctly() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mat = SynapseMatrix::random_sparse(5, 5, 1.0, 1.0, &mut rng);
        let currents = mat.propagate(&[0]);
        assert!(!currents.is_empty());
    }

    #[test]
    fn update_weight_modifies_synapse() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut mat = SynapseMatrix::random_sparse(5, 5, 1.0, 0.5, &mut rng);
        mat.sort_rows();
        let first_post = mat.col_idx[mat.row_ptr[0] as usize];
        let old_weight = mat.values[mat.row_ptr[0] as usize];
        assert!(mat.update_weight(0, first_post, 0.1));
        let new_weight = mat.values[mat.row_ptr[0] as usize];
        assert!((new_weight - old_weight - 0.1).abs() < 1e-6);
    }

    #[test]
    fn clamp_weights_works() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut mat = SynapseMatrix::random_sparse(5, 5, 1.0, 2.0, &mut rng);
        mat.clamp_weights(0.0, 1.0);
        assert!(mat.values.iter().all(|&w| w >= 0.0 && w <= 1.0));
    }
}
