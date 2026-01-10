use crate::layers::{Linear, ReLU, Softmax};
use crate::loss_function::{cross_entropy, mse};

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
}

#[derive(Serialize, Deserialize)]
pub struct SimpleNN {
    linear1: Linear,
    linear2: Linear,
    task_type: TaskType,

    // caches
    input_cache: Vec<Vec<f32>>,
    hidden_cache: Vec<Vec<f32>>,
    logits_cache: Vec<Vec<f32>>,
    probs_cache: Vec<Vec<f32>>,
}

impl SimpleNN {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        task_code: usize,
    ) -> Self {
        let task_type = if task_code == 1 {
            TaskType::Regression
        } else {
            TaskType::Classification
        };

        Self {
            linear1: Linear::new(input_dim, hidden_dim),
            linear2: Linear::new(hidden_dim, output_dim),
            task_type,
            input_cache: vec![],
            hidden_cache: vec![],
            logits_cache: vec![],
            probs_cache: vec![],
        }
    }

    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let hidden_pre = self.linear1.forward(x);
        let hidden_post = ReLU::forward(&hidden_pre);
        let logits = self.linear2.forward(&hidden_post);

        let output = match self.task_type {
            TaskType::Classification => {
                logits.iter().map(|l| Softmax::forward(l)).collect()
            }
            TaskType::Regression => logits.clone(),
        };

        self.input_cache = x.clone();
        self.hidden_cache = hidden_post;
        self.logits_cache = logits;
        self.probs_cache = output.clone();

        output
    }

    pub fn train(
        &mut self,
        x: &Vec<Vec<f32>>,
        y: &Vec<Vec<f32>>,
        epochs: usize,
        _lr: f32,
        batch_size: Option<usize>,
    ) -> Vec<f32> {
        // Mandatory batching
        let batch_size = batch_size.unwrap_or(32);
        let mut history = Vec::new();

        for _ in 0..epochs {
            for batch_start in (0..x.len()).step_by(batch_size) {
                let end = (batch_start + batch_size).min(x.len());
                let x_batch = &x[batch_start..end];
                let y_batch = &y[batch_start..end];

                let preds = self.forward(&x_batch.to_vec());
                let y_batch_vec = y_batch.to_vec();

                let loss = match self.task_type {
                    TaskType::Classification => {
                        cross_entropy(&preds, &y_batch_vec)
                    }
                    TaskType::Regression => {
                        mse(&preds, &y_batch_vec)
                    }
                };

                history.push(loss);
                // optimizer step will be added later
            }
        }

        history
    }

    pub fn predict(&mut self, x: &Vec<Vec<f32>>) -> Vec<f32> {
        let output = self.forward(x);

        match self.task_type {
            TaskType::Classification => output
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0)
                })
                .collect(),

            TaskType::Regression => output.iter().map(|row| row[0]).collect(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model: SimpleNN = serde_json::from_str(&contents)?;
        Ok(model)
    }
}
