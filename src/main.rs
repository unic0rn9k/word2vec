//! # Word2Vec written in rust
//!
//! # Issues
//! - Only words preceding current word, are used in training
//!
//! # Relevant stuff
//! - https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0
//! - https://arxiv.org/abs/1301.3781
//!
//! # Tooling
//! - https://github.com/unic0rn9k/slas
//! - https://github.com/unic0rn9k/exotic

#![feature(round_char_boundary)]
use exotic::rand::Rng;
use exotic::{
    anyhow::*,
    prelude::*,
    rand::{self, thread_rng},
};
use exotic_macro::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand_distr::StandardNormal;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
extern crate blas_src;
use slas_backend::Blas;

const UNIQUE_WORDS: usize = 27958;
const MAX_DISTANCE: usize = 5;

model! {(
derive: [],
name: "NeuralNet",
layers: [
    ("DenseHeapLayer::<f32, Blas, UNIQUE_WORDS, 100>", "DenseHeapLayer::random(0.01)"),
    ("Relu::<f32, 100>", "default()"),
    ("DenseHeapLayer::<f32, Blas, 100, UNIQUE_WORDS>", "DenseHeapLayer::random(0.01)"),
    ("Softmax::<f32, UNIQUE_WORDS>", "default()")
],
float_type: "f32",
input_len: 27958,
output_len: 27958,
)}

fn wordify(str: &mut String) -> Result<()> {
    let mut tmp = "".to_string();
    *str = str.to_lowercase();

    for c in str.bytes() {
        if c < 97 || c > 122 {
            continue;
        }
        tmp += &format!("{}", c as char)
    }
    if tmp.is_empty() {
        bail!("String {:?} doesn't contain a word", str)
    }
    *str = tmp;
    Ok(())
}

fn ring_index(mut idx: isize, len: usize) -> usize {
    idx = idx % len as isize;
    if idx < 0 {
        (len as isize + idx) as usize
    } else {
        idx as usize
    }
}

fn main() -> Result<()> {
    println!("Initializing stuff...");
    let mut rng = thread_rng();

    let mut net = NeuralNet::new();
    let mut buffer = unsafe { NeuralNet::uninit_cache() };

    let mut word_buffer = vec![];
    let mut word_id = HashMap::new();
    let mut id_word = vec![];

    let reader = BufReader::new(File::open("shake.txt").unwrap());

    let spinner = ProgressBar::new(0);
    spinner.set_style(
        ProgressStyle::default_bar().template("{prefix:.bold.dim} {spinner} {wide_msg}"),
    );
    spinner.set_prefix("Training");

    let mut word_count = 0;
    let mut epoch = 0;
    let mut cost_sum = 0.;

    for line in reader.lines() {
        for word in line.unwrap().split_whitespace() {
            let mut word = word.to_string();
            match wordify(&mut word) {
                Err(_) => continue,
                _ => {}
            };

            spinner.inc(1);

            if !word_id.contains_key(&word) {
                word_id.insert(word.clone(), word_id.len());
                id_word.push(word.clone());
            }

            if word_buffer.len() < MAX_DISTANCE * 2 {
                word_buffer.push(word.clone());
                continue;
            }
            word_buffer[ring_index(word_count as isize, MAX_DISTANCE * 2)] = word.clone();

            let mut word_delta = 0;
            while word_delta == 0 {
                word_delta = (rng.sample::<f32, _>(StandardNormal) * MAX_DISTANCE as f32) as isize;
            }

            if rand::random() {
                word_delta *= -1;
            }
            let word2 = word_buffer
                [ring_index(word_count as isize + word_delta, word_buffer.len())]
            .clone();

            let i: [f32; UNIQUE_WORDS] = onehot(word_id[&word]);

            net.predict(i, &mut buffer)?;

            let y: [f32; UNIQUE_WORDS] = if let Some(id) = word_id.get(&word2) {
                onehot(*id)
            } else {
                let id = word_id.len();
                word_id.insert(word2.clone(), id);
                id_word.push(word2.clone());
                onehot(id)
            };

            let o = &buffer[buffer.len() - UNIQUE_WORDS..buffer.len()];
            let dy = moo![|n| o[n] - y[n]; UNIQUE_WORDS];

            let cost = o
                .iter()
                .zip(y.iter())
                .map(|(o, y)| (o - y).powi_(2))
                .sum::<f32>()
                .abs();

            if cost.is_nan() || cost.is_infinite_() {
                println!("Cost had invalid value");
                continue;
            }

            cost_sum += cost;

            if epoch % 50 == 0 {
                net.l0.lr *= 0.999;
                net.l2.lr *= 0.999;

                spinner.set_message(format!(
                    "cost: {:.3}, lr: {:.5}, input: {:?}, predicted: {:?} ({:?})",
                    cost_sum / epoch as f32,
                    net.l0.lr,
                    word,
                    id_word.get(argmax(o)),
                    o[argmax(o)],
                ));
                cost_sum = 0.;
            }

            net.backpropagate(i, &buffer, dy)?;

            epoch += 1;
            word_count += 1;
            if word_count >= word_buffer.len() {
                word_count = 0
            }
        }
    }

    Ok(())
}
