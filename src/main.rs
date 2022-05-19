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
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
extern crate blas_src;
use slas_backend::Blas;

use crate::dictionary::Dictionary;

mod dictionary;

//const UNIQUE_WORDS: usize = 27958;
const DICTIONARY_SIZE: usize = 1000;
const MAX_DISTANCE: usize = 4;
const INPUT_FILE: &'static str = "shake.txt";
const PROGRESSBAR_UPDATE_RATE: usize = 100;
const EMBEDDED_SIZE: usize = 120;

model! {(
    derive: [],
    name: "NeuralNet",
    layers: [
        ("DenseHeapLayer::<f32, Blas, DICTIONARY_SIZE, EMBEDDED_SIZE>", "DenseHeapLayer::random(0.04)"),
        ("DenseHeapLayer::<f32, Blas, EMBEDDED_SIZE, DICTIONARY_SIZE>", "DenseHeapLayer::random(0.04)"),
        ("Softmax::<f32, DICTIONARY_SIZE>", "default()")
    ],
    float_type: "f32",
    input_len: 1000,
    output_len: 1000,
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

fn load_net(net: &mut NeuralNet) -> Result<()> {
    let mut f = File::open("l0_parameters")?;
    let mut buffer = vec![];
    f.read_to_end(&mut buffer)?;
    deserialize_dense_layer(&mut net.l0, &mut buffer.iter().copied());
    f = File::open("l2_parameters")?;
    f.read_to_end(&mut buffer)?;
    deserialize_dense_layer(&mut net.l1, &mut buffer.iter().copied());
    Ok(())
}

fn main() -> Result<()> {
    println!("Initializing stuff...");
    let mut rng = thread_rng();

    let mut net = NeuralNet::new();
    println!("{:?}", load_net(&mut net));
    let mut buffer = unsafe { NeuralNet::uninit_cache() };

    let mut word_buffer = vec![];

    let Dictionary { word_id, id_word } = Dictionary::open(INPUT_FILE);

    let reader = BufReader::new(File::open(INPUT_FILE).unwrap());

    let spinner = ProgressBar::new(0);
    spinner.set_style(
        ProgressStyle::default_bar().template(" {spinner} {prefix:.bold.dim}\t{wide_msg}"),
    );

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
                continue;
            }

            if word_buffer.len() < MAX_DISTANCE * 2 {
                word_buffer.push(word.clone());
                continue;
            }
            word_buffer[ring_index(word_count as isize, MAX_DISTANCE * 2)] = word.clone();

            let mut word2 = String::new();
            while !word_id.contains_key(&word2) {
                let mut word_delta = 0;
                while word_delta == 0 {
                    word_delta = rng.gen_range(0..MAX_DISTANCE) as isize;
                }

                if rand::random() {
                    word_delta *= -1;
                }
                word2 = word_buffer[ring_index(
                    word_count as isize + word_delta - MAX_DISTANCE as isize,
                    word_buffer.len(),
                )]
                .clone();
            }

            let i: [f32; DICTIONARY_SIZE] = onehot(
                word_id[&word_buffer[ring_index(
                    word_count as isize - MAX_DISTANCE as isize,
                    word_buffer.len(),
                )]],
            );

            net.predict(i, &mut buffer)?;

            let y: [f32; DICTIONARY_SIZE] = onehot(word_id[&word2]);

            let o = &buffer[buffer.len() - DICTIONARY_SIZE..buffer.len()];
            let dy = moo![|n| o[n] - y[n]; DICTIONARY_SIZE];

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

            if epoch % PROGRESSBAR_UPDATE_RATE == 0 && epoch != 0 {
                let beautiful = {
                    let id = word_id["beauty"];
                    let mut beauty_vec = [0.; EMBEDDED_SIZE];

                    net.l0.predict(onehot(id), &mut beauty_vec)?;

                    let mut min_dist = 100000.;
                    let mut min_word = 0;

                    for i in 0..DICTIONARY_SIZE {
                        spinner.inc(1);

                        let buffer = net
                            .l0
                            .weights
                            .moo_ref::<{ EMBEDDED_SIZE * DICTIONARY_SIZE }>()
                            .matrix_ref::<Blas, EMBEDDED_SIZE, DICTIONARY_SIZE>();

                        let mut dist = 0.;
                        let mut argmax = 0;

                        for j in 0..EMBEDDED_SIZE {
                            dist += (buffer[(j, i)] - beauty_vec[j]).powi(2);
                            if buffer[(j, i)] > buffer[(argmax, i)] {
                                argmax = j
                            }
                        }
                        dist = dist.sqrt();

                        if dist < min_dist {
                            min_dist = dist;
                            min_word = argmax;
                        }
                    }

                    id_word[min_word].clone()
                };

                spinner.set_message(format!(
                    "cost: {:.6}, lr: {:.5}, input: {}, predicted: {} ({}), beauty: {}",
                    cost_sum / PROGRESSBAR_UPDATE_RATE as f32,
                    net.l0.lr,
                    word,
                    id_word[argmax(o)],
                    o[argmax(o)],
                    beautiful
                ));
                cost_sum = 0.;
                spinner.set_prefix("Training");
            }

            if epoch % (PROGRESSBAR_UPDATE_RATE * 10) == 0 && epoch != 0 {
                spinner.set_prefix("Saving model");
                let mut f = File::create("./l0_parameters")?;
                let mut buffer: Vec<_> = serialize_dense_layer(&net.l0).collect();

                f.write_all(&buffer[..])?;
                f = File::create("./l2_parameters")?;

                buffer = serialize_dense_layer(&net.l1).collect();
                f.write_all(&buffer[..])?;
            }

            net.l0.lr *= 0.99999;
            net.l1.lr *= 0.99999;
            net.backpropagate(i, &buffer, dy)?;

            net.l0.biasies = vec![0.; EMBEDDED_SIZE];

            epoch += 1;
            word_count += 1;
            if word_count >= word_buffer.len() {
                word_count = 0
            }
        }
    }

    Ok(())
}
