//! # Word2Vec written in rust
//!
//! [Link to docs](https://docs.google.com/document/d/1SSQzQ1NM7aYmGaPahpi5g9JdX8bc_reYN9-eSZybi7w/edit#)
//!
//! # Issues
//! - Needs cross entropy
//!
//! # Sources
//! - https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0
//! - https://arxiv.org/abs/1301.3781
//! - https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt
//!
//! # Tooling
//! - https://github.com/unic0rn9k/slas
//! - https://github.com/unic0rn9k/exotic

#![feature(round_char_boundary)]

use exotic::rand::Rng;
use exotic::{anyhow::*, prelude::*, rand::thread_rng};
use exotic_macro::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
extern crate blas_src;
use slas_backend::Blas;

use crate::dictionary::Dictionary;

mod dictionary;

//const UNIQUE_WORDS: usize = 27958;
const DICTIONARY_SIZE: usize = 2500;
const MAX_DISTANCE: usize = 4;
const INPUT_FILE: &'static str = "wikitext-103/train.csv";
const PROGRESSBAR_UPDATE_RATE: usize = 100;
const EMBEDDED_SIZE: usize = 200;

model! {(
    derive: [],
    name: "NeuralNet",
    layers: [
        ("DenseLayer::<f32, Blas, DICTIONARY_SIZE, EMBEDDED_SIZE>", "DenseLayer::random(0.02)"),
        ("DenseLayer::<f32, Blas, EMBEDDED_SIZE, DICTIONARY_SIZE>", "DenseLayer::random(0.02)"),
        ("Softmax::<f32, DICTIONARY_SIZE>", "default()")
    ],
    float_type: "f32",
    input_len: 2500,
    output_len: 2500,
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

// n = argmax(y)
// dy = -1/log(o[n])
//fn back_prop(x: &[f32; DICTIONARY_SIZE], net: &mut NeuralNet) {
//    let scores = net
//        .weights
//        .matrix_ref::<Blas, EMBEDDED_SIZE, DICTIONARY_SIZE>()
//        .vector_mul(x);
//}

fn normal_float(f: f32) -> f32 {
    if !f.is_finite() {
        0.
    } else {
        f
    } //.abs().min(50.) * f.signum()
}

fn main() -> Result<()> {
    println!("Initializing stuff...");
    let mut rng = thread_rng();

    let mut net = NeuralNet::new();
    println!("Load existing model? {:?}", load_net(&mut net));
    let mut buffer = unsafe { NeuralNet::uninit_cache() };

    let mut word_buffer = vec![];

    let Dictionary { word_id, id_word } = Dictionary::open(INPUT_FILE);

    loop {
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
                word_buffer[word_count] = word.clone();

                word = word_buffer[ring_index(
                    word_count as isize - MAX_DISTANCE as isize,
                    MAX_DISTANCE * 2,
                )]
                .clone();

                //println!("{word_buffer:?}");

                word_count += 1;
                if word_count >= word_buffer.len() {
                    word_count = 0
                }

                let mut word2 = word_count;
                while word2 == word_count {
                    word2 = rng.gen_range(0..MAX_DISTANCE * 2)
                }

                let i: [f32; DICTIONARY_SIZE] = onehot(word_id[&word]);

                net.predict(i, &mut buffer)?;

                let o = &buffer[buffer.len() - DICTIONARY_SIZE..buffer.len()];

                //let dy = moo![|n| o[n] - if n == word2{1.}else{0.}; DICTIONARY_SIZE];
                //let dy = moo![|n|normal_float(
                //    if n == word2{
                //        -1./o[n]
                //    }else{
                //        o[n]
                //    }
                //); DICTIONARY_SIZE];
                let mut dy = *(o.moo_ref());
                dy[word2] = -1. / o[word2];

                let cost = o
                    .iter()
                    .enumerate()
                    .map(|(n, o)| (o - if n == word2 { 1. } else { 0. }).powi_(2))
                    .sum::<f32>();

                //let cost = -o[word_id[&word2]].ln();

                //let dy = -1. / o[word_id[&word2]].max(0.08).min(20.);

                //if !dy.is_finite() {
                //    println!(
                //        "Cost had invalid value: {cost}, delta: {dy}, o: {}",
                //        o[word_id[&word2]]
                //    );
                //    //println!("{word_buffer:?}");
                //    continue;
                //}

                //let dy = moo![dy; DICTIONARY_SIZE];

                cost_sum += cost;

                net.l0.lr *= 0.9999;
                net.l1.lr *= 0.9999;
                net.backpropagate(i, &buffer, dy)?;
                net.l0.biasies = [0.; EMBEDDED_SIZE];
                {
                    let mut weights = net
                        .l0
                        .weights
                        .mut_moo_ref()
                        .reshape_mut_ref(m![DICTIONARY_SIZE, EMBEDDED_SIZE], Blas);
                    //.matrix_mut_ref::<Blas, DICTIONARY_SIZE, EMBEDDED_SIZE>();
                    for i in 0..EMBEDDED_SIZE {
                        weights.index_slice_mut(i).data.normalize();
                        //let w_slice = &mut weights.index_slice_mut(i);
                        //let norm = w_slice.data.norm();
                        //if norm > 1.{
                        //    w_slice.data.iter_mut().for_each(|n|*n/=norm);
                        //}

                        //let mut w_norm = 0.;
                        //for j in 0..DICTIONARY_SIZE{
                        //    w_norm += weights[(j, i)].powi_(2);
                        //}
                        //w_norm = w_norm.sqrt_();
                        //for j in 0..DICTIONARY_SIZE{
                        //    weights[(j, i)] /= w_norm;
                        //}
                    }
                }

                if epoch % PROGRESSBAR_UPDATE_RATE == 0 {
                    let beautiful = {
                        let id = word_id[&word];
                        let mut beauty_vec = [0.; EMBEDDED_SIZE];

                        net.l0.predict(onehot(id), &mut beauty_vec)?;
                        let mut word_dists = [(10000., 0); DICTIONARY_SIZE];

                        for i in 0..DICTIONARY_SIZE {
                            spinner.inc(1);

                            let buffer = net
                                .l0
                                .weights
                                .moo_ref()
                                .matrix_ref::<Blas, DICTIONARY_SIZE, EMBEDDED_SIZE>();

                            let mut dist = 0.;
                            let mut argmax = 0;

                            for j in 0..EMBEDDED_SIZE {
                                dist += (buffer[(i, j)] - beauty_vec[j]).powi(2);
                                if buffer[(i, j)] > buffer[(i, argmax)] {
                                    argmax = j
                                }
                            }
                            dist = dist.sqrt();

                            word_dists[i] = (dist, i);
                        }

                        word_dists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                        word_dists
                    };

                    //println!("{word_buffer:?}");
                    spinner.set_message(format!(
                        "cost: {:.6}, lr: {:.5}, input: {}, predicted: {} ({}), synonym: {:?}",
                        cost_sum / PROGRESSBAR_UPDATE_RATE as f32,
                        net.l0.lr,
                        word,
                        id_word[argmax(o)],
                        o[argmax(o)],
                        beautiful[0..4]
                            .iter()
                            .map(|(_, id)| id_word[*id].clone())
                            .collect::<Vec<_>>()
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

                epoch += 1;
            }
        }
    }

    //Ok(())
}
