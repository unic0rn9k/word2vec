use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter},
};

use crate::{wordify, DICTIONARY_SIZE};
use serde::{Deserialize, Serialize};
use serde_yaml as yaml;

const DICTIONARY_FILE: &'static str = "dictionary.yaml";
const STOP_WORDS: &'static [&'static str] = &[
    "a", "and", "be", "with", "can", "will", "an", "of", "the", "in", "it", "is", "i", "on", "for",
    "where", "was", "who", "to", "as", "s", "at", "that", "were", "unk", "this", "by", "had",
    "has", "these", "which", "but", "from", "not", "are", "when", "or", "either", "its", "a", "b",
    "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
    "x", "y", "z", "their", "than", "you", "me", "whose", "my", "if", "become",
];
//const STOP_WORDS: &'static [&'static str] = &[];

#[derive(Deserialize, Serialize)]
pub struct Dictionary {
    pub word_id: HashMap<String, usize>,
    pub id_word: Vec<String>,
}

impl Dictionary {
    pub fn open(text_file: &str) -> Self {
        if std::path::Path::new(DICTIONARY_FILE).exists() {
            println!("Dictionary file already exists, so it will be reused.");
            return yaml::from_reader(BufReader::new(File::open(DICTIONARY_FILE).unwrap()))
                .unwrap();
        }
        println!("Dictionary file not found, so it will be generated.");

        let mut count = HashMap::new();

        let reader = BufReader::new(File::open(text_file).unwrap());

        for line in reader.lines() {
            for word in line.unwrap().split_whitespace() {
                let mut word = word.to_string();
                if wordify(&mut word).is_err() || STOP_WORDS.iter().any(|w| w.to_string() == word) {
                    continue;
                }

                match count.get_mut(&word) {
                    Some(c) => *c += 1,
                    Option::None => {
                        count.insert(word, 1);
                    }
                }
            }
        }

        let mut count_vec: Vec<_> = count.iter().collect();
        count_vec.sort_by(|a, b| b.1.cmp(a.1));

        let mut m = HashMap::new();
        let mut v = vec![];

        for n in 0..DICTIONARY_SIZE {
            v.push(count_vec[n].0.clone());
            m.insert(v[n].clone(), n);
        }

        let tmp = Self {
            word_id: m,
            id_word: v,
        };

        yaml::to_writer(BufWriter::new(File::create(DICTIONARY_FILE).unwrap()), &tmp).unwrap();

        tmp
    }
}
