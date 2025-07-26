#include "bytoken.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <set>
#include <fstream>

#include "nlohmann/json.hpp"

ByToken::ByToken() : vocab_size(0), max_key(0) {

}


/**
 * @brief Trains the tokenizer on the provided text corpus.
 * 
 * This function builds a vocabulary up to the given size by applying
 * Byte Pair Encoding (BPE) merges on the text.
 * 
 * @param text_corpus The input text corpus for training.
 * @param vocab_size Desired vocabulary size after training.
 * @param verbose Enable verbose output for debugging or progress tracking.
 */
void ByToken::train(std::string text_corpus, int vocab_size, bool verbose) {
    this->vocab_size = vocab_size;
    std::set<char> text_corpus_unique(text_corpus.begin(), text_corpus.end());

    // + 1 for <UNK> token
    if (text_corpus_unique.size() + 1 > vocab_size) {
        std::ostringstream oss;
        oss << "vocab_size (" << vocab_size << ") must be greater than or equal to the number of unique chars (" << text_corpus_unique.size() << ") in the text corpus";
        throw std::runtime_error(oss.str());
    }
    
    // adding <UNK> token to vocab
    stoi["<UNK>"] = max_key;
    itos[max_key] = "<UNK>";
    max_key++;

    for (char ch : text_corpus_unique) {
        std::string str(1, ch);
        stoi[str] = max_key;
        itos[max_key] = str;
        max_key++;
    }

    std::vector<int> text_idx;
    for (char ch : text_corpus) {
        std::string str(1, ch);
        text_idx.push_back(stoi[str]);
    }

    std::set<int> milestones{0};
    for (int i = 2; i < 11; i += 2) {
        float frac = 0.1 * i;
        int milestone = static_cast<int>(vocab_size * frac) - 1;
        milestones.insert(milestone);
    }

    while (max_key < vocab_size) {
        if (verbose) {
            if (milestones.find(max_key) != milestones.end()) {
                double progress = static_cast<double>(max_key + 1) / vocab_size;
                int percent_trained = static_cast<int>(std::round(progress * 100));
                std::cout << "Training progress " << std::to_string(percent_trained) << "%\n";
            }
        }

        std::unordered_map<std::pair<int, int>, int, pair_hash> pair_count;
        for (int i = 0; i < text_idx.size()-1; i++) {
            pair_count[std::make_pair(text_idx[i], text_idx[i+1])]++;
        }

        if (pair_count.empty()) break;

        auto max_pair = std::max_element(pair_count.begin(), pair_count.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->first;

        std::vector<int> new_text_idx;

        int i{0};
        while (i < text_idx.size()-1) {
            if (std::pair<int, int>{text_idx[i], text_idx[i+1]} == max_pair) {
                new_text_idx.push_back(max_key);
                i += 2;
            } else {
                new_text_idx.push_back(text_idx[i]);
                i++;
            }
        }

        std::string merged_pair = itos[max_pair.first] + itos[max_pair.second];
        itos[max_key] = merged_pair;
        stoi[merged_pair] = max_key;

        merges[max_pair] = max_key;
        max_key++;
        text_idx = std::move(new_text_idx);
    }

    final_vocab.clear();
    for (const auto& [str, id] : stoi) {
        final_vocab.emplace_back(str, id);
    }
    std::sort(final_vocab.begin(), final_vocab.end(),
        [](const auto& a, const auto& b) {
            return a.first.length() > b.first.length();
        });
    
    if (verbose) {
        std::cout << "Tokenizer successfully trained! Final vocab size: " << stoi.size() << "\n";
    }
}

/**
 * @brief Function to encode a given string.
 * 
 * @param text The text (string) which is to be encoded.
 * 
 * @return A vector of encoded integers.
*/
std::vector<int> ByToken::encode(std::string text) {
    std::vector<int> encoded_idx;
    size_t pos = 0;

    while (pos < text.size()) {
        bool matched = false;

        for (const auto& [substr, subkey] : this->final_vocab) {
            size_t len = substr.length();

            if (pos + len <= text.length() && text.compare(pos, len, substr) == 0) {
                encoded_idx.push_back(subkey);
                pos += len;
                matched = true;
                break;
            }
        }

        if (!matched) {
            encoded_idx.push_back(stoi["<UNK>"]);
            ++pos;
        }
    }
    
    return encoded_idx;
}

/**
 * @brief Function to decode a given vector of encoded indices.
 * 
 * @param idx The vector of encoded indices (encoding) which is to be decoded.
 * 
 * @return The decoded string.
*/
std::string ByToken::decode(std::vector<int> idx)
{
    // add support for unsupported/unknown tokens
    std::string decoded_str;
    for (int i : idx) {
        if (itos.count(i)) {
            decoded_str += itos[i];
        } else {
            decoded_str += "<INVALID_ID>";
        }
    }

    return decoded_str;
}

/**
 * @brief Saves the tokenizer's state to a JSON file.
 * 
 * This serializes the vocabulary, merges, and configuration, allowing the
 * exact state of the trained tokenizer to be reloaded later.
 * 
 * @param path The file path where the tokenizer will be saved.
 */
void ByToken::save(const std::string& path) const {
    nlohmann::json j;

    j["config"]["vocab_size"] = this->vocab_size;
    j["config"]["max_key"] = this->max_key; // for multiple training iterations

    j["model"]["vocab"]["stoi"] = this->stoi;
    j["model"]["vocab"]["itos"] = this->itos;
    j["model"]["vocab"]["final_vocab"] = this->final_vocab;

    nlohmann::json merges_json;
    for (const auto& [pair, merge_id] : this->merges) {
        std::string key = std::to_string(pair.first) + "," + std::to_string(pair.second);
        merges_json[key] = merge_id;
    }
    j["model"]["merges"] = merges_json;

    std::ofstream o(path);
    o << j.dump(2, ' ', true, nlohmann::json::error_handler_t::ignore);

    o.close();
}

/**
 * @brief Loads a tokenizer from a saved JSON file.
 * This is a static factory function that constructs a new ByToken instance
 * by deserializing its state from a file.
 * 
 * @param path The file path of the saved tokenizer.
 * 
 * @return A new ByToken instance with the loaded state.
 */
ByToken ByToken::from_file(const std::string& path) {
    std::ifstream i(path);
    if (!i.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    nlohmann::json j;
    i >> j;

    ByToken tokenizer;

    tokenizer.vocab_size = j["config"]["vocab_size"].get<int>();
    tokenizer.max_key = j["config"]["vocab_size"].get<int>();

    tokenizer.stoi = j["model"]["stoi"].get<std::unordered_map<std::string, int>>();
    tokenizer.itos = j["model"]["itos"].get<std::unordered_map<int, std::string>>();
    tokenizer.final_vocab = j["model"]["final_vocab"].get<std::vector<std::pair<std::string, int>>>();

    const auto& merges_json = j["model"]["merges"];
    for (auto it = merges_json.begin(); it != merges_json.end(); ++it) {
        std::string key_str = it.key();
        size_t comma_pos = key_str.find(',');
        if (comma_pos == std::string::npos) continue;

        int first_id = std::stoi(key_str.substr(0, comma_pos));
        int second_id = std::stoi(key_str.substr(comma_pos + 1));
        int merge_id = it.value().get<int>();

        tokenizer.merges[std::make_pair(first_id, second_id)] = merge_id;
    }
    
    return tokenizer;
}
