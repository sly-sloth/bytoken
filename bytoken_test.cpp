#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono> // For timing
#include <cassert> // For testing the loaded tokenizer

#include "bytoken.h" // Your tokenizer class header

int main() {
    // --- 1. Load the training data ---
    std::string train_corpus;
    std::ifstream file("data/sherlock.txt"); // Assuming your data is in data/book.txt

    if (!file.is_open()) {
        std::cerr << "Error: Could not open data/book.txt. Please create this file." << std::endl;
        return 1;
    }
    train_corpus.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    std::cout << "Corpus size: " << train_corpus.length() << " characters" << std::endl;

    // --- 2. Train the tokenizer ---
    ByToken tokenizer;
    auto start_train = std::chrono::high_resolution_clock::now();
    tokenizer.train(train_corpus, 256, true);
    auto end_train = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = end_train - start_train;
    std::cout << "Training completed in " << train_duration.count() << "s" << std::endl;

    // --- 3. Benchmark encode/decode ---
    std::string sample = "But on the edge of town, drills were driven out of his mind by something else.";
    std::cout << "\nStarting benchmark..." << std::endl;
    auto start_bench = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50000; ++i) {
        std::vector<int> encoded = tokenizer.encode(sample);
        std::string decoded = tokenizer.decode(encoded);
    }
    auto end_bench = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bench_duration = end_bench - start_bench;
    std::cout << "Benchmark completed." << std::endl;
    std::cout << "Total time taken: " << bench_duration.count() << "s" << std::endl;

    // --- 4. Save the tokenizer ---
    std::cout << "\nSaving tokenizer to enc_trial.json..." << std::endl;
    tokenizer.save("enc_trial.json");
    std::cout << "Saved successfully." << std::endl;

    // --- 5. Load the tokenizer from the file and test it ---
    std::cout << "\nLoading tokenizer from enc_trial.json..." << std::endl;
    ByToken loaded_tokenizer = ByToken::from_file("enc_trial.json");
    std::cout << "Loaded successfully." << std::endl;

    // Verify that the loaded tokenizer gives the same result as the original
    std::string test_string = "hello from the loaded tokenizer";
    std::vector<int> original_encoding = tokenizer.encode(test_string);
    std::vector<int> loaded_encoding = loaded_tokenizer.encode(test_string);

    assert(original_encoding == loaded_encoding);
    std::cout << "Verification successful: Original and loaded tokenizers produce the same output." << std::endl;
    
    return 0;
}