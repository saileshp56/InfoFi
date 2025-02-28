use methods::{METHOD_ELF, METHOD_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::*;
use std::fs::File;
use std::io::Read;

#[derive(Serialize, Deserialize)]
struct InferenceInput {
    model_bytes: Vec<u8>,
    test_data: DenseMatrix<f64>,
}

#[derive(Serialize, Deserialize)]
struct InferenceOutput {
    predictions: Vec<u32>,
}

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    // Load the serialized model
    let mut model_file = File::open("../../res/ml-model/tree_model_bytes.bin").expect("Failed to open model file");
    let mut model_bytes = Vec::new();
    model_file.read_to_end(&mut model_bytes).expect("Failed to read model file");

    // Load the test data (using the same data for simplicity)
    let mut data_file = File::open("../../res/input-data/tree_model_data_bytes.bin").expect("Failed to open data file");
    let mut data_bytes = Vec::new();
    data_file.read_to_end(&mut data_bytes).expect("Failed to read data file");
    
    // Deserialize the test data
    let test_data: DenseMatrix<f64> = rmp_serde::from_slice(&data_bytes).expect("Failed to deserialize test data");

    // Create the input for the guest
    let input = InferenceInput {
        model_bytes,
        test_data,
    };

    // Set up the executor environment with our input
    let env = ExecutorEnv::builder()
        .write(&input)
        .unwrap()
        .build()
        .unwrap();

    // Get the default prover
    let prover = default_prover();

    // Prove the execution
    println!("Running the prover...");
    let prove_info = prover.prove(env, METHOD_ELF).unwrap();

    // Extract the receipt
    let receipt = prove_info.receipt;

    // Verify the receipt
    receipt.verify(METHOD_ID).unwrap();
    println!("Receipt verified successfully!");

    // Decode the journal (public output)
    let output: InferenceOutput = receipt.journal.decode().unwrap();
    
    // Print the predictions
    println!("Predictions (first 5): {:?}", &output.predictions[..5.min(output.predictions.len())]);
    println!("Total predictions: {}", output.predictions.len());
}