use methods::{METHOD_ELF, METHOD_ID};
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::*;
use std::fs::{self, File};
use std::io::{Read, Write};
use hex;

#[derive(Serialize, Deserialize)]
struct InferenceInput {
    model_bytes: Vec<u8>,
    test_data: DenseMatrix<f64>,
    expected_accuracy: f64,
}

#[derive(Serialize, Deserialize)]
struct InferenceOutput {
    predictions: Vec<u32>,
    actual_accuracy: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ProofOutput {
    pub proof: String,
    pub pub_inputs: String,
    pub image_id: String,
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

    // Try to load the ground truth labels if available
    let true_labels = match File::open("../../res/input-data/tree_model_labels.bin") {
        Ok(mut labels_file) => {
            let mut labels_bytes = Vec::new();
            labels_file.read_to_end(&mut labels_bytes).expect("Failed to read labels file");
            Some(rmp_serde::from_slice::<Vec<u32>>(&labels_bytes).expect("Failed to deserialize labels"))
        },
        Err(_) => {
            println!("Warning: Labels file not found. Skipping accuracy calculation.");
            None
        }
    };

    // Read the expected accuracy from a file
    let expected_accuracy = match fs::read_to_string("../../res/input-data/expected_accuracy.txt") {
        Ok(content) => content.trim().parse::<f64>().unwrap_or(0.5),
        Err(_) => {
            println!("Warning: Expected accuracy file not found. Using default of 50%.");
            0.5
        }
    };
    println!("Expected accuracy: {:.2}%", expected_accuracy * 100.0);

    // Create the input for the guest
    let input = InferenceInput {
        model_bytes,
        test_data,
        expected_accuracy,
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

    // Create zkVerify compatible proof output
    let mut bin_receipt = Vec::new();
    ciborium::into_writer(&receipt, &mut bin_receipt).unwrap();
    let proof = hex::encode(&bin_receipt);

    fs::write("proof.txt", hex::encode(&bin_receipt)).unwrap();
    let receipt_journal_bytes_array = &receipt.journal.bytes.as_slice();
    let pub_inputs = hex::encode(&receipt_journal_bytes_array);
    
    let image_id_hex = hex::encode(
        METHOD_ID
            .into_iter()
            .flat_map(|v| v.to_le_bytes().into_iter())
            .collect::<Vec<_>>(),
    );
    
    let proof_output = ProofOutput {
        proof: "0x".to_owned() + &proof,
        pub_inputs: "0x".to_owned() + &pub_inputs,
        image_id: "0x".to_owned() + &image_id_hex,
    };

    let proof_output_json = serde_json::to_string_pretty(&proof_output).unwrap();
    fs::write("proof.json", proof_output_json).unwrap();
    println!("zkVerify compatible proof saved to proof.json");

    // Save the original receipt to a file (keeping this for compatibility)
    save_receipt_to_file(&receipt, "ml_inference_receipt.json").expect("Failed to save receipt");
    // useless ^^^
    println!("Original receipt saved to ml_inference_receipt.json");

    // Decode the journal (public output)
    let output: InferenceOutput = receipt.journal.decode().unwrap();
    
    // Print the predictions
    println!("Predictions (first 5): {:?}", &output.predictions[..5.min(output.predictions.len())]);
    println!("Total predictions: {}", output.predictions.len());
    println!("Actual accuracy: {:.2}%", output.actual_accuracy * 100.0);

    // Calculate accuracy if labels are available
    if let Some(labels) = true_labels {
        let mut correct = 0;
        for (pred, true_label) in output.predictions.iter().zip(labels.iter()) {
            if pred == true_label {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / output.predictions.len() as f64;
        println!("Host-calculated accuracy: {:.2}%", accuracy * 100.0);
        
        // Verify accuracy meets the expected threshold
        if accuracy < expected_accuracy {
            println!("ERROR: Model accuracy {:.2}% is below the expected threshold of {:.2}%", 
                     accuracy * 100.0, expected_accuracy * 100.0);
            std::process::exit(1);
        }
        
        println!("Accuracy verification passed: >= {:.2}%", expected_accuracy * 100.0);
    }
}

// Function to save the receipt to a file
fn save_receipt_to_file(receipt: &Receipt, filename: &str) -> std::io::Result<()> {
    // Serialize the receipt to JSON
    let receipt_json = serde_json::to_string_pretty(receipt)
        .expect("Failed to serialize receipt to JSON");
    
    // Write to file
    let mut file = File::create(filename)?;
    file.write_all(receipt_json.as_bytes())?;
    
    Ok(())
}