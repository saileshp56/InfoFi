use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::numbers::realnum::RealNumber;

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
    // Read the serialized model and test data from the host
    let input: InferenceInput = env::read();
    
    // Deserialize the model
    let model: DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> = 
        rmp_serde::from_slice(&input.model_bytes).unwrap();
    
    // Perform inference
    let predictions = model.predict(&input.test_data).unwrap();
    
    // Write the predictions to the journal (public output)
    let output = InferenceOutput { predictions };
    env::commit(&output);
}