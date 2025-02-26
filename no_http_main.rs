use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::*;
use smartcore::readers;
use std::fs::File;
use std::io::{Read, Write};
use polars::prelude::*;
use serde_json;
use rmp_serde;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the iris dataset from CSV files
    let input = readers::csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
        File::open("src/iris_input_data.csv").unwrap(),
        readers::csv::CSVDefinition::default()
    ).unwrap();

    let filepath_iris_classes = "src/iris_classes.csv";

    let y_u32s: Vec<u32> = CsvReader::new(File::open(filepath_iris_classes)?).finish()?
                .column("variety")?.clone()
                .i64()?.into_no_null_iter().collect::<Vec<i64>>()
                .into_iter().map(|x| x as u32).collect::<Vec<u32>>();

    // Train another model with custom parameters
    let model_with_custom_params = DecisionTreeClassifier::fit(
        &input,
        &y_u32s,
        DecisionTreeClassifierParameters {
            criterion: SplitCriterion::Entropy,
            max_depth: Some(3),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: Option::None
        }
    )?;

    // Make predictions with both models
    let custom_predictions = model_with_custom_params.predict(&input)?;
    

    println!("Custom model predictions (first 5): {:?}", &custom_predictions[..5]);

    // Create directories if they don't exist
    std::fs::create_dir_all("res/ml-model")?;
    std::fs::create_dir_all("res/input-data")?;

    // Serialize and save the model and input data
    let model_bytes = rmp_serde::to_vec(&model_with_custom_params)?;
    let data_bytes = rmp_serde::to_vec(&input)?;

    let mut f = File::create("res/ml-model/tree_model_bytes.bin")?;
    f.write_all(&model_bytes)?;

    let mut f1 = File::create("res/input-data/tree_model_data_bytes.bin")?;
    f1.write_all(&data_bytes)?;

    println!("Model and data saved successfully!");

    Ok(())
}