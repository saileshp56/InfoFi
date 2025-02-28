use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::*;
use smartcore::readers;
use std::fs::File;
use std::io::{Read, Write};
use polars::prelude::*;
use rmp_serde;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create directories if they don't exist
    std::fs::create_dir_all("res/ml-model")?;
    std::fs::create_dir_all("res/input-data")?;

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

    // Train a model with custom parameters
    let model = DecisionTreeClassifier::fit(
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

    // Make predictions
    let predictions = model.predict(&input)?;
    println!("Model predictions (first 5): {:?}", &predictions[..5]);

    // Serialize and save the model and input data
    let model_bytes = rmp_serde::to_vec(&model)?;
    let data_bytes = rmp_serde::to_vec(&input)?;
    let labels_bytes = rmp_serde::to_vec(&y_u32s)?;

    let mut f = File::create("res/ml-model/tree_model_bytes.bin")?;
    f.write_all(&model_bytes)?;

    let mut f1 = File::create("res/input-data/tree_model_data_bytes.bin")?;
    f1.write_all(&data_bytes)?;

    let mut f2 = File::create("res/input-data/tree_model_labels.bin")?;
    f2.write_all(&labels_bytes)?;

    println!("Model, data, and labels saved successfully!");
    Ok(())
}