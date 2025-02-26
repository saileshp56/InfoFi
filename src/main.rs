use actix_web::{web, App, HttpServer, HttpResponse};
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::*;
use smartcore::ensemble::random_forest_classifier::*;
use smartcore::linear::linear_regression::*;
use smartcore::readers;
use std::fs::File;
use std::io::{Read, Write};
use polars::prelude::*;
use serde_json;
use rmp_serde;
use std::fs;
use actix_multipart::Multipart;
use futures_util::TryStreamExt as _;
use actix_cors::Cors;
use actix_web::http::header;

#[derive(Deserialize)]
struct MLParameters {
    max_depth: Option<u16>,
    min_samples_leaf: u16,
    min_samples_split: u16,
    use_entropy: bool,
}

#[derive(Deserialize)]
struct RFParameters {
    max_depth: Option<u16>,
    min_samples_leaf: u16,
    min_samples_split: u16,
    n_trees: u16,
}

#[derive(Serialize, Deserialize, Default)]
struct Dataset {
    title: String,
    description: String,
    size: String,
    format: String,
    categories: Vec<String>,
    price: String,
}

#[derive(Serialize, Deserialize)]
struct Datasets {
    datasets: Vec<Dataset>,
}

async fn train_decision_tree(params: web::Json<MLParameters>) -> HttpResponse {
    let model_params = DecisionTreeClassifierParameters {
        criterion: if params.use_entropy { 
            SplitCriterion::Entropy 
        } else { 
            SplitCriterion::Gini 
        },
        max_depth: params.max_depth,
        min_samples_leaf: params.min_samples_leaf as usize,
        min_samples_split: params.min_samples_split as usize,
        seed: Option::None
    };

    match train_dt_model(model_params) {
        Ok(predictions) => HttpResponse::Ok().json(predictions),
        Err(_) => HttpResponse::InternalServerError().finish()
    }
}

async fn train_random_forest(params: web::Json<RFParameters>) -> HttpResponse {
    let model_params = RandomForestClassifierParameters {
        max_depth: params.max_depth,
        min_samples_leaf: params.min_samples_leaf as usize,
        min_samples_split: params.min_samples_split as usize,
        n_trees: params.n_trees,
        criterion: SplitCriterion::Gini,
        m: Some(4),
        keep_samples: false,
        seed: 42,
    };

    match train_rf_model(model_params) {
        Ok(predictions) => HttpResponse::Ok().json(predictions),
        Err(_) => HttpResponse::InternalServerError().finish()
    }
}

async fn train_linear_regression() -> HttpResponse {
    match train_lr_model() {
        Ok(predictions) => HttpResponse::Ok().json(predictions),
        Err(_) => HttpResponse::InternalServerError().finish()
    }
}

async fn get_datasets() -> HttpResponse {
    match fs::read_to_string("src/datasets.json") {
        Ok(json_str) => {
            match serde_json::from_str::<Datasets>(&json_str) {
                Ok(datasets) => HttpResponse::Ok().json(datasets),
                Err(_) => HttpResponse::InternalServerError().finish()
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish()
    }
}
async fn add_dataset(mut payload: Multipart) -> HttpResponse {
    // Create data directory if it doesn't exist
    std::fs::create_dir_all("src").unwrap_or_default();
    
    println!("Adding dataset");
    let mut dataset = Dataset::default();
    let mut csv_data: Option<Vec<u8>> = None;
    
    // First pass: extract all fields
    while let Ok(Some(mut field)) = payload.try_next().await {
        let content_disposition = field.content_disposition();
        let field_name = content_disposition.get_name().unwrap_or("");
        println!("Processing field: {}", field_name);
        
        match field_name {
            "file" => {
                println!("Processing file upload");
                let mut data = Vec::new();
                while let Ok(Some(chunk)) = field.try_next().await {
                    data.extend_from_slice(&chunk);
                }
                csv_data = Some(data);
            },
            "metadata" => {
                println!("Processing metadata");
                let mut metadata = String::new();
                while let Ok(Some(chunk)) = field.try_next().await {
                    metadata.extend(std::str::from_utf8(&chunk).unwrap().chars());
                }
                println!("Received metadata: {}", metadata);
                if let Ok(parsed_dataset) = serde_json::from_str(&metadata) {
                    dataset = parsed_dataset;
                    println!("Parsed dataset title: {}", dataset.title);
                } else {
                    println!("Failed to parse metadata");
                }
            },
            _ => {
                println!("Unknown field: {}", field_name);
            }
        }
    }
    
    // Now save the file after we have the dataset title
    if let Some(data) = csv_data {
        if dataset.title.is_empty() {
            return HttpResponse::BadRequest().body("Dataset title is required");
        }
        
        if let Ok(_) = std::fs::write(
            format!("src/{}.csv", dataset.title),
            &data
        ) {
            println!("CSV file saved to src/{}.csv", dataset.title);
        } else {
            println!("Failed to save CSV file");
            return HttpResponse::InternalServerError().body("Failed to save CSV file");
        }
    } else {
        return HttpResponse::BadRequest().body("No file uploaded");
    }
    
    // Add to datasets.json
    match fs::read_to_string("src/datasets.json") {
        Ok(json_str) => {
            match serde_json::from_str::<Datasets>(&json_str) {
                Ok(mut datasets) => {
                    datasets.datasets.push(dataset);
                    match fs::write("src/datasets.json", serde_json::to_string_pretty(&datasets).unwrap()) {
                        Ok(_) => HttpResponse::Ok().json(datasets),
                        Err(_) => HttpResponse::InternalServerError().finish()
                    }
                },
                Err(_) => HttpResponse::InternalServerError().finish()
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish()
    }
}

fn train_dt_model(params: DecisionTreeClassifierParameters) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let input = readers::csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
        File::open("src/iris_input_data.csv").unwrap(),
        readers::csv::CSVDefinition::default()
    ).unwrap();

    let filepath_iris_classes = "src/iris_classes.csv";
    let y_u32s: Vec<u32> = CsvReader::new(File::open(filepath_iris_classes).unwrap()).finish().unwrap()
        .column("variety").unwrap().clone()
        .i64().unwrap().into_no_null_iter().collect::<Vec<i64>>()
        .into_iter().map(|x| x as u32).collect();

    let model = DecisionTreeClassifier::fit(&input, &y_u32s, params).unwrap();
    let predictions = model.predict(&input).unwrap();
    Ok(predictions)
}

fn train_rf_model(params: RandomForestClassifierParameters) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let input = readers::csv::matrix_from_csv_source::<f32, Vec<_>, DenseMatrix<_>>(
        File::open("src/iris_input_data.csv").unwrap(),
        readers::csv::CSVDefinition::default()
    ).unwrap();

    let y_u8s: Vec<u8> = CsvReader::new(File::open("src/iris_classes.csv").unwrap()).finish().unwrap()
        .column("variety").unwrap().clone()
        .i64().unwrap().into_no_null_iter().collect::<Vec<i64>>()
        .into_iter().map(|x| x as u8).collect();

    let model = RandomForestClassifier::fit(&input, &y_u8s, params).unwrap();
    let predictions = model.predict(&input).unwrap();
    Ok(predictions)
}

fn train_lr_model() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let input = readers::csv::matrix_from_csv_source::<f32, Vec<_>, DenseMatrix<_>>(
        File::open("src/iris_input_data.csv").unwrap(),
        readers::csv::CSVDefinition::default()
    ).unwrap();

    let y_f32s: Vec<f32> = CsvReader::new(File::open("src/iris_classes.csv").unwrap()).finish().unwrap()
        .column("variety").unwrap().clone()
        .i64().unwrap().into_no_null_iter().collect::<Vec<i64>>()
        .into_iter().map(|x| x as f32).collect();

    let model = LinearRegression::fit(&input, &y_f32s, Default::default()).unwrap();
    let predictions = model.predict(&input).unwrap();
    Ok(predictions)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting server at http://127.0.0.1:8080");
    HttpServer::new(|| {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .route("/datasets", web::get().to(get_datasets))
            .route("/datasets", web::post().to(add_dataset))
            .route("/train/dt", web::post().to(train_decision_tree))
            .route("/train/rf", web::post().to(train_random_forest))
            .route("/train/lr", web::post().to(train_linear_regression))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

// spolav after train_dt_model, we just run the verification right here right now!!!