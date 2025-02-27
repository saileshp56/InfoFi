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
use ethers::{
    prelude::*,
    providers::{Http, Provider},
    signers::{LocalWallet, Signer},
    contract::Contract,
};
use std::sync::Arc;
use std::str::FromStr;
use rand;
use lazy_static;
use dotenv::dotenv;
use std::env;

lazy_static::lazy_static! {
    static ref PROVIDER_URLS: std::collections::HashMap<u64, String> = {
        let mut m = std::collections::HashMap::new();
        m.insert(1, env::var("RPC_URL_ETH_MAINNET").unwrap_or_default());
        m.insert(137, env::var("RPC_URL_POLYGON").unwrap_or_default());
        m.insert(42161, env::var("RPC_URL_ARBITRUM").unwrap_or_default());
        m.insert(10, env::var("RPC_URL_OPTIMISM").unwrap_or_default());
        m.insert(56, env::var("RPC_URL_BSC").unwrap_or_default());
        m.insert(11155111, env::var("RPC_URL_SEPOLIA").unwrap_or_default());
        m.insert(1315, env::var("RPC_URL_AENEID").unwrap_or_default());

        m
    };

    static ref WALLET: LocalWallet = {
        let private_key = env::var("WALLET_PRIVATE_KEY").expect("WALLET_PRIVATE_KEY must be set in .env file");
        let private_key = if private_key.starts_with("0x") {
            private_key[2..].to_string()
        } else {
            private_key
        };
        private_key.parse::<LocalWallet>().expect("Invalid private key in .env file")
    };

    static ref DEFAULT_CHAIN_ID: u64 = {
        env::var("DEFAULT_CHAIN_ID").unwrap_or_else(|_| "11155111".to_string())
            .parse::<u64>().unwrap_or(11155111)
    };

    static ref FACTORY_ADDRESSES: std::collections::HashMap<u64, Address> = {
        let mut m = std::collections::HashMap::new();
        
        // Try to load factory addresses for different chains from env vars
        if let Ok(addr) = env::var("FACTORY_ADDRESS_AENEID") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(1315, parsed);
            }
        }
        
        // Add a fallback address if none are configured
        if m.is_empty() {
            let fallback = "0x1234567890123456789012345678901234567890";
            m.insert(*DEFAULT_CHAIN_ID, Address::from_str(fallback).unwrap());
        }
        
        m
    };

    static ref PAYMENT_TOKENS: std::collections::HashMap<u64, Address> = {
        let mut m = std::collections::HashMap::new();
        
        // Try to load payment token addresses from environment variables
        if let Ok(addr) = env::var("PAYMENT_TOKEN_ETH_MAINNET") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(1, parsed);
            }
        } else {
            // Fallback to hardcoded USDC address
            m.insert(1, Address::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap());
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKEN_POLYGON") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(137, parsed);
            }
        } else {
            m.insert(137, Address::from_str("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174").unwrap());
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKEN_ARBITRUM") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(42161, parsed);
            }
        } else {
            m.insert(42161, Address::from_str("0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8").unwrap());
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKEN_OPTIMISM") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(10, parsed);
            }
        } else {
            m.insert(10, Address::from_str("0x7F5c764cBc14f9669B88837ca1490cCa17c31607").unwrap());
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKEN_BSC") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(56, parsed);
            }
        } else {
            m.insert(56, Address::from_str("0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56").unwrap());
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKENS_AENEID") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(1315, parsed);
            }
        } else {
            // For testnets, default to ETH (special address)
            m.insert(11155111, Address::from_str("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE").unwrap());
        }
        
        m
    };
}

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
    chain_id: Option<u64>,
}

#[derive(Serialize, Deserialize)]
struct Datasets {
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct EvmRequest {
    chain_id: Option<u64>,
    contract_address: Option<String>,
    function_name: Option<String>,
    function_args: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct BondingCurveInfo {
    address: String,
    name: String,
    symbol: String,
    chain_id: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct BondingCurves {
    curves: std::collections::HashMap<String, BondingCurveInfo>,
}

#[derive(Deserialize)]
struct CreateBondingCurveRequest {
    dataset_title: String,
    chain_id: Option<u64>,
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
    
    // Create bonding curve for the dataset if chain_id is provided
    if dataset.chain_id.is_some() {
        let req = CreateBondingCurveRequest {
            dataset_title: dataset.title.clone(),
            chain_id: dataset.chain_id,
        };
        
        match create_bonding_curve_for_dataset(req).await {
            Ok(curve_info) => {
                println!("Created bonding curve at address: {}", curve_info.address);
            },
            Err(e) => {
                println!("Failed to create bonding curve: {}", e);
            }
        }
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

async fn create_bonding_curve(req: web::Json<CreateBondingCurveRequest>) -> HttpResponse {
    match create_bonding_curve_for_dataset(req.into_inner()).await {
        Ok(result) => HttpResponse::Ok().json(result),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e))
    }
}

async fn create_bonding_curve_for_dataset(req: CreateBondingCurveRequest) -> Result<BondingCurveInfo, Box<dyn std::error::Error>> {
    // Use the global wallet
    let wallet = WALLET.clone();
    
    // Determine which chain to connect to using the global PROVIDER_URLS
    let chain_id = req.chain_id.unwrap_or(*DEFAULT_CHAIN_ID);
    let provider_url = PROVIDER_URLS.get(&chain_id)
        .ok_or_else(|| format!("No provider URL configured for chain ID {}", chain_id))?;
    
    let provider = Provider::<Http>::try_from(provider_url.as_str())?;
    let client = Arc::new(SignerMiddleware::new(provider, wallet.with_chain_id(chain_id)));
    
    // Load factory ABI
    let abi_json = std::fs::read_to_string("src/abi/factory_abi.json")?;
    
    // Get factory address for this chain
    let factory_address = FACTORY_ADDRESSES.get(&chain_id)
        .ok_or_else(|| format!("No factory address configured for chain ID {}", chain_id))?;
    
    // Create contract instance
    let abi = serde_json::from_str::<ethers::abi::Abi>(&abi_json)?;
    let factory = Contract::new(*factory_address, abi, client);
    
    // Generate random name and symbol based on dataset title
    let dataset_title = req.dataset_title.clone();
    let random_suffix = format!("{:x}", rand::random::<u32>());
    let name = format!("{} Token", dataset_title);
    let symbol = format!("{}{}", dataset_title.chars().take(3).collect::<String>().to_uppercase(), random_suffix.chars().take(3).collect::<String>());
    
    // Get the payment token for this chain
    let payment_token = PAYMENT_TOKENS.get(&chain_id)
        .copied()
        .unwrap_or(Address::from_str("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE").unwrap()); // Default to ETH
    
    // Call createBondingCurve function
    let tx = factory.method::<_, Address>(
        "createBondingCurve", 
        (payment_token, name.clone(), symbol.clone())
    )?;
    
    let pending_tx = tx.send().await?;
    let receipt = pending_tx.await?;
    
    // Extract the bonding curve address from the event logs
    let logs = receipt.unwrap().logs;
    
    // Find the BondingCurveCreated event
    let mut bonding_curve_address = String::new();
    for log in logs {
        if log.topics.len() >= 3 {
            // The first topic is the event signature
            if log.topics[0] == ethers::utils::keccak256("BondingCurveCreated(uint256,address)".as_bytes()).into() {
                // The third topic contains the address
                let address_bytes = log.topics[2].as_bytes();
                let address = Address::from_slice(&address_bytes[12..32]); // Extract the address part
                bonding_curve_address = format!("{:?}", address);
                break;
            }
        }
    }
    
    if bonding_curve_address.is_empty() {
        return Err("Failed to extract bonding curve address from transaction receipt".into());
    }
    
    // Create bonding curve info
    let curve_info = BondingCurveInfo {
        address: bonding_curve_address,
        name,
        symbol,
        chain_id,
    };
    
    // Save to bonding_curves.json
    let curves_file = "src/bonding_curves.json";
    let mut bonding_curves = match fs::read_to_string(curves_file) {
        Ok(json_str) => serde_json::from_str::<BondingCurves>(&json_str)
            .unwrap_or_else(|_| BondingCurves::default()),
        Err(_) => BondingCurves::default(),
    };
    
    bonding_curves.curves.insert(dataset_title, curve_info.clone());
    
    fs::write(
        curves_file,
        serde_json::to_string_pretty(&bonding_curves).unwrap()
    )?;

    println!("Bonding curve created for dataset: {}", req.dataset_title);
    println!("Bonding curve address: {}", curve_info.address);
    
    Ok(curve_info)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Create directories if they don't exist
    std::fs::create_dir_all("src/abi").unwrap_or_default();
    
    // Create bonding_curves.json if it doesn't exist
    if !std::path::Path::new("src/bonding_curves.json").exists() {
        fs::write(
            "src/bonding_curves.json",
            serde_json::to_string_pretty(&BondingCurves::default()).unwrap()
        ).unwrap_or_default();
    }
    
    // Create datasets.json if it doesn't exist
    if !std::path::Path::new("src/datasets.json").exists() {
        fs::write(
            "src/datasets.json",
            serde_json::to_string_pretty(&Datasets { datasets: Vec::new() }).unwrap()
        ).unwrap_or_default();
    }
    
    println!("Starting server at http://127.0.0.1:8080");
    println!("Using wallet address: {:?}", WALLET.address());
    println!("Default chain ID: {}", *DEFAULT_CHAIN_ID);
    
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
            // .route("/bonding-curve/create", web::post().to(create_bonding_curve))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

// spolav after train_dt_model, we just run the verification right here right now!!!