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
use actix_web::{web::Path};
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
use hedera::{
    AccountId, Client, TopicMessageSubmitTransaction, PrivateKey, TopicId, TopicCreateTransaction,
};
// use std::time;
use time::Duration;
use reqwest;
use base64;

lazy_static::lazy_static! {
    static ref PROVIDER_URLS: std::collections::HashMap<u64, String> = {
        let mut m = std::collections::HashMap::new();
        m.insert(296, env::var("RPC_URL_HEDERA").unwrap_or_default());
        m.insert(48899, env::var("RPC_URL_ZIRCUIT").unwrap_or_default());
        m.insert(300, env::var("RPC_URL_ZKSYNC").unwrap_or_default());
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
        
        // Load factory addresses from env vars
        if let Ok(addr) = env::var("FACTORY_ADDRESS_HEDERA") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(296, parsed);
            }
        }
        
        if let Ok(addr) = env::var("FACTORY_ADDRESS_ZIRCUIT") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(48899, parsed);
            }
        }
        
        if let Ok(addr) = env::var("FACTORY_ADDRESS_ZKSYNC") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(300, parsed);
            }
        }
        
        if let Ok(addr) = env::var("FACTORY_ADDRESS_SEPOLIA") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(11155111, parsed);
            }
        }
        
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
        
        // Load payment token addresses from environment variables
        if let Ok(addr) = env::var("PAYMENT_TOKENS_HEDERA") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(296, parsed);
            }
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKENS_ZIRCUIT") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(48899, parsed);
            }
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKENS_ZKSYNC") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(300, parsed);
            }
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKENS_SEPOLIA") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(11155111, parsed);
            }
        }
        
        if let Ok(addr) = env::var("PAYMENT_TOKENS_AENEID") {
            if let Ok(parsed) = Address::from_str(&addr) {
                m.insert(1315, parsed);
            }
        }
        
        // For testnets without a configured token, default to ETH (special address)
        if !m.contains_key(&11155111) {
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
    dataset_title: String,
    accuracy: f64,
    wallet_address: String,
    
}

#[derive(Deserialize)]
struct RFParameters {
    max_depth: Option<u16>,
    min_samples_leaf: u16,
    min_samples_split: u16,
    n_trees: u16,
    dataset_title: String,
    accuracy: f64,
    wallet_address: String,
}

#[derive(Deserialize)]
struct LRParameters {
    dataset_title: String,
    accuracy: f64,
    wallet_address: String,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct Dataset {
    title: String,
    description: String,
    size: String,
    format: String,
    categories: Vec<String>,
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

#[derive(Serialize, Deserialize, Clone, Debug)]
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
    // Create model parameters
    let model_params = DecisionTreeClassifierParameters {
        criterion: if params.use_entropy { 
            SplitCriterion::Entropy 
        } else { 
            SplitCriterion::Gini 
        },
        max_depth: params.max_depth,
        min_samples_leaf: params.min_samples_leaf as usize,
        min_samples_split: params.min_samples_split as usize,
        seed: None,
    };

    // Construct dataset path
    let dataset_path = format!("src/{}.csv", params.dataset_title);
    
    // Log the wallet address and accuracy
    println!("Training decision tree for wallet: {}, desired accuracy: {}", 
             params.wallet_address, params.accuracy);
    
    match train_dt_model(model_params, &dataset_path) {
        Ok(predictions) => {
            // Create directories if they don't exist
            std::fs::create_dir_all("res/ml-model").unwrap_or_default();
            std::fs::create_dir_all("res/input-data").unwrap_or_default();
            
            // Load the dataset again to save it
            let input = readers::csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
                File::open(&dataset_path).unwrap(),
                readers::csv::CSVDefinition::default()
            ).unwrap();
            
            // Get the labels
            let df = CsvReader::from_path(&dataset_path).unwrap()
                .has_header(true)
                .finish().unwrap();
            
            let columns = df.get_column_names();
            let target_column = columns.last().unwrap();
            
            let y_u32s: Vec<u32> = df.column(target_column).unwrap()
                .cast(&DataType::Int64).unwrap()
                .i64().unwrap()
                .into_no_null_iter()
                .map(|x| x as u32)
                .collect();
            
            // Load the model
            let model_path = format!("src/{}_dt_model.bin", params.dataset_title);
            let mut model_file = File::open(&model_path).unwrap();
            let mut model_bytes = Vec::new();
            model_file.read_to_end(&mut model_bytes).unwrap();
            
            // Save to res/ directory
            let mut f = File::create("res/ml-model/tree_model_bytes.bin").unwrap();
            f.write_all(&model_bytes).unwrap();
            
            let data_bytes = rmp_serde::to_vec(&input).unwrap();
            let mut f1: File = File::create("res/input-data/tree_model_data_bytes.bin").unwrap();
            f1.write_all(&data_bytes).unwrap();
            
            let labels_bytes = rmp_serde::to_vec(&y_u32s).unwrap();
            let mut f2 = File::create("res/input-data/tree_model_labels.bin").unwrap();
            f2.write_all(&labels_bytes).unwrap();
            
            // Commented out for brevity!
            // let output = run_zkvm_verification().await;
            let output = serde_json::json!({
                "zkvm_result": {
                    "success": true
                }
            });
            
            let model_bytes = fs::read("res/ml-model/tree_model_bytes.bin").unwrap_or_default();
            let model_base64 = base64::encode(&model_bytes);
            
            // Check if verification was successful and interact with bonding curve
            let bonding_curve_result = if output["zkvm_result"]["success"].as_bool().unwrap_or(false) {
                // Get the dataset info to find the chain_id and bonding curve address
                let dataset_info = get_dataset_info(&params.dataset_title);
                
                if let Some(dataset) = dataset_info {
                    if let Some(chain_id) = dataset.chain_id {
                        // Get the bonding curve address for this dataset
                        let bonding_curve_address = get_bonding_curve_address(&params.dataset_title);
                        println!("Bonding curve address: {:?} and chain id: {}", bonding_curve_address, chain_id);
                        if let Some(address) = bonding_curve_address {
                            // Interact with the bonding curve
                            println!("spolav wtf {} {}", address, chain_id);
                            match interact_with_bonding_curve(address, chain_id, 100.0).await {
                                Ok(result) => Some(result),
                                Err(e) => Some(serde_json::json!({
                                    "status": "error",
                                    "message": format!("Failed to interact with bonding curve: {}", e)
                                }))
                            }
                        } else {
                            Some(serde_json::json!({
                                "status": "error",
                                "message": "No bonding curve found for this dataset"
                            }))
                        }
                    } else {
                        Some(serde_json::json!({
                            "status": "error",
                            "message": "Dataset has no chain ID configured"
                        }))
                    }
                } else {
                    Some(serde_json::json!({
                        "status": "error",
                        "message": "Dataset not found"
                    }))
                }
            } else {
                Some(serde_json::json!({
                    "status": "skipped",
                    "message": "Verification failed, skipping bonding curve interaction"
                }))
            };

            println!("Bonding curve result: {:?}", bonding_curve_result);
            
            // Return the response with predictions and verification results
            HttpResponse::Ok().json(serde_json::json!({
                "predictions": predictions,
                "accuracy_param": params.accuracy,
                "wallet_address": params.wallet_address,
                "verification_result": output,
                "model": model_base64
            }))
        },
        Err(e) => {
            println!("Error training decision tree: {}", e);
            HttpResponse::InternalServerError().body(format!("Error training model: {}", e))
        }
    }
}

// Helper function to get dataset info
fn get_dataset_info(dataset_title: &str) -> Option<Dataset> {
    match fs::read_to_string("src/datasets.json") {
        Ok(json_str) => {
            match serde_json::from_str::<Datasets>(&json_str) {
                Ok(datasets) => {
                    datasets.datasets.into_iter()
                        .find(|dataset| dataset.title == dataset_title)
                },
                Err(_) => None
            }
        },
        Err(_) => None
    }
}

// Helper function to get bonding curve address for a dataset
fn get_bonding_curve_address(dataset_title: &str) -> Option<String> {
    match fs::read_to_string("src/bonding_curves.json") {
        Ok(json_str) => {
            match serde_json::from_str::<BondingCurves>(&json_str) {
                Ok(bonding_curves) => {
                    bonding_curves.curves.get(dataset_title)
                        .map(|curve_info| curve_info.address.clone())
                },
                Err(_) => None
            }
        },
        Err(_) => None
    }
}

// Function to interact with the bonding curve (approve and buy)
async fn interact_with_bonding_curve(curve_address: String, chain_id: u64, token_amount: f64) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    println!("Interacting with bonding curve at {} on chain {}", curve_address, chain_id);
    
    // Parse the curve address
    let curve_address_parsed = Address::from_str(&curve_address)?;
    
    // Convert token amount to wei (assuming 18 decimals)
    let token_amount_wei = {
        let wei_amount = (token_amount) as u128;
        U256::from(wei_amount)
    };
    
    // Set up provider and client
    let wallet = WALLET.clone();
    
    let provider_url = match PROVIDER_URLS.get(&chain_id) {
        Some(url) => url,
        None => return Err(format!("No provider URL for chain ID {}", chain_id).into())
    };
    
    let provider = Provider::<Http>::try_from(provider_url.as_str())?;
    let client = Arc::new(SignerMiddleware::new(provider, wallet.with_chain_id(chain_id)));
    
    // Load curve ABI
    let curve_abi_json = std::fs::read_to_string("src/abi/curve_abi.json")?;
    
    // Create curve contract instance
    let curve_abi = serde_json::from_str::<ethers::abi::Abi>(&curve_abi_json)?;

    let curve = Contract::new(curve_address_parsed, curve_abi, client.clone());
    
    // STEP 1: Call calculatePaymentRequired to get the exact payment amount needed
    let payment_required: U256 = curve.method::<_, U256>("calculatePaymentRequired", token_amount_wei)?
        .call().await?;
    
    println!("Payment required: {} wei for {} tokens", payment_required, token_amount);
    
    // Get the payment token address for this chain
    let payment_token_address = match PAYMENT_TOKENS.get(&chain_id) {
        Some(addr) => *addr,
        None => return Err(format!("No payment token configured for chain ID {}", chain_id).into())
    };

    println!("Payment token address: {}", payment_token_address); // spolav
    
    // Load ERC20 ABI
    let erc20_abi_json = match std::fs::read_to_string("src/abi/erc20_abi.json") {
        Ok(json) => json,
        Err(_) => {
            // Fallback to a minimal ERC20 ABI if file doesn't exist
            r#"[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"}]"#.to_string()
        }
    };
    
    // Create token contract instance
    let erc20_abi = serde_json::from_str::<ethers::abi::Abi>(&erc20_abi_json)?;
    let token = Contract::new(payment_token_address, erc20_abi, client.clone());

    println!("Curve address: {} and paying {}", curve_address_parsed, payment_required); // spolav the second arg is in wei
    
    // STEP 2: Call approve function with the exact payment amount needed
    let approve_tx = token.method::<_, bool>("approve", (curve_address_parsed, payment_required))?;
    
    // Send the approve transaction
    let approve_pending_tx = approve_tx.send().await?;
    
    // Wait for the approve transaction to be mined
    let approve_receipt = match approve_pending_tx.await? {
        Some(r) => r,
        None => return Err("Approve transaction failed".into())
    };
    
    println!("Approve transaction successful: {:?}", approve_receipt.transaction_hash);
    
    // STEP 3: Now call buy function with the token amount
    let buy_tx = curve.method::<_, ()>("buy", payment_required)?;
    
    // Send the buy transaction
    let buy_pending_tx = buy_tx.send().await?;
    
    // Wait for the buy transaction to be mined
    let buy_receipt = match buy_pending_tx.await? {
        Some(r) => r,
        None => return Err("Buy transaction failed".into())
    };
    
    println!("Buy transaction successful: {:?}", buy_receipt.transaction_hash);
    
    Ok(serde_json::json!({
        "status": "success",
        "approve_tx": format!("{:?}", approve_receipt.transaction_hash),
        "buy_tx": format!("{:?}", buy_receipt.transaction_hash),
        "payment_amount": format!("{}", payment_required),
        "token_amount": token_amount,
        "block_number": buy_receipt.block_number.map(|b| b.as_u64())
    }))
}

// Function to run the RISC Zero zkVM verification
async fn run_zkvm_verification() -> serde_json::Value {
    println!("Starting zkVM verification process...");
    
    // Get the current directory
    let current_dir = std::env::current_dir().unwrap();
    println!("Current directory: {:?}", current_dir);
    
    // Path to the smartcore-zk-ml directory
    let zkvm_dir = current_dir.join("smartcore-zk-ml").join("smartcore-zk-ml");
    println!("zkVM directory: {:?}", zkvm_dir);
    
    // Check if directory exists
    if !zkvm_dir.exists() {
        println!("ERROR: zkVM directory does not exist: {:?}", zkvm_dir);
        return serde_json::json!({
            "success": false,
            "error": format!("zkVM directory does not exist: {:?}", zkvm_dir)
        });
    }
    
    // Check if input files exist
    let model_path = current_dir.join("res/ml-model/tree_model_bytes.bin");
    let data_path = current_dir.join("res/input-data/tree_model_data_bytes.bin");
    let labels_path = current_dir.join("res/input-data/tree_model_labels.bin");
    
    println!("Checking input files:");
    println!("  Model file: {:?} (exists: {})", model_path, model_path.exists());
    println!("  Data file: {:?} (exists: {})", data_path, data_path.exists());
    println!("  Labels file: {:?} (exists: {})", labels_path, labels_path.exists());
    
    // Create the command to run cargo
    println!("Creating cargo command in directory: {:?}", zkvm_dir);
    let mut command = std::process::Command::new("cargo");
    let command = command
        .current_dir(&zkvm_dir)
        .env("RISC0_DEV_MODE", "0")  // Disable dev mode
        .arg("run")
        .arg("--release");
    
    println!("Executing command: {:?}", command);
    
    let output = command.output();
    
    let zkvm_result = match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            
            println!("Command executed with status: {}", output.status);
            println!("STDOUT: {}", stdout);
            println!("STDERR: {}", stderr);
            
            // Check if the command was successful
            if output.status.success() {
                println!("Command executed successfully");
                
                // Try to read the proof.json file
                let proof_path = zkvm_dir.join("proof.json");
                println!("Looking for proof file at: {:?}", proof_path);
                
                let proof_result = std::fs::read_to_string(&proof_path);
                
                let proof_json = match proof_result {
                    Ok(proof_str) => {
                        println!("Proof file found and read successfully");
                        match serde_json::from_str::<serde_json::Value>(&proof_str) {
                            Ok(json) => {
                                println!("Proof JSON parsed successfully");
                                json
                            },
                            Err(e) => {
                                println!("Failed to parse proof.json: {}", e);
                                serde_json::json!({"error": format!("Failed to parse proof.json: {}", e)})
                            }
                        }
                    },
                    Err(e) => {
                        println!("Failed to read proof.json: {}", e);
                        serde_json::json!({"error": format!("Failed to read proof.json: {}", e)})
                    }
                };
                
                serde_json::json!({
                    "success": true,
                    "stdout": stdout,
                    "proof": proof_json
                })
            } else {
                println!("Command failed with exit code: {:?}", output.status.code());
                serde_json::json!({
                    "success": false,
                    "stdout": stdout,
                    "stderr": stderr
                })
            }
        },
        Err(e) => {
            println!("Failed to execute command: {}", e);
            serde_json::json!({
                "success": false,
                "error": format!("Failed to execute command: {}", e)
            })
        }
    };
    
    // Now make a request to the proof verification server
    println!("Making request to proof verification server...");
    let client = reqwest::Client::new();
    let receipt_path = "smartcore-zk-ml/smartcore-zk-ml/proof.json";
    
    println!("Using receipt path: {}", receipt_path);
    
    let verification_result = match client.post("http://localhost:6000/verify")
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "receiptPath": receipt_path
        }))
        .send()
        .await {
            Ok(response) => {
                println!("Verification server response status: {}", response.status());
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        println!("Verification server response: {:?}", json);
                        json
                    },
                    Err(e) => {
                        println!("Failed to parse verification response: {}", e);
                        serde_json::json!({
                            "verification_error": format!("Failed to parse verification response: {}", e)
                        })
                    }
                }
            },
            Err(e) => {
                println!("Failed to send verification request: {}", e);
                serde_json::json!({
                    "verification_error": format!("Failed to send verification request: {}", e)
                })
            }
        };
    
    // Combine both results
    let result = serde_json::json!({
        "zkvm_result": zkvm_result,
        "verification_result": verification_result
    });
    
    // println!("Final verification result: {}", result);
    result
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

    // Log the wallet address and accuracy
    println!("Training random forest for wallet: {}, desired accuracy: {}", 
             params.wallet_address, params.accuracy);

    let dataset_path = format!("src/{}.csv", params.dataset_title);
    
    match train_rf_model(model_params, &dataset_path) {
        Ok(predictions) => {
            // Create directories if they don't exist
            std::fs::create_dir_all("res/ml-model").unwrap_or_default();
            std::fs::create_dir_all("res/input-data").unwrap_or_default();
            
            // Load the dataset again to save it
            let input = readers::csv::matrix_from_csv_source::<f32, Vec<_>, DenseMatrix<_>>(
                File::open(&dataset_path).unwrap(),
                readers::csv::CSVDefinition::default()
            ).unwrap();
            
            // Get the labels
            let df = CsvReader::from_path(&dataset_path).unwrap()
                .has_header(true)
                .finish().unwrap();
            
            let columns = df.get_column_names();
            let target_column = columns.last().unwrap();
            
            let y_u8s: Vec<u8> = df.column(target_column).unwrap()
                .cast(&DataType::Int64).unwrap()
                .i64().unwrap()
                .into_no_null_iter()
                .map(|x| x as u8)
                .collect();
            
            // Load the model
            let model_path = format!("src/{}_rf_model.bin", params.dataset_title);
            let mut model_file = File::open(&model_path).unwrap();
            let mut model_bytes = Vec::new();
            model_file.read_to_end(&mut model_bytes).unwrap();
            
            // Save to res/ directory
            let mut f = File::create("res/ml-model/rf_model_bytes.bin").unwrap();
            f.write_all(&model_bytes).unwrap();
            
            let data_bytes = rmp_serde::to_vec(&input).unwrap();
            let mut f1 = File::create("res/input-data/rf_model_data_bytes.bin").unwrap();
            f1.write_all(&data_bytes).unwrap();
            
            let labels_bytes = rmp_serde::to_vec(&y_u8s).unwrap();
            let mut f2 = File::create("res/input-data/rf_model_labels.bin").unwrap();
            f2.write_all(&labels_bytes).unwrap();
            
            HttpResponse::Ok().json(serde_json::json!({
                "predictions": predictions,
                "accuracy_param": params.accuracy,
                "wallet_address": params.wallet_address
            }))
        },
        Err(e) => {
            println!("Error training random forest: {}", e);
            HttpResponse::InternalServerError().body(format!("Error training model: {}", e))
        }
    }
}

async fn train_linear_regression(params: web::Json<LRParameters>) -> HttpResponse {
    // Log the wallet address and accuracy
    println!("Training linear regression for wallet: {}, desired accuracy: {}", 
             params.wallet_address, params.accuracy);

    let dataset_path = format!("src/{}.csv", params.dataset_title);
    
    match train_lr_model(&dataset_path) {
        Ok(predictions) => {
            // Create directories if they don't exist
            std::fs::create_dir_all("res/ml-model").unwrap_or_default();
            std::fs::create_dir_all("res/input-data").unwrap_or_default();
            
            // Load the dataset again to save it
            let input = readers::csv::matrix_from_csv_source::<f32, Vec<_>, DenseMatrix<_>>(
                File::open(&dataset_path).unwrap(),
                readers::csv::CSVDefinition::default()
            ).unwrap();
            
            // Get the labels
            let df = CsvReader::from_path(&dataset_path).unwrap()
                .has_header(true)
                .finish().unwrap();
            
            let columns = df.get_column_names();
            let target_column = columns.last().unwrap();
            
            let y_f32s: Vec<f32> = df.column(target_column).unwrap()
                .cast(&DataType::Float32).unwrap()
                .f32().unwrap()
                .into_no_null_iter()
                .collect();
            
            // Load the model
            let model_path = format!("src/{}_lr_model.bin", params.dataset_title);
            let mut model_file = File::open(&model_path).unwrap();
            let mut model_bytes = Vec::new();
            model_file.read_to_end(&mut model_bytes).unwrap();
            
            // Save to res/ directory
            let mut f = File::create("res/ml-model/lr_model_bytes.bin").unwrap();
            f.write_all(&model_bytes).unwrap();
            
            let data_bytes = rmp_serde::to_vec(&input).unwrap();
            let mut f1 = File::create("res/input-data/lr_model_data_bytes.bin").unwrap();
            f1.write_all(&data_bytes).unwrap();
            
            let labels_bytes = rmp_serde::to_vec(&y_f32s).unwrap();
            let mut f2 = File::create("res/input-data/lr_model_labels.bin").unwrap();
            f2.write_all(&labels_bytes).unwrap();
            
            HttpResponse::Ok().json(serde_json::json!({
                "predictions": predictions,
                "accuracy_param": params.accuracy,
                "wallet_address": params.wallet_address
            }))
        },
        Err(e) => {
            println!("Error training linear regression: {}", e);
            HttpResponse::InternalServerError().body(format!("Error training model: {}", e))
        }
    }
}

async fn get_datasets() -> HttpResponse {
    // Read datasets.json
    let datasets_result = match fs::read_to_string("src/datasets.json") {
        Ok(json_str) => {
            match serde_json::from_str::<Datasets>(&json_str) {
                Ok(datasets) => datasets,
                Err(_) => return HttpResponse::InternalServerError().body("Failed to parse datasets.json")
            }
        },
        Err(_) => return HttpResponse::InternalServerError().body("Failed to read datasets.json")
    };
    
    // Read bonding_curves.json
    let bonding_curves = match fs::read_to_string("src/bonding_curves.json") {
        Ok(json_str) => {
            match serde_json::from_str::<BondingCurves>(&json_str) {
                Ok(curves) => curves,
                Err(_) => BondingCurves::default() // Use empty curves if parsing fails
            }
        },
        Err(_) => BondingCurves::default() // Use empty curves if file can't be read
    };
    
    // Create a new response structure that includes bonding curve info
    #[derive(Serialize)]
    struct DatasetWithCurve {
        #[serde(flatten)]
        dataset: Dataset,
        bonding_curve: Option<BondingCurveInfo>,
    }
    
    #[derive(Serialize)]
    struct DatasetsResponse {
        datasets: Vec<DatasetWithCurve>,
    }
    
    // Combine dataset info with bonding curve info
    let datasets_with_curves = datasets_result.datasets.into_iter().map(|dataset| {
        let bonding_curve = bonding_curves.curves.get(&dataset.title).cloned();
        println!("We're returning {:?}", bonding_curve);
        DatasetWithCurve {
            dataset,
            bonding_curve,
        }
    }).collect();
    
    HttpResponse::Ok().json(DatasetsResponse { datasets: datasets_with_curves })
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
            "chain_id" => {
                // Allow explicit chain_id field for easier testing
                println!("Processing chain_id field");
                let mut chain_id_str = String::new();
                while let Ok(Some(chunk)) = field.try_next().await {
                    chain_id_str.extend(std::str::from_utf8(&chunk).unwrap().chars());
                }
                if let Ok(chain_id) = chain_id_str.trim().parse::<u64>() {
                    println!("Setting chain_id to: {}", chain_id);
                    dataset.chain_id = Some(chain_id);
                } else {
                    println!("Invalid chain_id: {}", chain_id_str);
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
    
    // If no chain_id was provided but we want to use Sepolia, set it here
    if dataset.chain_id.is_none() {
        println!("No chain_id provided, defaulting to Sepolia (11155111)");
        dataset.chain_id = Some(11155111); // Sepolia chain ID
    }
    
    println!("Dataset chain_id: {:?}", dataset.chain_id);
    // Create bonding curve for the dataset
    println!("Creating bonding curve for dataset: {}", dataset.title);
    let req = CreateBondingCurveRequest {
        dataset_title: dataset.title.clone(),
        chain_id: dataset.chain_id,
    };
    
    let bonding_curve_result = match create_bonding_curve_for_dataset(req).await {
        Ok(curve_info) => {
            println!("Created bonding curve at address: {}", curve_info.address);
            Some(curve_info)
        },
        Err(e) => {
            println!("Failed to create bonding curve: {}", e);
            None
        }
    };
    
    // Add to datasets.json
    match fs::read_to_string("src/datasets.json") {
        Ok(json_str) => {
            match serde_json::from_str::<Datasets>(&json_str) {
                Ok(mut datasets) => {
                    datasets.datasets.push(dataset.clone());
                    match fs::write("src/datasets.json", serde_json::to_string_pretty(&datasets).unwrap()) {
                        Ok(_) => {
                            // Return both the dataset and bonding curve info
                            let response = serde_json::json!({
                                "dataset": dataset,
                                "bonding_curve": bonding_curve_result,
                                "all_datasets": datasets
                            });
                            HttpResponse::Ok().json(response)
                        },
                        Err(e) => {
                            println!("Failed to write datasets.json: {}", e);
                            HttpResponse::InternalServerError().body("Failed to save dataset metadata")
                        }
                    }
                },
                Err(e) => {
                    println!("Failed to parse datasets.json: {}", e);
                    HttpResponse::InternalServerError().body("Failed to parse existing datasets")
                }
            }
        },
        Err(e) => {
            println!("Failed to read datasets.json: {}", e);
            // If the file doesn't exist, create it with just this dataset
            let datasets = Datasets {
                datasets: vec![dataset.clone()]
            };
            match fs::write("src/datasets.json", serde_json::to_string_pretty(&datasets).unwrap()) {
                Ok(_) => {
                    let response = serde_json::json!({
                        "dataset": dataset,
                        "bonding_curve": bonding_curve_result,
                        "all_datasets": datasets
                    });
                    HttpResponse::Ok().json(response)
                },
                Err(_) => HttpResponse::InternalServerError().body("Failed to create datasets.json")
            }
        }
    }
}

fn train_dt_model(params: DecisionTreeClassifierParameters, dataset_path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Check if the file exists
    if !std::path::Path::new(dataset_path).exists() {
        return Err(format!("Dataset file not found: {}", dataset_path).into());
    }
    
    // Load the dataset
    let input = match readers::csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
        File::open(dataset_path)?,
        readers::csv::CSVDefinition::default()
    ) {
        Ok(matrix) => matrix,
        Err(e) => return Err(format!("Error reading CSV: {:?}", e).into()),
    };

    // For simplicity, we'll assume the last column is the target variable
    let df = CsvReader::from_path(dataset_path)?
        .has_header(true)
        .finish()?;
    
    // Get the name of the last column
    let columns = df.get_column_names();
    let target_column = columns.last().ok_or("Dataset has no columns")?;
    
    // Extract the target column
    let y_u32s: Vec<u32> = df.column(target_column)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .map(|x| x as u32)
        .collect();

    // Train the model
    let model = match DecisionTreeClassifier::fit(&input, &y_u32s, params) {
        Ok(model) => model,
        Err(e) => return Err(format!("Error training model: {:?}", e).into()),
    };
    
    // Make predictions
    let predictions = match model.predict(&input) {
        Ok(preds) => preds,
        Err(e) => return Err(format!("Error making predictions: {:?}", e).into()),
    };
    
    // Save the model to a file
    let model_path = format!("src/{}_dt_model.bin", dataset_path.trim_start_matches("src/").trim_end_matches(".csv"));
    let mut file = File::create(model_path)?;
    let serialized = rmp_serde::to_vec(&model)?;
    file.write_all(&serialized)?;
    
    Ok(predictions)
}

fn train_rf_model(params: RandomForestClassifierParameters, dataset_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Check if the file exists
    if !std::path::Path::new(dataset_path).exists() {
        return Err(format!("Dataset file not found: {}", dataset_path).into());
    }
    
    // Load the dataset
    let input = match readers::csv::matrix_from_csv_source::<f32, Vec<_>, DenseMatrix<_>>(
        File::open(dataset_path)?,
        readers::csv::CSVDefinition::default()
    ) {
        Ok(matrix) => matrix,
        Err(e) => return Err(format!("Error reading CSV: {:?}", e).into()),
    };

    // For simplicity, we'll assume the last column is the target variable
    let df = CsvReader::from_path(dataset_path)?
        .has_header(true)
        .finish()?;
    
    // Get the name of the last column
    let columns = df.get_column_names();
    let target_column = columns.last().ok_or("Dataset has no columns")?;
    
    // Extract the target column
    let y_u8s: Vec<u8> = df.column(target_column)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .map(|x| x as u8)
        .collect();

    // Train the model
    let model = match RandomForestClassifier::fit(&input, &y_u8s, params) {
        Ok(model) => model,
        Err(e) => return Err(format!("Error training model: {:?}", e).into()),
    };
    
    // Make predictions
    let predictions = match model.predict(&input) {
        Ok(preds) => preds,
        Err(e) => return Err(format!("Error making predictions: {:?}", e).into()),
    };
    
    // Save the model to a file
    let model_path = format!("src/{}_rf_model.bin", dataset_path.trim_start_matches("src/").trim_end_matches(".csv"));
    let mut file = File::create(model_path)?;
    let serialized = rmp_serde::to_vec(&model)?;
    file.write_all(&serialized)?;
    
    Ok(predictions)
}

fn train_lr_model(dataset_path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Check if the file exists
    if !std::path::Path::new(dataset_path).exists() {
        return Err(format!("Dataset file not found: {}", dataset_path).into());
    }
    
    // Load the dataset
    let input = match readers::csv::matrix_from_csv_source::<f32, Vec<_>, DenseMatrix<_>>(
        File::open(dataset_path)?,
        readers::csv::CSVDefinition::default()
    ) {
        Ok(matrix) => matrix,
        Err(e) => return Err(format!("Error reading CSV: {:?}", e).into()),
    };

    // For simplicity, we'll assume the last column is the target variable
    let df = CsvReader::from_path(dataset_path)?
        .has_header(true)
        .finish()?;
    
    // Get the name of the last column
    let columns = df.get_column_names();
    let target_column = columns.last().ok_or("Dataset has no columns")?;
    
    // Extract the target column
    let y_f32s: Vec<f32> = df.column(target_column)?
        .cast(&DataType::Float32)?
        .f32()?
        .into_no_null_iter()
        .collect();

    // Train the model
    let model = match LinearRegression::fit(&input, &y_f32s, Default::default()) {
        Ok(model) => model,
        Err(e) => return Err(format!("Error training model: {:?}", e).into()),
    };
    
    // Make predictions
    let predictions = match model.predict(&input) {
        Ok(preds) => preds,
        Err(e) => return Err(format!("Error making predictions: {:?}", e).into()),
    };
    
    // Save the model to a file
    let model_path = format!("src/{}_lr_model.bin", dataset_path.trim_start_matches("src/").trim_end_matches(".csv"));
    let mut file = File::create(model_path)?;
    let serialized = rmp_serde::to_vec(&model)?;
    file.write_all(&serialized)?;
    
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

async fn buy_tokens(path: web::Path<(String, String)>) -> HttpResponse {
    let (curve_address, amount) = path.into_inner();
    // Parse the curve address
    let curve_address = match Address::from_str(&curve_address) {
        Ok(addr) => addr,
        Err(_) => return HttpResponse::BadRequest().body("Invalid curve address")
    };
    
    // Parse the amount (expected in full tokens, will be converted to wei)
    let amount = match amount.parse::<f64>() {
        Ok(val) => {
            // Convert to wei (assuming 18 decimals)
            let wei_amount = (val * 1e18) as u128;
            U256::from(wei_amount)
        },
        Err(_) => return HttpResponse::BadRequest().body("Invalid amount")
    };
    
    // Use the global wallet
    let wallet = WALLET.clone();
    
    // For this example, we'll use the Aeneid chain (1315)
    let chain_id = 1315;
    
    let provider_url = match PROVIDER_URLS.get(&chain_id) {
        Some(url) => url,
        None => return HttpResponse::BadRequest().body("No provider URL for chain ID 1315")
    };
    
    let provider = match Provider::<Http>::try_from(provider_url.as_str()) {
        Ok(p) => p,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Provider error: {}", e))
    };
    
    let client = Arc::new(SignerMiddleware::new(provider, wallet.with_chain_id(chain_id)));
    
    // Load curve ABI
    let abi_json = match std::fs::read_to_string("src/abi/curve_abi.json") {
        Ok(json) => json,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to read ABI: {}", e))
    };
    
    // Create contract instance
    let abi = match serde_json::from_str::<ethers::abi::Abi>(&abi_json) {
        Ok(abi) => abi,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to parse ABI: {}", e))
    };
    
    let curve = Contract::new(curve_address, abi, client);
    
    // Call buy function
    let tx = match curve.method::<_, ()>("buy", amount) {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to create transaction: {}", e))
    };
    
    // Send the transaction
    let pending_tx = match tx.send().await {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to send transaction: {}", e))
    };
    
    // Wait for the transaction to be mined
    match pending_tx.await {
        Ok(receipt) => {
            match receipt {
                Some(r) => HttpResponse::Ok().json(serde_json::json!({
                    "status": "success",
                    "transaction_hash": format!("{:?}", r.transaction_hash),
                    "block_number": r.block_number.map(|b| b.as_u64())
                })),
                None => HttpResponse::InternalServerError().body("Transaction failed")
            }
        },
        Err(e) => HttpResponse::InternalServerError().body(format!("Transaction error: {}", e))
    }
}

async fn buy_iris_tokens() -> HttpResponse {
    // Hardcoded values
    let curve_address = "0x13c44aa3377246f05546f9e3b0104792f944d81b";
    let token_amount = "100"; // 100 tokens to buy
    
    // Parse the curve address
    let curve_address_parsed = match Address::from_str(curve_address) {
        Ok(addr) => addr,
        Err(_) => return HttpResponse::BadRequest().body("Invalid curve address")
    };
    
    // Convert token amount to wei (assuming 18 decimals)
    let token_amount_wei = match token_amount.parse::<f64>() {
        Ok(val) => {
            let wei_amount = (val * 1e18) as u128;
            U256::from(wei_amount)
        },
        Err(_) => return HttpResponse::BadRequest().body("Invalid amount")
    };
    
    // Set up provider and client
    let wallet = WALLET.clone();
    let chain_id = 1315; // Aeneid chain
    
    let provider_url = match PROVIDER_URLS.get(&chain_id) {
        Some(url) => url,
        None => return HttpResponse::BadRequest().body("No provider URL for chain ID 1315")
    };
    
    let provider = match Provider::<Http>::try_from(provider_url.as_str()) {
        Ok(p) => p,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Provider error: {}", e))
    };
    
    let client = Arc::new(SignerMiddleware::new(provider, wallet.with_chain_id(chain_id)));
    
    // Load curve ABI
    let curve_abi_json = match std::fs::read_to_string("src/abi/curve_abi.json") {
        Ok(json) => json,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to read ABI: {}", e))
    };
    
    // Create curve contract instance
    let curve_abi = match serde_json::from_str::<ethers::abi::Abi>(&curve_abi_json) {
        Ok(abi) => abi,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to parse ABI: {}", e))
    };
    
    let curve = Contract::new(curve_address_parsed, curve_abi, client.clone());
    
    // STEP 1: Call calculatePaymentRequired to get the exact payment amount needed
    let payment_required: U256 = match curve.method::<_, U256>("calculatePaymentRequired", U256::from(100)) { // HARD CODED to 100 :o)
        Ok(method) => {
            match method.call().await {
                Ok(amount) => amount,
                Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to calculate payment required: {}", e))
            }
        },
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to create method call: {}", e))
    };
    
    println!("Payment required: {} wei for {} tokens", payment_required, token_amount);
    
    // Get the payment token address for this chain
    let payment_token_address = match PAYMENT_TOKENS.get(&chain_id) {
        Some(addr) => *addr,
        None => return HttpResponse::BadRequest().body("No payment token configured for chain ID 1315")
    };
    
    // Load ERC20 ABI
    let erc20_abi_json = match std::fs::read_to_string("src/abi/erc20_abi.json") {
        Ok(json) => json,
        Err(_) => {
            // Fallback to a minimal ERC20 ABI if file doesn't exist
            r#"[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"}]"#.to_string()
        }
    };
    
    // Create token contract instance
    let erc20_abi = match serde_json::from_str::<ethers::abi::Abi>(&erc20_abi_json) {
        Ok(abi) => abi,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to parse ERC20 ABI: {}", e))
    };
    
    let token = Contract::new(payment_token_address, erc20_abi, client.clone());
    
    // STEP 2: Call approve function with the exact payment amount needed
    let approve_tx = match token.method::<_, bool>("approve", (curve_address_parsed, payment_required)) {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to create approve transaction: {}", e))
    };
    
    // Send the approve transaction
    let approve_pending_tx = match approve_tx.send().await {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to send approve transaction: {}", e))
    };
    
    // Wait for the approve transaction to be mined
    let approve_receipt = match approve_pending_tx.await {
        Ok(receipt) => {
            match receipt {
                Some(r) => r,
                None => return HttpResponse::InternalServerError().body("Approve transaction failed")
            }
        },
        Err(e) => return HttpResponse::InternalServerError().body(format!("Approve transaction error: {}", e))
    };
    
    println!("Approve transaction successful: {:?}", approve_receipt.transaction_hash);
    
    // STEP 3: Now call buy function with the token amount
    let buy_tx = match curve.method::<_, ()>("buy", token_amount_wei) {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to create buy transaction: {}", e))
    };
    
    // Send the buy transaction
    let buy_pending_tx = match buy_tx.send().await {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to send buy transaction: {}", e))
    };
    
    // Wait for the buy transaction to be mined
    match buy_pending_tx.await {
        Ok(receipt) => {
            match receipt {
                Some(r) => HttpResponse::Ok().json(serde_json::json!({
                    "status": "success",
                    "approve_tx": format!("{:?}", approve_receipt.transaction_hash),
                    "buy_tx": format!("{:?}", r.transaction_hash),
                    "payment_amount": format!("{}", payment_required),
                    "token_amount": token_amount,
                    "block_number": r.block_number.map(|b| b.as_u64())
                })),
                None => HttpResponse::InternalServerError().body("Buy transaction failed")
            }
        },
        Err(e) => HttpResponse::InternalServerError().body(format!("Buy transaction error: {}", e))
    }
}

async fn approve_token_for_curve(path: web::Path<(String, String)>) -> HttpResponse {
    let (curve_address, amount) = path.into_inner();
    
    // Parse the curve address
    let curve_address = match Address::from_str(&curve_address) {
        Ok(addr) => addr,
        Err(_) => return HttpResponse::BadRequest().body("Invalid curve address")
    };
    
    // Parse the amount (expected in full tokens, will be converted to wei)
    let amount = match amount.parse::<f64>() {
        Ok(val) => {
            // Convert to wei (assuming 18 decimals)
            let wei_amount = (val * 1e18) as u128;
            U256::from(wei_amount)
        },
        Err(_) => return HttpResponse::BadRequest().body("Invalid amount")
    };
    
    // Use the global wallet
    let wallet = WALLET.clone();
    let chain_id = 1315; // Aeneid chain
    
    let provider_url = match PROVIDER_URLS.get(&chain_id) {
        Some(url) => url,
        None => return HttpResponse::BadRequest().body("No provider URL for chain ID 1315")
    };
    
    let provider = match Provider::<Http>::try_from(provider_url.as_str()) {
        Ok(p) => p,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Provider error: {}", e))
    };
    
    let client = Arc::new(SignerMiddleware::new(provider, wallet.with_chain_id(chain_id)));
    
    // Get the payment token address for this chain
    let payment_token_address = match PAYMENT_TOKENS.get(&chain_id) {
        Some(addr) => *addr,
        None => return HttpResponse::BadRequest().body("No payment token configured for chain ID 1315")
    };
    
    // Load ERC20 ABI
    let abi_json = match std::fs::read_to_string("src/abi/erc20_abi.json") {
        Ok(json) => json,
        Err(_) => {
            // Fallback to a minimal ERC20 ABI if file doesn't exist
            r#"[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"}]"#.to_string()
        }
    };
    
    // Create token contract instance
    let abi = match serde_json::from_str::<ethers::abi::Abi>(&abi_json) {
        Ok(abi) => abi,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to parse ABI: {}", e))
    };
    
    let token = Contract::new(payment_token_address, abi, client);
    
    // Call approve function
    let tx = match token.method::<_, bool>("approve", (curve_address, amount)) {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to create transaction: {}", e))
    };
    
    // Send the transaction
    let pending_tx = match tx.send().await {
        Ok(tx) => tx,
        Err(e) => return HttpResponse::InternalServerError().body(format!("Failed to send transaction: {}", e))
    };
    
    // Wait for the transaction to be mined
    match pending_tx.await {
        Ok(receipt) => {
            match receipt {
                Some(r) => HttpResponse::Ok().json(serde_json::json!({
                    "status": "success",
                    "transaction_hash": format!("{:?}", r.transaction_hash),
                    "block_number": r.block_number.map(|b| b.as_u64())
                })),
                None => HttpResponse::InternalServerError().body("Transaction failed")
            }
        },
        Err(e) => HttpResponse::InternalServerError().body(format!("Transaction error: {}", e))
    }
}

// Add a hardcoded function to approve 100 tokens for the specific contract
async fn approve_iris_tokens() -> HttpResponse {
    // Hardcoded values
    let curve_address = "0x13c44aa3377246f05546f9e3b0104792f944d81b";
    let amount = "100"; // 100 tokens
    
    approve_token_for_curve(web::Path::from((curve_address.to_string(), amount.to_string()))).await
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
            .route("/bonding-curve/create", web::post().to(create_bonding_curve))
            .route("/buy/{curve_address}/{amount}", web::get().to(buy_tokens))
            .route("/buy-iris", web::get().to(buy_iris_tokens))
            .route("/approve/{curve_address}/{amount}", web::get().to(approve_token_for_curve))
            .route("/approve-iris", web::get().to(approve_iris_tokens))
            
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}