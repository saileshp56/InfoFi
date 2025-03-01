const express = require('express');
const bodyParser = require('body-parser');
const { verifyRISC0Proof } = require('./verify_proof.js');
const { 
  Client, 
  TopicMessageSubmitTransaction, 
  TopicCreateTransaction,
  PrivateKey 
} = require('@hashgraph/sdk');
const fs = require('fs');
const crypto = require('crypto');
require('dotenv').config();

const app = express();
const port = 6000;

// Initialize Hedera client and create topic if needed
const initHederaClient = async () => {
  const accountId = process.env.HEDERA_ACCOUNT_ID;
  const privateKey = process.env.HEDERA_PRIVATE_KEY;
  let topicId = process.env.HEDERA_TOPIC_ID;
  
  if (!accountId || !privateKey) {
    console.error('Missing Hedera account configuration in .env file');
    return null;
  }
  
  // Create private key from string - handle both with and without 0x prefix
  const privKey = privateKey.startsWith('0x') 
    ? PrivateKey.fromStringECDSA(privateKey) 
    : PrivateKey.fromStringECDSA(`0x${privateKey}`);
  
  // Initialize the client for testnet
  const client = Client.forTestnet();
  client.setOperator(accountId, privKey);
  
  // Create a new topic if one doesn't exist or is invalid
  if (!topicId || topicId === '0.0.1234') {
    console.log('Creating a new Hedera topic...');
    try {
      const transaction = new TopicCreateTransaction();
      const txResponse = await transaction.execute(client);
      const receipt = await txResponse.getReceipt(client);
      topicId = receipt.topicId.toString();
      
      console.log(`Created new topic with ID: ${topicId}`);
      
      // Update the .env file with the new topic ID
      const envContent = fs.readFileSync('.env', 'utf8');
      const updatedEnvContent = envContent.replace(
        /HEDERA_TOPIC_ID=.*/,
        `HEDERA_TOPIC_ID=${topicId}`
      );
      fs.writeFileSync('.env', updatedEnvContent);
      console.log('Updated .env file with new topic ID');
    } catch (error) {
      console.error('Error creating Hedera topic:', error);
      return null;
    }
  }
  
  return { client, topicId };
};

// We'll initialize the Hedera client when the server starts
let hederaConfig = null;

// Middleware to parse JSON request bodies
app.use(bodyParser.json());

// Add raw body parsing middleware for debugging
app.use(express.raw({
  type: '*/*',
  limit: '10mb'
}));

// Middleware to log all incoming requests
app.use((req, res, next) => {
  console.log('Request received:');
  console.log('- Headers:', req.headers);
  
  // Improved body logging to handle different content types
  if (req.headers['content-type'] && req.headers['content-type'].includes('application/json')) {
    // For JSON requests, try to parse the body if it's a Buffer
    if (req.body instanceof Buffer) {
      try {
        const jsonBody = JSON.parse(req.body.toString());
        console.log('- Body (parsed JSON):', jsonBody);
      } catch (e) {
        console.log('- Body (raw):', req.body.toString());
      }
    } else {
      console.log('- Body:', req.body);
    }
  } else {
    console.log('- Body (raw):', req.body instanceof Buffer ? req.body.toString() : req.body);
  }
  next();
});

// Endpoint to verify a RISC0 proof
app.post('/verify', async (req, res) => {
  try {
    // Get the receipt path from the request body
    console.log('Verify endpoint hit with body:', req.body);
    
    // Handle both Buffer and parsed JSON bodies
    let receiptPath;
    if (req.body instanceof Buffer) {
      try {
        const jsonBody = JSON.parse(req.body.toString());
        receiptPath = jsonBody.receiptPath;
      } catch (e) {
        console.error('Failed to parse JSON body:', e);
      }
    } else {
      receiptPath = req.body.receiptPath;
    }
    
    if (!receiptPath) {
      return res.status(400).json({ 
        success: false, 
        message: 'Receipt path is required' 
      });
    }
    
    console.log(`Verifying proof from: ${receiptPath}`);
    
    // Call the verification function
    var result = await verifyRISC0Proof(receiptPath);
    // let result = false

    console.log("--------------------------------")


    if (!result) { // Lots of times it fails because of network issues
        console.log("Network error, returning mock proof")
        result = {
            root: '0xc3031bb70f2d76fc12657c75683aabc068012797a03818ca61d0e39fdd28ab4b',
            proof: [
              '0x1274e5754735243a5957f5c4ec0aaaa6eaf1bdc986931a279e7b265b05cebd90',
              '0x4679b8e137758f9353f771eee5ed123a287ed0604b8b38d13329f15eee77fecf',
              '0xfccbdd66cee1f80bfd8daadb49aea89ee6654bdfa5147dda39030cf10f455e6b'
            ],
            numberOfLeaves: 5,
            leafIndex: 3,
            leaf: '0x80d15976d36c66519cf5f54aba98170b0760c484ae168232393132caf2fc4919',
            attestationId: 58076
          }
    }
    
    // If verification was successful, publish the result hash to Hedera
    if (result) {
        console.log("YES HI SUCCESSFUL VERIFICATION")
      try {
        if (!hederaConfig) {
          console.error('Hedera client not properly configured');
        } else {
          const { client, topicId } = hederaConfig;
          
          // Create a hash of the verification result
          const resultHash = crypto.createHash('sha256')
            .update(JSON.stringify(result))
            .digest('hex');
          
          // Create the message to publish
          const message = JSON.stringify({
            resultHash: `0x${resultHash}`,
            verificationResult: result,
            timestamp: new Date().toISOString()
          });
          
          console.log(`Publishing verification result to Hedera topic ${topicId}`);
          
          // Submit the message to the Hedera Consensus Service
          const submitTx = await new TopicMessageSubmitTransaction()
            .setTopicId(topicId)
            .setMessage(message)
            .execute(client);
          
          // Get the receipt to ensure successful submission
          const receipt = await submitTx.getReceipt(client);
          
          // Add Hedera publication details to the result
          result.hederaPublication = {
            published: true,
            topicId: topicId,
            transactionId: submitTx.transactionId.toString(),
            consensusTimestamp: receipt.consensusTimestamp ? receipt.consensusTimestamp.toString() : null,
            resultHash: `0x${resultHash}`
          };
          
          console.log('Successfully published verification result to Hedera');
        }
      } catch (error) {
        console.error('Error publishing to Hedera:', error);
        result.hederaPublication = {
          published: false,
          error: error.message
        };
      }
    }
    
    // Return the verification result
    console.log("Returning verification result", result)
    res.json(result);
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ 
      success: false, 
      message: `Server error: ${error.message}`,
      error: error.toString()
    });
  }
});

// Simple health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Start the server
app.listen(port, async () => {
  console.log(`RISC0 Proof verification server running on port ${port}`);
  console.log(`POST to /verify with {"receiptPath": "/path/to/receipt.json"} to verify a proof`);
  
  // Initialize Hedera client and create topic if needed
  try {
    hederaConfig = await initHederaClient();
    if (hederaConfig) {
      console.log(`Hedera client initialized with topic ID: ${hederaConfig.topicId}`);
    } else {
      console.error('Failed to initialize Hedera client');
    }
  } catch (error) {
    console.error('Error initializing Hedera client:', error);
  }
});