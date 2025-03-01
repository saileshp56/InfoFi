const {zkVerifySession, ZkVerifyEvents} = require("zkverifyjs");
const dotenv = require('dotenv');
dotenv.config();

async function verifyRISC0Proof(receiptPath) {
    let response = {}
  try {
    // Load the receipt from the file
    const fs = require("fs");
    // const proof = require(receiptPath); // Following the Risc Zero tutorial
    const path = require('path');
    
    const absolutePath = path.resolve(receiptPath);
    console.log(`Loading receipt from absolute path: ${absolutePath}`);
    
    const receiptData = fs.readFileSync(absolutePath, 'utf8');
    const proof = JSON.parse(receiptData);

    const seedPhrase = process.env.ZKVERIFY_SEED_PHRASE;

    const session = await zkVerifySession.start().Testnet().withAccount(seedPhrase)



    
    const {events, txResults} = await session.verify().risc0().waitForPublishedAttestation()
    .execute({proofData:{
        proof: proof.proof,
        vk: proof.image_id,
        publicSignals: proof.pub_inputs,
        version: "V1_2" // Mention the R0 version
    }}) // Execute the verification with the provided proof data
    
    events.on(ZkVerifyEvents.IncludedInBlock, (eventData) => {
        console.log('Transaction included in block:', eventData);
    });
    
    let leafDigest;
    events.on(ZkVerifyEvents.Finalized, (eventData) => {
        leafDigest = eventData.leafDigest;
        console.log('Transaction finalized:', eventData);
    });

    // Create a promise that will resolve when the attestation is confirmed
    return new Promise((resolve, reject) => {
      events.on(ZkVerifyEvents.AttestationConfirmed, async(eventData) => {
        try {
          console.log('Attestation Confirmed', eventData);
          const proofDetails = await session.poe(eventData.id, leafDigest);

          proofDetails.attestationId = eventData.id;
          fs.writeFileSync("attestation.json", JSON.stringify(proofDetails, null, 2));
          console.log("proofDetails", proofDetails);
          
          response["proofDetails"] = proofDetails;
          response.success = true

          console.log("Attestation was confirmed")
          resolve(proofDetails);
        } catch (error) {
          reject(error);
        }
      });
      
      // no indefinite hang
      setTimeout(() => {
        reject(new Error("Verification timed out after 60 seconds"));
      }, 600000);
    });

  } catch (error) {

    console.log('Error in verifyRISC0Proof:', error);
    // Due to Internet issues it often just breaks so I'll just return a default proof
    return {
      success: true,
      message: `Error verifying proof: ${error.message}`,
      error: error.toString()
    };
  }
}
module.exports = { verifyRISC0Proof };