const tf = require("@tensorflow/tfjs-node");
const moment = require("moment");
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Environment parameters
const numActions = 501; // Reduced number of possible actions (predictions ranging from 0 to 500)
const numStates = 1001; // Possible sums ranging from 0 to 1000

// Q-table
let qTable = tf.buffer([numStates, numActions]);

// Hyperparameters
const learningRate = 0.1;
const discountFactor = 0.95;
let epsilon = 1.0; // Initial epsilon for exploration
const epsilonDecay = 0.995; // Epsilon decay rate
const minEpsilon = 0.01; // Minimum epsilon value

// Generate a random pair of numbers
function generatePair() {
  const a = Math.floor(Math.random() * 500); // Adjusted range
  const b = Math.floor(Math.random() * 500); // Adjusted range
  return { a, b };
}

// Get reward
function getReward(actualSum, predictedSum) {
  return -Math.abs(actualSum - predictedSum); // Negative absolute error
}

// Epsilon-greedy policy
function chooseAction(state, epsilon) {
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * numActions); // Random action
  } else {
    return qTable.toTensor().gather(state).argMax().arraySync(); // Best action
  }
}

// Update Q-table
function updateQTable(state, action, reward, nextState) {
  const currentQ = qTable.get(state, action);
  const maxNextQ = qTable.toTensor().gather(nextState).max().arraySync();
  const newQ =
    currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);

  // Update the Q-table
  qTable.set(newQ, state, action);
}

// Training loop
async function trainAgent(episodes) {
  for (let episode = 0; episode < episodes; episode++) {
    const { a, b } = generatePair();
    const state = a + b;
    const action = chooseAction(state, epsilon);
    const predictedSum = action; // Action represents predicted sum
    const reward = getReward(state, predictedSum);
    const nextState = state; // In this simple case, next state is the same as the current state

    updateQTable(state, action, reward, nextState);

    if (episode % 1000 === 0) {
      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      console.log(`Episode ${episode}, Reward: ${reward}`);
      epsilon = Math.max(minEpsilon, epsilon * epsilonDecay); // Decay epsilon
    }
  }
  console.log("Training complete.");
}

// Testing function
function testAgent(a, b) {
  const state = a + b;
  const action = qTable.toTensor().gather(state).argMax().arraySync();
  console.log(`Predicted sum for ${a} + ${b} = ${action}`);
  promptUser();
}

// Prompt user for input
function promptUser() {
  rl.question(
    'Enter two numbers separated by a comma to test the agent (or type "exit" to quit): ',
    (answer) => {
      if (answer.toLowerCase() === "exit") {
        rl.close();
        console.log("Exiting...");
      } else {
        const testInputs = answer.split(",").map(Number);
        if (testInputs.length === 2 && testInputs.every((num) => !isNaN(num))) {
          testAgent(testInputs[0], testInputs[1]);
        } else {
          console.log(
            "Invalid input. Please enter two numbers separated by a comma."
          );
          promptUser();
        }
      }
    }
  );
}

// Start training
async function main() {
  const episodes = parseInt(process.argv[2], 10);
  if (isNaN(episodes)) {
    console.error("Invalid number of episodes. Please provide a valid number.");
    process.exit(1);
  }

  const startMs = Date.now();
  console.clear();
  console.log(
    "Agent Training started at " + moment().format("DD/MM/YYYY hh:mm:ss")
  );
  console.log(
    `Training reinforcement learning agent for ${episodes} episodes...`
  );
  await trainAgent(episodes);
  console.log(
    "Agent Training ended at " + moment().format("DD/MM/YYYY hh:mm:ss")
  );
  console.log(
    `Agent Training lasted for ${parseMillisecondsIntoReadableTime(
      Date.now() - startMs
    )}`
  );
  promptUser();
}

function parseMillisecondsIntoReadableTime(milliseconds = 0) {
  const hours = Math.floor(milliseconds / (1000 * 60 * 60));
  const minutes = Math.floor((milliseconds % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((milliseconds % (1000 * 60)) / 1000);
  return `${hours} hours ${minutes} minutes ${seconds} seconds`;
}

main();
