process.env.TF_ENABLE_ONEDNN_OPTS = "0";
process.env.TF_CPP_MIN_LOG_LEVEL = "2";

const tf = require("@tensorflow/tfjs-node");
const moment = require("moment");
const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Generate a batch of pairs of numbers and their sums
function* dataGenerator(numBatches, batchSize) {
  for (let i = 0; i < numBatches; i++) {
    const xs = [];
    const ys = [];
    for (let j = 0; j < batchSize; j++) {
      const a = Math.random() * 1000;
      const b = Math.random() * 1000;
      xs.push([a, b]);
      ys.push([a + b]);
    }
    yield {
      xs: tf.tensor2d(xs),
      ys: tf.tensor2d(ys, [batchSize, 1]),
    };
  }
}

// Normalize the data
function normalizeData(xs, ys) {
  const inputMax = xs.max(0);
  const inputMin = xs.min(0);
  const labelMax = ys.max();
  const labelMin = ys.min();

  const inputRange = inputMax.sub(inputMin);
  const labelRange = labelMax.sub(labelMin);

  const xsNorm = inputRange.equal(0).any().dataSync()[0]
    ? xs
    : xs.sub(inputMin).div(inputRange);
  const ysNorm = labelRange.equal(0).any().dataSync()[0]
    ? ys
    : ys.sub(labelMin).div(labelRange);

  return { xsNorm, ysNorm, inputMax, inputMin, labelMax, labelMin };
}

// Define the model
const model = tf.sequential();
model.add(
  tf.layers.dense({
    units: 64, // Increased units for better learning
    inputShape: [2],
    activation: "relu",
    kernelInitializer: "heNormal",
  })
);
model.add(
  tf.layers.dense({
    units: 1,
    activation: "linear",
    kernelInitializer: "heNormal",
  })
);

// Compile the model with a lower learning rate
model.compile({
  optimizer: tf.train.adam(0.001),
  loss: "meanSquaredError",
});

async function trainModel(numSamples) {
  const batchSize = 32;
  const numBatches = Math.ceil(numSamples / batchSize);

  const dataset = tf.data
    .generator(() => dataGenerator(numBatches, batchSize))
    .map(({ xs, ys }) => {
      const { xsNorm, ysNorm } = normalizeData(xs, ys);
      return { xs: xsNorm, ys: ysNorm };
    })
    .repeat();

  await model.fitDataset(dataset, {
    epochs: 200, // Increased epochs for better training
    batchesPerEpoch: numBatches,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (isNaN(logs.loss)) {
          throw new Error("Training stopped due to NaN loss");
        }
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}`);
      },
    },
  });

  console.log("Initial model training complete");
}

function testModel(input1, input2, inputMin, inputMax, labelMin, labelMax) {
  const inputTensor = tf.tensor2d([[input1, input2]], [1, 2]);
  const inputRange = inputMax.sub(inputMin);
  const labelRange = labelMax.sub(labelMin);

  const normalizedInput = inputRange.equal(0).any().dataSync()[0]
    ? inputTensor
    : inputTensor.sub(inputMin).div(inputRange);
  const prediction = model.predict(normalizedInput);
  const denormalizedPrediction = labelRange.equal(0).any().dataSync()[0]
    ? prediction
    : prediction.mul(labelRange).add(labelMin);

  denormalizedPrediction.array().then((arr) => {
    console.log(`Prediction: ${arr[0][0]}`);
    promptUser();
  });
}

function promptUser() {
  rl.question(
    'Enter two numbers separated by a comma to test the model (or type "exit" to quit): ',
    (answer) => {
      if (answer.toLowerCase() === "exit") {
        rl.close();
        console.log("Exiting...");
      } else {
        const testInputs = answer.split(",").map(Number);
        if (testInputs.length === 2 && testInputs.every((num) => !isNaN(num))) {
          const xs = tf.tensor2d([testInputs]);
          const ys = tf.tensor2d([[testInputs[0] + testInputs[1]]]);
          const { inputMax, inputMin, labelMax, labelMin } = normalizeData(
            xs,
            ys
          );
          testModel(
            testInputs[0],
            testInputs[1],
            inputMin,
            inputMax,
            labelMin,
            labelMax
          );
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

function parseMillisecondsIntoReadableTime(milliseconds = 0) {
  const hours = Math.floor(milliseconds / (1000 * 60 * 60));
  const minutes = Math.floor((milliseconds % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((milliseconds % (1000 * 60)) / 1000);
  return `${hours} hours ${minutes} minutes ${seconds} seconds`;
}

async function main() {
  const datasetLength = parseInt(process.argv[2], 10);
  if (isNaN(datasetLength)) {
    console.error("Invalid dataset length. Please provide a valid number.");
    process.exit(1);
  }

  const startMs = Date.now();
  console.clear();
  console.log(
    "Model Training started at " + moment().format("DD/MM/YYYY hh:mm:ss")
  );
  console.log(`Training model on ${datasetLength} samples...`);
  readline.clearLine(process.stdout, 0);
  await trainModel(datasetLength);
  console.log(
    "Model Training ended at " + moment().format("DD/MM/YYYY hh:mm:ss")
  );
  console.log(
    `Model Training lasted for ${parseMillisecondsIntoReadableTime(
      Date.now() - startMs
    )}`
  );
  promptUser();
}

main();
