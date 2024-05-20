process.env.TF_ENABLE_ONEDNN_OPTS = "0";
process.env.TF_CPP_MIN_LOG_LEVEL = "2";

const tf = require("@tensorflow/tfjs-node");
const moment = require("moment");
const readline = require("readline");

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
    units: 1,
    inputShape: [2],
    kernelInitializer: "heNormal",
  })
);

// Compile the model with a lower learning rate
model.compile({
  optimizer: tf.train.sgd(0.001),
  loss: "meanSquaredError",
});

async function trainModel(numSamples) {
  const batchSize = 1024;
  const numBatches = Math.ceil(numSamples / batchSize);

  const dataset = tf.data
    .generator(() => dataGenerator(numBatches, batchSize))
    .map(({ xs, ys }) => {
      const { xsNorm, ysNorm } = normalizeData(xs, ys);
      return { xs: xsNorm, ys: ysNorm };
    })
    .repeat(); // Ensure the dataset can provide enough data for all epochs

  await model.fitDataset(dataset, {
    epochs: 10,
    batchesPerEpoch: numBatches,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (isNaN(logs.loss)) {
          throw new Error("Training stopped due to NaN loss");
        }
      },
    },
  });

  console.log("Initial model training complete");
  promptUser();
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
    console.log(arr[0][0]); // Output the prediction as a single number
    promptUser();
  });
}

function promptUser() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.question(
    'Enter "test" to test the model or "exit" to quit: ',
    (answer) => {
      if (answer.toLowerCase() === "test") {
        rl.question(
          'Enter two numbers separated by a comma to test (e.g., "9,10"): ',
          (testInputAnswer) => {
            const testInputs = testInputAnswer.split(",").map(Number);
            if (
              testInputs.length === 2 &&
              testInputs.every((num) => !isNaN(num))
            ) {
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
              rl.close();
              promptUser();
            }
          }
        );
      } else if (answer.toLowerCase() === "exit") {
        rl.close();
        console.log("Exiting...");
      } else {
        console.log('Invalid command. Please enter "test" or "exit".');
        rl.close();
        promptUser();
      }
    }
  );
}

function parseMillisecondsIntoReadableTime(milliseconds = 0) {
  //Get hours from milliseconds
  var hours = milliseconds / (1000 * 60 * 60);
  var absoluteHours = Math.floor(hours);
  var h = absoluteHours > 9 ? absoluteHours : "0" + absoluteHours;

  //Get remainder from hours and convert to minutes
  var minutes = (hours - absoluteHours) * 60;
  var absoluteMinutes = Math.floor(minutes);
  var m = absoluteMinutes > 9 ? absoluteMinutes : "0" + absoluteMinutes;

  //Get remainder from minutes and convert to seconds
  var seconds = (minutes - absoluteMinutes) * 60;
  var absoluteSeconds = Math.floor(seconds);
  var s = absoluteSeconds > 9 ? absoluteSeconds : "0" + absoluteSeconds;

  return h + "h " + m + "m " + s + "s";
}

async function main() {
  const startMs = Date.now();
  console.log("Model Training started at " + moment().format("DD:MM:YYYY"));
  console.log("Training model on 100 million samples...");
  await trainModel(100000000); // Train with 100 million samples
  console.log("Model Training ended at " + moment().format("DD:MM:YYYY"));
  console.log(
    "Model Training lasted for " +
      parseMillisecondsIntoReadableTime(Date.now() - startMs) / 3600 +
      " "
  );
}

main();
