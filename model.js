const tf = require("@tensorflow/tfjs");
const readline = require("readline");

// Initial synthetic data for training
let xs = tf.tensor2d(
  [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
  ],
  [8, 2]
);

let ys = tf.tensor2d([[3], [5], [7], [9], [11], [13], [15], [17]], [8, 1]);

// Normalize the data
let { xsNorm, ysNorm, inputMax, inputMin, labelMax, labelMin } = normalizeData(
  xs,
  ys
);

function normalizeData(xs, ys) {
  const inputMax = xs.max(0);
  const inputMin = xs.min(0);
  const labelMax = ys.max();
  const labelMin = ys.min();

  const xsNorm = xs.sub(inputMin).div(inputMax.sub(inputMin));
  const ysNorm = ys.sub(labelMin).div(labelMax.sub(labelMin));

  return { xsNorm, ysNorm, inputMax, inputMin, labelMax, labelMin };
}

// Define the model
const model = tf.sequential();
model.add(
  tf.layers.dense({
    units: 1,
    inputShape: [2],
  })
);

// Compile the model with a lower learning rate
model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: "meanSquaredError",
});

async function trainModel() {
  await model.fit(xsNorm, ysNorm, {
    epochs: 500,
  });
  console.log("Initial model training complete");
  promptUser();
}

function testModel(input1, input2) {
  const inputTensor = tf.tensor2d([[input1, input2]], [1, 2]);
  const normalizedInput = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
  const prediction = model.predict(normalizedInput);
  const denormalizedPrediction = prediction
    .mul(labelMax.sub(labelMin))
    .add(labelMin);
  denormalizedPrediction.print();
}

function promptUser() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.question(
    'Enter "train" to provide new data, "test" to test the model, or "exit" to quit: ',
    (answer) => {
      if (answer.toLowerCase() === "train") {
        rl.question(
          'Enter two numbers separated by a comma for input (e.g., "9,10"): ',
          (inputAnswer) => {
            const inputs = inputAnswer.split(",").map(Number);
            if (inputs.length === 2 && inputs.every((num) => !isNaN(num))) {
              rl.question(
                "Enter the expected output for these inputs: ",
                (outputAnswer) => {
                  const output = parseFloat(outputAnswer);
                  if (!isNaN(output)) {
                    // Update the dataset with the new input-output pair
                    const newX = tf.tensor2d([inputs], [1, 2]);
                    const newY = tf.tensor2d([[output]], [1, 1]);
                    xs = xs.concat(newX);
                    ys = ys.concat(newY);

                    const normalizedData = normalizeData(xs, ys);
                    xsNorm = normalizedData.xsNorm;
                    ysNorm = normalizedData.ysNorm;
                    inputMax = normalizedData.inputMax;
                    inputMin = normalizedData.inputMin;
                    labelMax = normalizedData.labelMax;
                    labelMin = normalizedData.labelMin;

                    // Retrain the model with the updated dataset
                    retrainModel().then(() => {
                      rl.close();
                      promptUser();
                    });
                  } else {
                    console.log("Invalid output. Please enter a valid number.");
                    rl.close();
                    promptUser();
                  }
                }
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
      } else if (answer.toLowerCase() === "test") {
        rl.question(
          'Enter two numbers separated by a comma to test (e.g., "9,10"): ',
          (testInputAnswer) => {
            const testInputs = testInputAnswer.split(",").map(Number);
            if (
              testInputs.length === 2 &&
              testInputs.every((num) => !isNaN(num))
            ) {
              testModel(testInputs[0], testInputs[1]);
            } else {
              console.log(
                "Invalid input. Please enter two numbers separated by a comma."
              );
            }
            rl.close();
            promptUser();
          }
        );
      } else if (answer.toLowerCase() === "exit") {
        rl.close();
        console.log("Exiting...");
      } else {
        console.log(
          'Invalid command. Please enter "train", "test", or "exit".'
        );
        rl.close();
        promptUser();
      }
    }
  );
}

async function retrainModel() {
  console.log("Retraining model with new data...");
  await model.fit(xsNorm, ysNorm, {
    epochs: 100,
  });
  console.log("Model retraining complete");
}

trainModel();
