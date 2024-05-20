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
  await model.fit(xs, ys, {
    epochs: 500,
  });
  console.log("Initial model training complete");
  promptUser();
}

function testModel(input1, input2) {
  const prediction = model.predict(tf.tensor2d([[input1, input2]], [1, 2]));
  prediction.print();
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
  await model.fit(xs, ys, {
    epochs: 100,
  });
  console.log("Model retraining complete");
}

trainModel();
