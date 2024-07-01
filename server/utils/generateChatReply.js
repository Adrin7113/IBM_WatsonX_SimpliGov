const { MODEL_PATH } = require("../data/global");

async function generateChatReply(message) {
  return new Promise((resolve, reject) => {
    const python = spawn("python3.12.exe", [MODEL_PATH, "--query", message]);
    let pythonOutput = "";

    python.stdout.on("data", (data) => {
      pythonOutput += data.toString();
    });

    python.stderr.on("data", (data) => {
      console.error(`stderr: ${data}`);
    });

    python.on("close", (code) => {
      console.log(`child process exited with code ${code}`);
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}`));
      } else {
        resolve(pythonOutput);
      }
    });
  });
}

module.exports = generateChatReply;