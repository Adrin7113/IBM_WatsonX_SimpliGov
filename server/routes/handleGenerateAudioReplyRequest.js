const fs = require("fs");
const sendResponse = require("../utils/sendResponse");

function handleGenerateAudioReplyRequest(req, res) {
  // Generate a unique id for the audio file
  const audioId = Date.now() + "_" + Math.random().toString(36).substr(2, 9);

  // Check if the 'audioStore' directory exists, if not, create it
  if (!fs.existsSync("./audioStore")) {
    fs.mkdirSync("./audioStore");
  }

  let body = [];
  req
    .on("data", (chunk) => {
      body.push(chunk);
    })
    .on("end", () => {
      // Decode the base64 audio
      const audioData = Buffer.from(body.join(""), "base64");

      // Write the decoded audio data to a file
      fs.writeFile(`./audioStore/${audioId}.wav`, audioData, (err) => {
        if (err) {
          console.error(err);
          sendResponse(res, 500, "text/plain", "Error saving audio file");
        } else {
          console.log(`Audio file saved with id: ${audioId}`);
          sendResponse(
            res,
            200,
            "text/plain",
            `Audio file saved with id: ${audioId}`
          );
        }
      });
    });
}

module.exports = handleGenerateAudioReplyRequest;
