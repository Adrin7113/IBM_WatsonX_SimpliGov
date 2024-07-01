const sendResponse = require("../utils/sendResponse");
const generateChatReply = require("../utils/generateChatReply");



const handleGenerateChatReplyRequest = async (req, res) => {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk.toString();
  });
  req.on("end", async () => {
    try {
      const data = JSON.parse(body);
      const message = data.message;

      console.log("message",message);
      try {
        const output = await generateChatReply(message);
        console.log("output",output);
        sendResponse(
          res,
          200,
          "application/json",
          JSON.stringify({ answer: output })
        );
      } catch (err) {
        sendResponse(
          res,
          500,
          "text/plain",
          `Error executing Python script: ${err.message}\n`
        );
      }
    } catch (err) {
      sendResponse(res, 400, "text/plain", "Invalid JSON\n");
    }
  });
};

module.exports = handleGenerateChatReplyRequest;
