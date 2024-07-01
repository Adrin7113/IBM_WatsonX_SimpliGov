const http = require("http");
const url = require("url");
const { PORT } = require("./data/global");
const handleGenerateChatReplyRequest = require("./routes/handleGenerateChatReplyReq");
const handleRootRequest = require("./routes/handleRootReq");
const handleOptionsRequest = require("./routes/handleOptionsReq");
const handleNotFoundRequest = require("./routes/handleNotFoundReq");
const handleGenerateAudioReplyRequest = require("./routes/handleGenerateAudioReplyRequest");
const translate = require("./routes/translate");

const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url);
  const pathname = parsedUrl.pathname;

  // Handle OPTIONS request for CORS preflight request from browser
  if (req.method === "OPTIONS") {
    handleOptionsRequest(req, res);
    return;
  }

  // Root route
  if (pathname === "/") {
    handleRootRequest(req, res);
  }

  // Chat reply route
  else if (pathname === "/generateChatReply" && req.method === "POST") {
    handleGenerateChatReplyRequest(req, res);
  }

  // Audio reply route
  else if (pathname === "/generateAudioReply" && req.method === "POST") {
    handleGenerateAudioReplyRequest(req, res);
  }

  else if (pathname === "/translateText" && req.method === "POST") {
    translate(req, res);
  }

  // Not found route
  else {
    handleNotFoundRequest(req, res);
  }
});

server.listen(PORT, () => {
  console.log(`Server is running on port: ${PORT}`);
});
