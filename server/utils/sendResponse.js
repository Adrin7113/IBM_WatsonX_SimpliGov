const { headers } = require("../data/global");

function sendResponse(res, statusCode, contentType, message) {
  res.writeHead(statusCode, {
    ...headers,
    "Content-Type": contentType,
  });
  res.end(message);
}

module.exports = sendResponse;
