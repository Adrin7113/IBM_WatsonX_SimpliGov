const sendResponse = require("../utils/sendResponse");

const handleRootRequest = (req, res) => {
  sendResponse(res, 200, "text/plain", "Hello, World!\n");
};

module.exports = handleRootRequest;
