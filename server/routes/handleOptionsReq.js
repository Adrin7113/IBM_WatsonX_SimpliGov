const sendResponse = require("../utils/sendResponse");

const handleOptionsRequest = (req, res) => {
  sendResponse(res, 204, "text/plain", "");
};

module.exports = handleOptionsRequest;
