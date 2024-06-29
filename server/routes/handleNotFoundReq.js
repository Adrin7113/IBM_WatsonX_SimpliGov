const handleNotFoundRequest = (req, res) => {
  sendResponse(res, 404, "text/plain", "Not Found\n");
};

module.exports = handleNotFoundRequest;
