const MODEL_PATH = "../model/main.py";

const PORT = process.env.PORT || 3000;

const headers = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "OPTIONS, POST, GET",
  "Access-Control-Allow-Headers": "Content-Type",
};

module.exports = {
  MODEL_PATH,
  PORT,
  headers,
};
